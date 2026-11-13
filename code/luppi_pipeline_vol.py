"""Volumetric preprocessing of the HCP minimally preprocessed data following Luppi et al. (2026) (Section 2.3), and its comparison
with the authors' distributed example time series.
Usage: python luppi_pipeline_vol.py validate OUT_JSON"""
from paths import DATA_ROOT
import sys, json, itertools
import numpy as np

RAW = DATA_ROOT + "raw/p33_hcp_vol/"; TR = 0.72; DROP = 11
ATLAS = RAW + "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
WM_LABELS = [2, 41, 5001, 5002] + list(range(3000, 3036)) + list(range(4000, 4036))     # cerebral WM incl. wmparc subdivisions
ASEG_WM = [2, 41, 77, 251, 252, 253, 254, 255]; ASEG_CSF = [4, 5, 14, 15, 24, 43, 44]          # subject aparc+aseg (0.7 mm)
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44]                                                 # lateral / 3rd / 4th ventricles, CSF


def load_subject(s):
    import nibabel as nib
    img = nib.load(RAW + f"{s}_rfMRI_REST1_LR.nii.gz"); X = np.asarray(img.dataobj, dtype=np.float32)       # (91, 109, 91, T)
    lab = np.asarray(nib.load(RAW + f"{s}_Atlas_wmparc.2.nii.gz").dataobj).astype(int)
    bm = np.asarray(nib.load(RAW + f"{s}_brainmask_fs.2.nii.gz").dataobj) > 0
    atl = np.asarray(nib.load(ATLAS).dataobj).astype(int)
    mot = np.loadtxt(RAW + f"{s}_Movement_Regressors.txt")
    if X.shape[:3] != lab.shape or lab.shape != atl.shape:
        raise SystemExit(f"{s}: grid mismatch {X.shape} {lab.shape} {atl.shape}")
    return X, lab, bm, atl, mot


def erode(m, n):
    from scipy import ndimage
    return ndimage.binary_erosion(m, iterations=n) if n else m


def comps(V, kind, k=5):
    """V: (T, nvox) voxel signals -> (T, k) confound components."""
    V = V - V.mean(0); V = V / (V.std(0) + 1e-9); t = np.arange(V.shape[0]); D = np.c_[np.ones_like(t), t].astype(float)
    V = V - D @ np.linalg.lstsq(D, V, rcond=None)[0]
    if kind == "mean4":
        m = V.mean(1, keepdims=True); R = V - m @ np.linalg.lstsq(m, V, rcond=None)[0]
        U, S, _ = np.linalg.svd(R, full_matrices=False); return np.c_[m, U[:, :k - 1]]
    U, S, _ = np.linalg.svd(V, full_matrices=False); return U[:, :k]


def art_dummies(gs, mot, mode):
    if mode == "none":
        return np.zeros((len(gs), 0))
    Z, M = {"int": (5.0, 0.9), "cons": (3.0, 0.5)}[mode]
    dg = np.r_[0, np.diff(gs)]; z = np.abs((dg - dg.mean()) / dg.std())
    tr, rot = mot[:, :3], mot[:, 3:6] * np.pi / 180 * 50                   # rotations (deg) -> mm on a 50-mm sphere
    fd = np.r_[0, np.abs(np.diff(np.c_[tr, rot], axis=0)).sum(1)]
    out = np.flatnonzero((z > Z) | (fd > M)); D = np.zeros((len(gs), len(out)))
    D[out, np.arange(len(out))] = 1.0
    return D


def bandpass(Y, kind):
    if kind == "butter":
        import luppi_r4e_lib as Lib
        return Lib.bandpass(Y)
    T = Y.shape[1]; f = np.fft.rfftfreq(T, TR); F = np.fft.rfft(Y - Y.mean(1, keepdims=True), axis=1); F[:, (f < 0.008) | (f > 0.09)] = 0
    return np.fft.irfft(F, T, axis=1)


def prepare(X, lab, bm, atl):
    """One pass over the 4D data: ROI means (100, T), global signal, and WM / CSF voxel pools for erosion 0 and 1."""
    T = X.shape[3]; Xf = X.reshape(-1, T); a = atl.ravel()
    Y = np.stack([Xf[a == p].mean(0) for p in range(1, 101)]).astype(float); gs = Xf[bm.ravel()].mean(0).astype(float)
    pools = {e: (Xf[erode(np.isin(lab, WM_LABELS) & bm, e).ravel()].T.astype(float), Xf[erode(np.isin(lab, CSF_LABELS) & bm, e).ravel()].T.astype(float)) for e in (0, 1)}
    return Y, gs, pools


def pipeline(Y, gs, pools, mot, comp="pca5", art="int", filt="fft", ero=1):
    T = Y.shape[1]; Vw, Vc = pools[ero]; t = np.arange(T, dtype=float)
    C = np.c_[comps(Vw, comp), comps(Vc, comp), mot[:, :12], art_dummies(gs, mot, art), np.ones(T), t]
    B = np.linalg.lstsq(C, Y.T, rcond=None)[0]; R = (Y.T - C @ B).T
    return bandpass(R, filt)[:, DROP:], dict(n_wm=int(Vw.shape[1]), n_csf=int(Vc.shape[1]), n_art=int(C.shape[1] - 10 - 12 - 2))


def corr_rows(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def validate(out):
    import scipy.io as sio
    demo = [np.asarray(x, float) for x in sio.loadmat(DATA_ROOT + "raw/luppi_demo/gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel()]
    res = {}
    for k, s in enumerate(["100307", "100408", "101107"]):
        X, lab, bm, atl, mot = load_subject(s); res[s] = {}; Y, gs, pools = prepare(X, lab, bm, atl); T = X.shape[3]; del X
        for comp, art, filt, ero in itertools.product(("pca5", "mean4"), ("none", "int", "cons"), ("fft", "butter"), (0, 1)):
            Yd, info = pipeline(Y, gs, pools, mot, comp, art, filt, ero); r = corr_rows(Yd, demo[k])
            res[s][f"{comp}|{art}|{filt}|ero{ero}"] = dict(median_r=float(np.median(r)), min_r=float(r.min()), **info)
        # no-denoising reference (minimal volumetric + band-pass only)
        t = np.arange(T, dtype=float)
        D = np.c_[np.ones(T), t]; R0 = (Y.T - D @ np.linalg.lstsq(D, Y.T, rcond=None)[0]).T
        r0 = corr_rows(bandpass(R0, "fft")[:, DROP:], demo[k]); res[s]["no_denoise|fft"] = dict(median_r=float(np.median(r0)), min_r=float(r0.min()))
        best = max(res[s].items(), key=lambda kv: kv[1]["median_r"])
        print(s, "best", best[0], round(best[1]["median_r"], 4), round(best[1]["min_r"], 4), "| no-denoise", round(res[s]["no_denoise|fft"]["median_r"], 4), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(sys.argv[2])


# ---------------------------------------------------------------- validation round 2 (subject masks, "effect of rest")
def subject_masks(s, ref_img, thr):
    """Fraction of WM / CSF 0.7-mm aparc+aseg voxels inside each 2-mm voxel (both in MNI space, mapped by the affines);
 mask = fraction >= thr."""
    import nibabel as nib
    a = nib.load(RAW + f"{s}_aparc+aseg.nii.gz"); lab = np.asarray(a.dataobj).astype(int); shp = ref_img.shape[:3]
    ijk = np.argwhere(lab > 0); xyz = np.c_[ijk, np.ones(len(ijk))] @ a.affine.T; ind = np.rint(xyz @ np.linalg.inv(ref_img.affine).T)[:, :3].astype(int)
    ok = np.all((ind >= 0) & (ind < np.array(shp)), 1); ind, l = ind[ok], lab[tuple(ijk[ok].T)]
    flat = np.ravel_multi_index(ind.T, shp); n_all = np.bincount(flat, minlength=np.prod(shp)).astype(float)
    # all 0.7-mm voxels (including label 0) per 2-mm voxel, for the denominator
    ijk0 = np.argwhere(np.ones(lab.shape, bool)); xyz0 = np.c_[ijk0, np.ones(len(ijk0))] @ a.affine.T; ind0 = np.rint(xyz0 @ np.linalg.inv(ref_img.affine).T)[:, :3].astype(int)
    ok0 = np.all((ind0 >= 0) & (ind0 < np.array(shp)), 1); den = np.bincount(np.ravel_multi_index(ind0[ok0].T, shp), minlength=np.prod(shp)).astype(float)
    frac = lambda L: (np.bincount(flat[np.isin(l, L)], minlength=np.prod(shp)) / np.maximum(den, 1)).reshape(shp)
    return frac(ASEG_WM) >= thr, frac(ASEG_CSF) >= thr


def rest_effect(T):
    """CONN 'effect of rest' for a single resting session: HRF-convolved boxcar (all ones) and its temporal derivative."""
    from scipy.stats import gamma
    t = np.arange(0, 32, TR); h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0; h = h / h.sum()
    r = np.convolve(np.ones(T), h)[:T]; return np.c_[r, np.r_[0, np.diff(r)]]


def validate2(out):
    import nibabel as nib, scipy.io as sio
    demo = [np.asarray(x, float) for x in sio.loadmat(DATA_ROOT + "raw/luppi_demo/gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel()]
    res = {}
    for k, s in enumerate(["100307", "100408", "101107"]):
        X, lab, bm, atl, mot = load_subject(s); ref = nib.load(RAW + f"{s}_rfMRI_REST1_LR.nii.gz"); T = X.shape[3]; Xf = X.reshape(-1, T); a = atl.ravel()
        Y = np.stack([Xf[a == p].mean(0) for p in range(1, 101)]).astype(float); gs = Xf[bm.ravel()].mean(0).astype(float)
        pools = {"group|ero0": None}
        for thr in (0.5, 0.9):
            wm, csf = subject_masks(s, ref, thr)
            for e in (0, 1):
                w_, c_ = erode(wm & bm, e), erode(csf & bm, e)
                pools[f"subj{int(thr * 100)}|ero{e}"] = (Xf[w_.ravel()].T.astype(float), Xf[c_.ravel()].T.astype(float))
        g_w, g_c = np.isin(lab, WM_LABELS) & bm, np.isin(lab, CSF_LABELS) & bm
        pools["group|ero0"] = (Xf[g_w.ravel()].T.astype(float), Xf[g_c.ravel()].T.astype(float)); del X, Xf
        res[s] = {}; t = np.arange(T, dtype=float); RE = rest_effect(T)
        for pk, (Vw, Vc) in pools.items():
            for comp in ("mean4", "pca5"):
                for rest in (0, 1):
                    C = np.c_[comps(Vw, comp), comps(Vc, comp), mot[:, :12], art_dummies(gs, mot, "int"), np.ones(T), t] if Vw.shape[1] >= 5 and Vc.shape[1] >= 5 else None
                    if C is None:
                        res[s][f"{pk}|{comp}|rest{rest}"] = dict(median_r=np.nan, min_r=np.nan, n_wm=int(Vw.shape[1]), n_csf=int(Vc.shape[1])); continue
                    if rest:
                        C = np.c_[C, RE]
                    B = np.linalg.lstsq(C, Y.T, rcond=None)[0]; R = (Y.T - C @ B).T; r = corr_rows(bandpass(R, "fft")[:, DROP:], demo[k])
                    res[s][f"{pk}|{comp}|rest{rest}"] = dict(median_r=float(np.median(r)), min_r=float(r.min()), n_wm=int(Vw.shape[1]), n_csf=int(Vc.shape[1]))
        top = sorted(((v["median_r"], kk) for kk, v in res[s].items() if np.isfinite(v["median_r"])), reverse=True)[:5]
        print(s, "top", [(round(x, 4), kk) for x, kk in top], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "validate2":
    validate2(sys.argv[2])
