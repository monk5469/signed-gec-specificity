"""Preprocessing comparison, step 1 (Section 2.3): nuisance regression of the REST1 runs under each condition of Table 3, written as
unfiltered regional time series; band-pass filtering follows in the fitting stage.
Usage: python luppi_ppc_prep.py OUT_MANIFEST [--procs 16]"""
from paths import DATA_ROOT
import sys, os, json, hashlib, argparse, multiprocessing as mp
import numpy as np

RAW = DATA_ROOT + "raw/p33_hcp_vol/"; WORK = DATA_ROOT + "work/"
SUBJ_JSON = DATA_ROOT + "results/p33/p33_hcp_s200_v13.json"; MASK_SRC = RAW + "100307_Atlas_wmparc.2.nii.gz"
CONDS = ("vol0", "mot", "acc", "acc1", "acc3", "accero", "gsr", "null10")
RANK_TOL = 1e-8                     # relative numerical-rank tolerance (frozen)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def job(args):
    s, d = args
    import nibabel as nib
    import luppi_pipeline_vol as P, luppi_hopf as L, luppi_r4e_lib as Lib
    vol = RAW + f"{s}_rfMRI_REST1_{d}.nii.gz"; bmf = RAW + f"{s}_REST1_{d}_brainmask_fs.2.nii.gz"
    movf = RAW + f"{s}_REST1_{d}_Movement_Regressors.txt"                                     # same run only; missing -> error
    try:
        if not os.path.exists(movf):
            raise FileNotFoundError(movf)
        imgs = {n: nib.load(p) for n, p in (("vol", vol), ("mask", MASK_SRC), ("bm", bmf), ("atlas", P.ATLAS))}
        if len({tuple(i.shape[:3]) for i in imgs.values()}) != 1 or not all(np.allclose(i.affine, imgs["vol"].affine, atol=1e-3) for i in imgs.values()):
            raise ValueError("voxel grid / affine mismatch: " + str({n: (i.shape[:3], np.round(i.affine[:3, 3], 2).tolist()) for n, i in imgs.items()}))
        X = np.asarray(imgs["vol"].dataobj, dtype=np.float32); lab = np.asarray(imgs["mask"].dataobj).astype(int)
        bm = np.asarray(imgs["bm"].dataobj) > 0; atl = np.asarray(imgs["atlas"].dataobj).astype(int); mot = np.loadtxt(movf)
        T = X.shape[3]
        if T != 1200 or mot.shape != (T, 12):
            raise ValueError(f"frames {T} motion {mot.shape}")
        Y, gs, pools = P.prepare(X, lab, bm, atl); del X
        t = np.arange(T, dtype=float); base = np.c_[np.ones(T), t]; art = P.art_dummies(gs, mot, "int"); Vw, Vc = pools[0]; We, Ce = pools[1]
        mplus = np.c_[base, mot[:, :12], art]; tis = lambda A, B, k: np.c_[P.comps(A, "mean4", k), P.comps(B, "mean4", k)]
        acc_t = tis(Vw, Vc, 5); prng = np.random.default_rng([int(s), 0 if d == "LR" else 1, 20261009])
        def phase_rand(A):
            F = np.fft.rfft(A - A.mean(0), axis=0); ph = np.exp(1j * prng.uniform(0, 2 * np.pi, F.shape)); ph[0] = 1; ph[-1] = 1
            return np.fft.irfft(F * ph, A.shape[0], axis=0)
        Cs = dict(vol0=base, mot=mplus, acc=np.c_[mplus, acc_t], acc1=np.c_[mplus, tis(Vw, Vc, 1)], acc3=np.c_[mplus, tis(Vw, Vc, 3)],
                  accero=np.c_[mplus, tis(We, Ce, 5)], gsr=np.c_[mplus, gs], null10=np.c_[mplus, phase_rand(acc_t)])
        nz = lambda M: M / np.maximum(np.linalg.norm(M, axis=0), 1e-300)
        rk = lambda M: int(np.linalg.matrix_rank(nz(M), tol=RANK_TOL * np.linalg.svd(nz(M), compute_uv=False)[0]))      # frozen tolerance
        inc = {c: rk(Cs[c]) - rk(mplus) for c in ("acc", "null10")}
        if inc["acc"] != inc["null10"] or inc["acc"] != 10:
            raise ValueError(f"incremental design rank acc {inc['acc']} null10 {inc['null10']} (expected 10)")
        out = dict(subject=s, run=d, inputs={os.path.basename(p): sha(p) for p in (vol, bmf, movf)}, n_art=int(art.shape[1]), n_wm=int(Vw.shape[1]), n_csf=int(Vc.shape[1]), n_wm_ero=int(We.shape[1]), n_csf_ero=int(Ce.shape[1]), incremental_rank=inc, cond={})
        # calibration measures for the known-truth aCompCor simulation: band-passed, 11 frames dropped, detrended
        bp = lambda v: Lib.bandpass(np.atleast_2d(v)[:, 11:])[0]
        g_gm = bp(Y.mean(0)); r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        acc_reg = np.c_[P.comps(Vw, "mean4"), P.comps(Vc, "mean4")]; Ab = Lib.bandpass(acc_reg.T[:, 11:]).T
        beta = np.linalg.lstsq(Ab, g_gm, rcond=None)[0]
        out["calib"] = dict(r_wm_mean_gm_global=r(bp(Vw.mean(1)), g_gm), r_csf_mean_gm_global=r(bp(Vc.mean(1)), g_gm),
                            R2_gm_global_on_acc=float(1 - np.var(g_gm - Ab @ beta) / np.var(g_gm)),
                            wm_pc_var_explained=[float(v) for v in (lambda S: (S ** 2 / (S ** 2).sum())[:5])(np.linalg.svd(Vw - Vw.mean(0), full_matrices=False, compute_uv=False))])
        for c, C in Cs.items():
            R = (Y.T - C @ np.linalg.lstsq(C, Y.T, rcond=None)[0])                                # (T, 100), unfiltered
            if not (np.all(np.isfinite(R)) and R.std(0).min() > 0):
                raise ValueError(f"{c}: non-finite or constant output")
            os.makedirs(WORK + f"p33_ts_{c}/", exist_ok=True); p = WORK + f"p33_ts_{c}/{s}_REST1_{d}_s100.npy"; np.save(p, R)
            x = Lib.bandpass(R.T[:, 11:]); F = np.corrcoef(L.prep_emp(x)); iu = np.triu_indices(100, 1)
            out["cond"][c] = dict(npy_sha=sha(p), gs_share=float(np.var(x.mean(0)) / np.mean(x.var(1))), mean_fc=float(F[iu].mean()), neg_fc=float((F[iu] < 0).mean()))
        out["ok"] = True
        return out
    except Exception as e:
        return dict(subject=s, run=d, ok=False, error=repr(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--procs", type=int, default=16); a = ap.parse_args()
    subs = [x["subject"] for x in json.load(open(SUBJ_JSON))["subjects"]]
    with mp.Pool(a.procs) as p:
        res = p.map(job, [(s, d) for s in subs for d in ("LR", "RL")], chunksize=1)
    import luppi_pipeline_vol as P
    man = dict(n=len(res), n_ok=sum(r["ok"] for r in res), mask_source=dict(file=os.path.basename(MASK_SRC), sha=sha(MASK_SRC)), atlas=dict(file=os.path.basename(P.ATLAS), sha=sha(P.ATLAS)),
               code={c: sha(c) for c in ("luppi_ppc_prep.py", "luppi_pipeline_vol.py")}, records=res)
    json.dump(man, open(a.out, "x"), indent=1)
    bad = [(r["subject"], r["run"], r.get("error")) for r in res if not r["ok"]]
    print("runs", len(res), "ok", man["n_ok"], "bad", bad[:5], flush=True)
    for c in CONDS:
        g = [r["cond"][c]["gs_share"] for r in res if r["ok"] and r["run"] == "LR"]; n = [r["cond"][c]["neg_fc"] for r in res if r["ok"] and r["run"] == "LR"]
        print(c, "LR gs share median %.3f [%.3f, %.3f]" % (np.median(g), np.quantile(g, .25), np.quantile(g, .75)), "| neg FC median %.3f" % np.median(n), flush=True)
