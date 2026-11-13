"""Participant-specific SC (Section 2.4.2): builds the 232-region atlas, runs DSI Studio tractography on the QSDR reconstructions
for the four prespecified variants, and calibrates them against the authors' example SCs with the selection and fallback
rules.
Usage: python r23_sc.py atlas | build [--procs 56] | gsc"""
import sys, os, json, time, hashlib, argparse, subprocess, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

EXE = DATA_ROOT + "tools/dsi_studio_2025.04.16/dsi_studio_win/dsi_studio.exe"; QSDR = DATA_ROOT + "raw/p33_qsdr/"
ATL = DATA_ROOT + "raw/p33_atlas232/"; A100_SRC = DATA_ROOT + "raw/p33_hcp_vol/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
A200_SRC = ATL + "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"; TIAN_SRC = ATL + "Tian_Subcortex_S2_3T.nii"
A100 = ATL + "mni_schaefer100.nii.gz"; A232 = ATL + "mni_schaefer232.nii.gz"
OUT = DATA_ROOT + "results/p33/r23/"; SCDIR = OUT + "sc/"; GSC = OUT + "r23_gsc.json"; DEMO_SC = DATA_ROOT + "raw/luppi_demo/gec_3/SC.mat"
DEMO = ("100307", "100408", "101107"); COUNT = 1_000_000; SEED = 20261016


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def stage_atlas():
    import nibabel as nib
    s100, s200, tian = nib.load(A100_SRC), nib.load(A200_SRC), nib.load(TIAN_SRC)
    for im in (s200, tian):
        if im.shape[:3] != s100.shape[:3] or not np.allclose(im.affine, s100.affine, atol=1e-4):
            raise SystemExit(f"grid / affine mismatch: {im.shape} {np.round(im.affine, 2).tolist()}")
    L200 = np.asarray(s200.dataobj).astype(np.int16); T = np.rint(np.asarray(tian.dataobj)).astype(np.int16)
    overlap = int(((L200 > 0) & (T > 0)).sum()); A = L200.copy(); m = (L200 == 0) & (T > 0); A[m] = T[m] + 200
    sizes = {int(k): int((A == k).sum()) for k in range(1, 233)}
    if min(sizes.values()) == 0 or sorted(np.unique(T[T > 0]).tolist()) != list(range(1, 33)):
        raise SystemExit("empty parcel or unexpected Tian labels")
    nib.save(nib.Nifti1Image(A, s100.affine, s100.header), A232); nib.save(nib.Nifti1Image(np.asarray(s100.dataobj).astype(np.int16), s100.affine, s100.header), A100)
    rec = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), overlap_voxels_resolved_to_cortex=overlap, min_parcel_voxels=min(sizes.values()),
               sizes=sizes, sha={os.path.basename(p): sha(p) for p in (A100_SRC, A200_SRC, TIAN_SRC, A100, A232)})
    json.dump(rec, open(OUT + "r23_atlas.json", "w"), indent=1); print("atlas232 built | overlap voxels", overlap, "| min parcel voxels", rec["min_parcel_voxels"], flush=True)


def _mat(path, key="number of tracts r2r"):
    import scipy.io as sio
    z = sio.loadmat(path); M = np.asarray(z[key], float); names = bytes(np.asarray(z["name"]).ravel().astype(np.uint8)).decode(errors="replace").replace("\x00", "").split("\n")
    return M, [n for n in names if n.strip()]


def build_one(args):
    sid, ce, count, outdir = args
    tag = f"{sid}_ce{ce}"; tt = outdir + f"{tag}.tt.gz"; t0 = time.time()
    cmd = [EXE, "--action=trk", f"--source={QSDR}{sid}.qsdr.fz", "--method=0", f"--tract_count={count}", "--turning_angle=55", "--step_size=1.0",
           "--smoothing=0", "--min_length=10", "--max_length=400", "--otsu_threshold=0.6", "--fa_threshold=0", f"--random_seed={SEED}", f"--check_ending={ce}",
           f"--connectivity={A100},{A232}", "--connectivity_type=end", "--connectivity_value=count", "--connectivity_output=matrix", "--thread_count=1", f"--output={tt}"]
    try:
        subprocess.run(cmd, capture_output=True, timeout=6 * 3600)
        out = {}
        for nm, n in (("mni_schaefer100", 100), ("mni_schaefer232", 232)):
            M, names = _mat(f"{tt}.{nm}.connectivity.mat"); np.fill_diagonal(M, 0.0)
            order_ok = len(names) == n and all(x.strip().endswith(f"_{i + 1}") for i, x in enumerate(names))
            deg0 = int((M.sum(1) == 0).sum()); off = ~np.eye(n, dtype=bool)
            val = dict(finite=bool(np.all(np.isfinite(M))), nonneg=bool(np.all(M >= 0)), symmetric=bool(np.allclose(M, M.T)), order_ok=bool(order_ok),
                       zero_degree=deg0, density=float((M[off] > 0).mean()), counted_share=float(np.triu(M, 1).sum() / count))
            val["valid"] = bool(val["finite"] and val["nonneg"] and val["symmetric"] and order_ok and deg0 <= 2 and val["density"] >= 0.05 and val["counted_share"] >= 0.10)
            out[f"sc{n}"] = M; out[f"val{n}"] = val
            os.remove(f"{tt}.{nm}.connectivity.mat")
        if os.path.exists(tt):
            os.remove(tt)
        return sid, ce, out, round(time.time() - t0, 1), None
    except Exception as e:
        return sid, ce, None, round(time.time() - t0, 1), repr(e)[:300]


def stage_build(procs, ids=None, count=COUNT, outdir=SCDIR):
    os.makedirs(outdir, exist_ok=True)
    ids = ids or sorted(f.split(".")[0] for f in os.listdir(QSDR) if f.endswith(".qsdr.fz"))
    with mp.Pool(procs) as p:
        res = p.map(build_one, [(s, ce, count, outdir) for s in ids for ce in (1, 0)], chunksize=1)
    by = {}; log = []
    for sid, ce, out, secs, err in res:
        log.append(dict(id=sid, ce=ce, secs=secs, error=err, **({f"val{n}": out[f"val{n}"] for n in (100, 232)} if out else {})))
        if out:
            by.setdefault(sid, {}).update({f"sc{n}_ce{ce}": out[f"sc{n}"] for n in (100, 232)})
    for sid, d in by.items():
        np.savez(outdir + f"{sid}.npz", **d)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), count=count, seed=SEED, exe_sha=sha(EXE), runs=log,
                   npz_sha={s: sha(outdir + f"{s}.npz") for s in by}), open(outdir + "build_log.json", "w"), indent=1)
    bad = [(r["id"], r["ce"]) for r in log if r["error"] or not (r.get("val100", {}).get("valid") and r.get("val232", {}).get("valid"))]
    print("runs", len(log), "| participants", len(by), "| invalid / failed", bad[:10], "| median secs", np.median([r["secs"] for r in log]), flush=True)


def volumes(path):
    import nibabel as nib
    A = np.asarray(nib.load(path).dataobj).astype(int); return np.array([(A == k).sum() * 8.0 for k in range(1, int(A.max()) + 1)])


def candidates(d):
    v = volumes(A100); vn = v[:, None] + v[None, :]
    return {f"ce{ce}_{w}": (d[f"sc100_ce{ce}"] if w == "count" else d[f"sc100_ce{ce}"] / vn) for ce in (1, 0) for w in ("count", "volnorm")}


def stage_gsc():
    import scipy.io as sio
    from scipy.stats import spearmanr
    if os.path.exists(GSC):
        raise SystemExit("G-SC already recorded")
    ref = [np.asarray(x, float) for x in sio.loadmat(DEMO_SC)["DTI_fibers_VolNorm_HCP"].ravel()]; iu = np.triu_indices(100, 1); score = {}
    for i, sid in enumerate(DEMO):
        d = np.load(SCDIR + f"{sid}.npz"); R = ref[i][iu]
        for name, M in candidates(d).items():
            x = M[iu]; u = (x > 0) | (R > 0); jac = float(((x > 0) & (R > 0)).sum() / u.sum())
            score.setdefault(name, []).append(dict(id=sid, spearman=float(spearmanr(x[u], R[u])[0]), jaccard=jac, density=float((x > 0).mean())))
    med = {k: float(np.median([r["spearman"] for r in v])) for k, v in score.items()}
    order = sorted(med, key=lambda k: (-round(med[k], 12), 0 if k.startswith("ce1") else 1, 0 if k.endswith("count") else 1)); sel = order[0]
    jm = float(np.median([r["jaccard"] for r in score[sel]])); dens_ok = all(0.15 <= r["density"] <= 0.35 for r in score[sel])
    gate = bool(med[sel] >= 0.60 and jm >= 0.50 and dens_ok)
    rec = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), scores=score, median_spearman=med, selected=sel, median_jaccard_selected=jm,
               densities_in_range=dens_ok, gate_passed=gate, label="reproduction-calibrated individual SC" if gate else "independent tractography SC")
    json.dump(rec, open(GSC, "x"), indent=1)
    print("G-SC | median Spearman", {k: round(v, 3) for k, v in med.items()}, "| selected", sel, "| Jaccard", round(jm, 3), "| densities ok", dens_ok, "| gate", gate, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); ap.add_argument("--ids"); ap.add_argument("--count", type=int, default=COUNT)
    ap.add_argument("--out", default=SCDIR); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.stage == "atlas":
        stage_atlas()
    elif a.stage == "build":
        stage_build(a.procs, a.ids.split(",") if a.ids else None, a.count, a.out)
    elif a.stage == "gsc":
        stage_gsc()
