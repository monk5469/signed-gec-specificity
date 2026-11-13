"""Descriptive check of the SC calibration (Supplementary Note 12): effect of the atlas space on the agreement with the authors'
example SCs."""
import os, json, shutil, subprocess, multiprocessing as mp
import numpy as np, scipy.io as sio, nibabel as nib
from paths import DATA_ROOT
from scipy.stats import spearmanr
import r23_sc as S

SRC = DATA_ROOT + "raw/p33_atlas2009c/"; TMP = S.OUT + "gsc_diag3_tmp/"; iu = np.triu_indices(100, 1)
ATL = {"fsl": S.A100, "c2009_2mm": SRC + "mni2009c_schaefer100_2mm.nii.gz", "c2009_1mm": SRC + "mni2009c_schaefer100_1mm.nii.gz"}
KEY = {"fsl": "mni_schaefer100", "c2009_2mm": "mni2009c_schaefer100_2mm", "c2009_1mm": "mni2009c_schaefer100_1mm"}


def centroids(path):
    a = nib.load(path); L = np.rint(np.asarray(a.dataobj)).astype(int); ijk = np.argwhere(L > 0); xyz = nib.affines.apply_affine(a.affine, ijk); lab = L[tuple(ijk.T)]
    return np.array([xyz[lab == k].mean(0) for k in range(1, 101)]), np.array([(lab == k).sum() * abs(np.linalg.det(a.affine[:3, :3])) for k in range(1, 101)])


def run(sid):
    tt = TMP + f"{sid}.tt.gz"
    cmd = [S.EXE, "--action=trk", f"--source={S.QSDR}{sid}.qsdr.fz", "--method=0", f"--tract_count={S.COUNT}", "--turning_angle=55", "--step_size=1.0",
           "--smoothing=0", "--min_length=10", "--max_length=400", "--otsu_threshold=0.6", "--fa_threshold=0", f"--random_seed={S.SEED}", "--check_ending=1",
           "--connectivity=" + ",".join(ATL.values()), "--connectivity_type=end", "--connectivity_value=count", "--connectivity_output=matrix", "--thread_count=1", f"--output={tt}"]
    subprocess.run(cmd, capture_output=True)
    out = {}
    for k, nm in KEY.items():
        M, names = S._mat(f"{tt}.{nm}.connectivity.mat"); np.fill_diagonal(M, 0.0)
        out[k] = (M, len(names) == 100 and all(x.strip().endswith(f"_{i + 1}") for i, x in enumerate(names)))
    return sid, out


def score(x, R):
    u = (x > 0) | (R > 0); return dict(spearman=float(spearmanr(x[u], R[u])[0]), jaccard=float(((x > 0) & (R > 0)).sum() / u.sum()), density=float((x > 0).mean()))


if __name__ == "__main__":
    for res, k in (("02", "c2009_2mm"), ("01", "c2009_1mm")):
        a = nib.load(SRC + f"tpl-MNI152NLin2009cAsym_res-{res}_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz")
        nib.save(nib.Nifti1Image(np.rint(np.asarray(a.dataobj)).astype(np.int16), a.affine, a.header), ATL[k])
    names = [l.split("\t")[1] for l in open(SRC + "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.tsv").read().splitlines()[1:]]
    cf, vf = centroids(ATL["fsl"]); chk = dict(tsv_n=len(names), tsv_lh_first50=all("_LH_" in n for n in names[:50]), tsv_rh_last50=all("_RH_" in n for n in names[50:]))
    vols = {"fsl": vf}
    for k in ("c2009_2mm", "c2009_1mm"):
        c, v = centroids(ATL[k]); vols[k] = v; d = np.linalg.norm(c - cf, axis=1); nn = np.argmin(np.linalg.norm(c[:, None] - cf[None], axis=2), axis=1)
        chk[k] = dict(centroid_shift_mm_median=float(np.median(d)), max=float(d.max()), nearest_label_is_same=float((nn == np.arange(100)).mean()),
                      volume_ratio_median=float(np.median(v / vf)))
    print("label check", json.dumps(chk), flush=True)
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    with mp.Pool(3) as p:
        res = p.map(run, S.DEMO, chunksize=1)
    ref = [np.asarray(x, float) for x in sio.loadmat(S.DEMO_SC)["DTI_fibers_VolNorm_HCP"].ravel()]; rows = []
    for sid, out in res:
        R = ref[S.DEMO.index(sid)][iu]; prod = np.load(S.SCDIR + f"{sid}.npz")["sc100_ce1"].astype(float); np.fill_diagonal(prod, 0.0)
        for k, (M, order_ok) in out.items():
            vn = vols[k][:, None] + vols[k][None, :]; hom = int(sum(M[i, i + 50] > 0 for i in range(50)))
            rows.append(dict(id=sid, atlas=k, order_ok=order_ok, equals_production=bool(np.array_equal(M, prod)), count=score(M[iu], R), volnorm=score((M / vn)[iu], R),
                             homotopic=hom, rho_vs_fsl=float(spearmanr(M[iu], out["fsl"][0][iu])[0])))
    shutil.rmtree(TMP, ignore_errors=True)
    med = {k: {w: dict(spearman=float(np.median([r[w]["spearman"] for r in rows if r["atlas"] == k])), jaccard=float(np.median([r[w]["jaccard"] for r in rows if r["atlas"] == k])))
               for w in ("count", "volnorm")} for k in KEY}
    json.dump(dict(label_check=chk, rows=rows, medians=med), open(S.OUT + "r23_gsc_diag3.json", "w"), indent=1)
    for r in rows:
        print(r["atlas"], r["id"], "| order", r["order_ok"], "| = production", r["equals_production"], "| count rho %.3f J %.3f d %.3f" % (r["count"]["spearman"], r["count"]["jaccard"], r["count"]["density"]),
              "| volnorm rho %.3f" % r["volnorm"]["spearman"], "| homotopic", r["homotopic"], "| rho vs fsl %.3f" % r["rho_vs_fsl"], flush=True)
    print("medians", json.dumps(med), flush=True)
