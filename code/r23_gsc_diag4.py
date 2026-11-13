"""Descriptive check of the SC calibration (Supplementary Note 12): effect of the QSDR reconstruction version, using
reconstructions from the raw diffusion data of the three example participants."""
import os, json, shutil, subprocess, multiprocessing as mp
import numpy as np, scipy.io as sio
from paths import DATA_ROOT
from scipy.stats import spearmanr
import r23_sc as S

SRC = DATA_ROOT + "raw/p33_src/"; TMP = S.OUT + "gsc_diag4_tmp/"; iu = np.triu_indices(100, 1)
REC = {"ctrl": [], "old2mm": ["--param0=1.25", "--qsdr_reso=2"], "old2mm_nobfc": ["--param0=1.25", "--qsdr_reso=2", "--bias_field_correction=0"]}


def clean(s):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def job(args):
    sid, var = args; fz = TMP + f"{sid}_{var}.fz"; tt = TMP + f"{sid}_{var}.tt.gz"
    rl = clean(subprocess.run([S.EXE, "--action=rec", f"--source={SRC}{sid}.sz", "--method=7", *REC[var], "--thread_count=4", f"--output={fz}"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace").stdout)
    if not os.path.exists(fz):
        cand = [f for f in os.listdir(TMP) if f.startswith(f"{sid}_{var}") and f.endswith(".fz")]
        fz = TMP + cand[0] if cand else fz
    cmd = [S.EXE, "--action=trk", f"--source={fz}", "--method=0", f"--tract_count={S.COUNT}", "--turning_angle=55", "--step_size=1.0", "--smoothing=0",
           "--min_length=10", "--max_length=400", "--otsu_threshold=0.6", "--fa_threshold=0", f"--random_seed={S.SEED}", "--check_ending=1", f"--connectivity={S.A100}",
           "--connectivity_type=end", "--connectivity_value=count", "--connectivity_output=matrix", "--thread_count=1", f"--output={tt}"]
    if var == "old2mm_nobfc":
        cmd.append("--bias_field_correction=0")
    tl = clean(subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout)
    M, names = S._mat(f"{tt}.mni_schaefer100.connectivity.mat"); np.fill_diagonal(M, 0.0)
    info = [l.strip(" │├└─┬") for l in (rl + tl).splitlines() if any(k in l for k in ("fib_ver", "qsdr_reso", "param=", "bias_field_correction", "dim:"))][:10]
    return sid, var, M, len(names), info


def score(x, R):
    u = (x > 0) | (R > 0); return dict(spearman=float(spearmanr(x[u], R[u])[0]), jaccard=float(((x > 0) & (R > 0)).sum() / u.sum()), density=float((x > 0).mean()))


if __name__ == "__main__":
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(TMP)
    with mp.Pool(9) as p:
        res = p.map(job, [(s, v) for s in S.DEMO for v in REC], chunksize=1)
    ref = [np.asarray(x, float) for x in sio.loadmat(S.DEMO_SC)["DTI_fibers_VolNorm_HCP"].ravel()]; v = S.volumes(S.A100); vn = v[:, None] + v[None, :]; rows = []
    for sid, var, M, nn, info in res:
        R = ref[S.DEMO.index(sid)][iu]; prod = np.load(S.SCDIR + f"{sid}.npz")["sc100_ce1"].astype(float); np.fill_diagonal(prod, 0.0)
        rows.append(dict(id=sid, variant=var, names=nn, info=[x.encode("ascii", "replace").decode() for x in info], rho_vs_production=float(spearmanr(M[iu], prod[iu])[0]),
                         count=score(M[iu], R), volnorm=score((M / vn)[iu], R), homotopic=int(sum(M[i, i + 50] > 0 for i in range(50)))))
    shutil.rmtree(TMP, ignore_errors=True)
    med = {k: {w: dict(spearman=float(np.median([r[w]["spearman"] for r in rows if r["variant"] == k])), jaccard=float(np.median([r[w]["jaccard"] for r in rows if r["variant"] == k])))
               for w in ("count", "volnorm")} for k in REC}
    json.dump(dict(rows=rows, medians=med), open(S.OUT + "r23_gsc_diag4.json", "w"), indent=1)
    for r in sorted(rows, key=lambda r: (r["variant"], r["id"])):
        print(r["variant"], r["id"], "| rho vs production %.3f" % r["rho_vs_production"], "| count rho %.3f J %.3f d %.3f" % (r["count"]["spearman"], r["count"]["jaccard"], r["count"]["density"]),
              "| volnorm rho %.3f" % r["volnorm"]["spearman"], "| homotopic", r["homotopic"], "|", "; ".join(r["info"][:5]), flush=True)
    print("medians", json.dumps(med), flush=True)
