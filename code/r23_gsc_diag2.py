"""Descriptive check of the SC calibration (Supplementary Note 12): effect of DSI Studio bias-field correction on the agreement
with the authors' example SCs."""
import os, json, subprocess, multiprocessing as mp
import numpy as np, scipy.io as sio
from scipy.stats import spearmanr
import r23_sc as S

TMP = S.OUT + "gsc_diag2_tmp/"; iu = np.triu_indices(100, 1)


def run(args):
    sid, var = args; tt = TMP + f"{sid}_{var}.tt.gz"
    cmd = [S.EXE, "--action=trk", f"--source={S.QSDR}{sid}.qsdr.fz", "--method=0", f"--tract_count={S.COUNT}", "--turning_angle=55", "--step_size=1.0",
           "--smoothing=0", "--min_length=10", "--max_length=400", "--otsu_threshold=0.6", "--fa_threshold=0", f"--random_seed={S.SEED}", "--check_ending=1",
           f"--connectivity={S.A100}", "--connectivity_type=end", "--connectivity_value=count", "--connectivity_output=matrix", "--thread_count=1", f"--output={tt}"]
    if var == "nobfc":
        cmd.append("--bias_field_correction=0")
    log = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    M, names = S._mat(f"{tt}.mni_schaefer100.connectivity.mat"); np.fill_diagonal(M, 0.0)
    for f in (tt, f"{tt}.mni_schaefer100.connectivity.mat"):
        if os.path.exists(f):
            os.remove(f)
    return sid, var, M, ("bias_field_correction=0" in log.replace("\x1b[0;32m", "").replace("\x1b[0m", "")), len(names)


def score(x, R):
    u = (x > 0) | (R > 0); return dict(spearman=float(spearmanr(x[u], R[u])[0]), jaccard=float(((x > 0) & (R > 0)).sum() / u.sum()), density=float((x > 0).mean()))


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    with mp.Pool(6) as p:
        res = p.map(run, [(s, v) for s in S.DEMO for v in ("base", "nobfc")], chunksize=1)
    ref = [np.asarray(x, float) for x in sio.loadmat(S.DEMO_SC)["DTI_fibers_VolNorm_HCP"].ravel()]; v = S.volumes(S.A100); vn = v[:, None] + v[None, :]
    out = dict(rows=[])
    for sid, var, M, bfc_logged, nn in res:
        i = S.DEMO.index(sid); R = ref[i][iu]; prod = np.load(S.SCDIR + f"{sid}.npz")["sc100_ce1"].astype(float); np.fill_diagonal(prod, 0.0)
        out["rows"].append(dict(id=sid, variant=var, names=nn, bfc_off_in_log=bfc_logged, equals_production=bool(np.array_equal(M, prod)),
                                spearman_vs_production=float(spearmanr(M[iu], prod[iu])[0]), count=score(M[iu], R), volnorm=score((M / vn)[iu], R)))
    for var in ("base", "nobfc"):
        rs = [r for r in out["rows"] if r["variant"] == var]
        out[f"median_{var}"] = {w: dict(spearman=float(np.median([r[w]["spearman"] for r in rs])), jaccard=float(np.median([r[w]["jaccard"] for r in rs]))) for w in ("count", "volnorm")}
    json.dump(out, open(S.OUT + "r23_gsc_diag2.json", "w"), indent=1)
    for r in sorted(out["rows"], key=lambda r: (r["variant"], r["id"])):
        print(r["variant"], r["id"], "| bfc off logged", r["bfc_off_in_log"], "| equals production", r["equals_production"], "| rho vs production %.3f" % r["spearman_vs_production"],
              "| count rho %.3f J %.3f d %.3f" % (r["count"]["spearman"], r["count"]["jaccard"], r["count"]["density"]), "| volnorm rho %.3f" % r["volnorm"]["spearman"], flush=True)
    print("medians", {k: v for k, v in out.items() if k.startswith("median")}, flush=True)
    os.rmdir(TMP)
