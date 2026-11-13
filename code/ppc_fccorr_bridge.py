"""In-sample model comparison (descriptive; Sections 2.3.3, 3.2): Delta FCcorr = r_signed - r_coop of the PPC fits, the metric used by
Luppi et al. (2026), and its Spearman correlations with FP and P2.
Usage: python ppc_fccorr_bridge.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob
import numpy as np

DEMO = {"100307", "100408", "101107"}; CONDS = ("fix", "vol0", "mot", "acc", "acc1", "acc3", "accero", "gsr", "null10")
FIX_OUT = DATA_ROOT + "results/p33/r4e_e5fix/"


def load(cond):
    import luppi_r4e_ppc as W, luppi_r4e_run as RUN, luppi_r4e_lib as Lib
    X = W.X
    if cond == "fix":
        X.OUT = FIX_OUT; X.HCP = DATA_ROOT + "work/p33_ts_fix/"
    else:
        W.point(cond)
    fsha, fr = X.verify_freeze("e5"); expected = {j["key"]: j for j in X.job_list("e5")}; R = {}; seen = set(); rej = 0
    for p in sorted(glob.glob(X.OUT + "jobs_e5/*.json")):
        try:
            r = json.load(open(p)); assert isinstance(r, dict)
        except Exception:
            rej += 1; continue
        j = expected.get(r.get("key"))
        if (j is None or os.path.basename(j["file"]) != os.path.basename(p) or any(r.get(k) != v for k, v in j.items()) or r.get("state") != "done"
                or r.get("prov") != X.job_prov(j, fsha, fr) or j["key"] in seen):
            rej += 1; continue
        seen.add(j["key"])
        try:
            C = np.load(j["file"].replace(".json", "_C.npy"))
        except Exception:
            C = None
        if C is None or C.shape != (100, 100) or not np.all(np.isfinite(C)) or RUN.sha_arr(C) != r.get("C_sha"):
            rej += 1; continue
        R.setdefault(j["tag"], {})[j["model"]] = (r, C)
    SC = np.load(X.OUT + "e5/SC.npy"); m = Lib.support(SC); rows = {}
    for tag, d in R.items():
        if tag in DEMO or "signed" not in d or "coop" not in d:
            continue
        (rs, Cs), (rc, _) = d["signed"], d["coop"]
        vals = [rs.get("fc_corr", np.nan), rc.get("fc_corr", np.nan), rs.get("mse_ho", np.nan), rc.get("mse_ho", np.nan)]
        if not all(np.isfinite(v) for v in vals):
            continue
        rows[tag] = dict(fc_s=float(vals[0]), fc_c=float(vals[1]), dfc=float(vals[0] - vals[1]), p2=float(vals[2] - vals[3]), fp=float((Cs[m] < 0).mean()))
    return rows, dict(freeze=fsha, rejected=rej)


def main():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(13); out = {}
    for cond in CONDS:
        rows, inf = load(cond); v = list(rows.values()); n = len(v)
        if n < 15:
            out[cond] = dict(info=inf, n=n, label="not analysable"); continue
        dfc = np.array([x["dfc"] for x in v]); B = [np.median(dfc[rng.integers(0, n, n)]) for _ in range(2000)]
        out[cond] = dict(info=inf, n=n, fc_signed=float(np.median([x["fc_s"] for x in v])), fc_coop=float(np.median([x["fc_c"] for x in v])),
                         dfc=dict(median=float(np.median(dfc)), ci95=[float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))], share_pos=float(np.mean(dfc > 0))),
                         p2_median=float(np.median([x["p2"] for x in v])), fp_median=float(np.median([x["fp"] for x in v])),
                         rho_dfc_p2=float(spearmanr(dfc, [x["p2"] for x in v])[0]), rho_dfc_fp=float(spearmanr(dfc, [x["fp"] for x in v])[0]))
        o = out[cond]
        print(cond, "n", n, "rej", inf["rejected"], "| fc signed %.3f coop %.3f | dFC %.3f [%.3f, %.3f] pos %.2f | P2 %.4f FP %.3f | rho(dFC,P2) %.2f rho(dFC,FP) %.2f" % (
            o["fc_signed"], o["fc_coop"], o["dfc"]["median"], *o["dfc"]["ci95"], o["dfc"]["share_pos"], o["p2_median"], o["fp_median"], o["rho_dfc_p2"], o["rho_dfc_fp"]), flush=True)
    json.dump(dict(descriptive=True, note=__doc__, conds=out), open(sys.argv[1], "x"), indent=1)


if __name__ == "__main__":
    main()
