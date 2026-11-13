"""Implementation check, analysis (Section 2.2.3; Supplementary Note 13): FC correlations and FP of the authors' code and of the
reimplementation, and the aggregate distance D.
Usage: python vg_analyze.py OUT_JSON (create-once)"""
import sys, json
import numpy as np, scipy.io as sio
import vg_run as V

TARGET = dict(r_signed=0.74, r_coop=0.51, fp=0.25)


def fp(C, SC):
    m = (SC != 0) & ~np.eye(len(SC), dtype=bool); return float((C[m] < 0).mean())


if __name__ == "__main__":
    V.check_rules(); SCs, _ = V.data(); rows = []
    for s, k, coop in V.JOBS:
        m = sio.loadmat(V.mfile(s, k, coop)); p = np.load(V.pfile(s, k, coop), allow_pickle=True)
        rows.append(dict(subj=s, K=k, coop=coop,
                         M=dict(r=float(np.ravel(m["FC_corr"])[0]), fp=fp(np.asarray(m["GEC"], float), SCs[s - 1]), stop=str(np.ravel(m["stopline"])[0]),
                                secs=float(np.ravel(m["elapsed"])[0])),
                         P=dict(r=float(p["fc_corr_last"]), r_post=float(p["fc_corr_post"]), fp=fp(p["C"], SCs[s - 1]), stop=str(p["stop"]), it=int(p["iterations"]))))
    summ = {}
    for k in V.K:
        g = lambda impl, coop, q: float(np.median([r[impl][q] for r in rows if r["K"] == k and r["coop"] == coop]))
        d = {impl: dict(r_signed=g(impl, 0, "r"), r_coop=g(impl, 1, "r"), fp_signed=g(impl, 0, "fp")) for impl in ("M", "P")}
        d["D_M"] = abs(d["M"]["r_signed"] - TARGET["r_signed"]) + abs(d["M"]["r_coop"] - TARGET["r_coop"]) + abs(d["M"]["fp_signed"] - TARGET["fp"])
        d["agreement"] = dict(r_mad=float(np.median([abs(r["P"]["r"] - r["M"]["r"]) for r in rows if r["K"] == k])),
                              fp_mad=float(np.median([abs(r["P"]["fp"] - r["M"]["fp"]) for r in rows if r["K"] == k and r["coop"] == 0])))
        d["agreement"]["within_0.03"] = bool(d["agreement"]["r_mad"] < 0.03 and d["agreement"]["fp_mad"] < 0.03)
        summ[k] = d
    kstar = min(summ, key=lambda k: summ[k]["D_M"]); none = summ[kstar]["D_M"] > 0.15
    va = ["K3"] + ([kstar] if kstar not in ("K1", "K3") and not none else [])
    out = dict(target=TARGET, rows=rows, summary=summ, K_star=kstar, no_config_reproduces=none, VA_configs=va)
    json.dump(out, open(sys.argv[1], "x"), indent=1)
    for k, d in summ.items():
        print(k, "| M r_signed %.3f r_coop %.3f FP %.3f D %.3f" % (d["M"]["r_signed"], d["M"]["r_coop"], d["M"]["fp_signed"], d["D_M"]),
              "| P r_signed %.3f r_coop %.3f FP %.3f" % (d["P"]["r_signed"], d["P"]["r_coop"], d["P"]["fp_signed"]), "| agreement", d["agreement"], flush=True)
    print("K* =", kstar, "| no config reproduces:", none, "| V-A configurations:", va, flush=True)
