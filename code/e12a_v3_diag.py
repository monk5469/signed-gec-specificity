"""Exploratory transport of the readout law to simulated inputs (V3; Supplementary Note 6): joint-support distances of the
simulated rows and residuals inside versus outside the training support.
Usage: python e12a_v3_diag.py OUT_JSON"""
from paths import DATA_ROOT
import sys, json
import numpy as np
import e12a_validate as V, e12a_g_freeze as GF


def main():
    import scipy.io as sio, luppi_r4e_lib as Lib
    G = json.load(open(V.G)); a = json.load(open(GF.SRC))
    tr = [(v["neg_fc"], v["gs"]) for c in GF.VOL for s, v in a["rows"][c].items() if s not in GF.DEMO and np.isfinite(v["fp"])]
    T = np.array(tr); mu, sd = T.mean(0), T.std(0); Z = (T - mu) / sd
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf); thr = float(np.quantile(D.min(1), 0.95))
    rows, _ = V.rows_v3(); out = {}
    for r in rows:
        z = (np.array([r["nf"], r["gs"]]) - mu) / sd; r["nn"] = float(np.sqrt(((Z - z) ** 2).sum(1)).min()); r["joint"] = r["nn"] <= thr
        for m in GF.MODELS:
            r[f"res_{m}"] = float(r["fp"] - GF.predict(G, m, "fp", [r["nf"]], [r["gs"]])[0])
    for gname in sorted({r["group"] for r in rows}):
        R = [r for r in rows if r["group"] == gname]; J = [r for r in R if r["joint"]]
        out[gname] = dict(n=len(R), n_joint=len(J), nf=float(np.median([r["nf"] for r in R])), gs=float(np.median([r["gs"] for r in R])),
                          res_g_joint=float(np.median([r["res_g"] for r in J])) if J else None, res_gnf_joint=float(np.median([r["res_g_nf"] for r in J])) if J else None,
                          res_gnf_all=float(np.median([r["res_g_nf"] for r in R])))
    J = [r for r in rows if r["joint"]]
    summary = dict(threshold=thr, n=len(rows), n_joint=len(J),
                   joint=dict(mae_g=float(np.mean([abs(r["res_g"]) for r in J])), mae_gnf=float(np.mean([abs(r["res_g_nf"]) for r in J])),
                              med_g=float(np.median([r["res_g"] for r in J])), med_gnf=float(np.median([r["res_g_nf"] for r in J]))) if J else None)
    SCg = sio.loadmat(DATA_ROOT + "raw/luppi_demo/single/SC.mat")["SC"].astype(float); sc = {"group": int(Lib.support(SCg).sum())}
    for k in range(3):
        sc[f"demo{k}"] = int(Lib.support(Lib.demo_subject(k)[1]).sum())
    res = dict(summary=summary, by_cell=out, sc_support_entries=sc, note="exploratory; frequencies: real fits use group regional frequencies, simulated fits per-input")
    json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(dict(summary=summary, sc=sc), indent=1))
    for g, v in out.items():
        print(g, v, flush=True)


if __name__ == "__main__":
    main()
