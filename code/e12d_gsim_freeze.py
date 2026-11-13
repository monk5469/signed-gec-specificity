"""Simulation 4: fits and freezes the in-family law g_sim (a cubic in nf) on the cooperative simulations of Simulations 1-3
(446 rows). The frozen output is provenance/frozen_laws/e12d_gsim.json.
Usage: python e12d_gsim_freeze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, json, hashlib
import numpy as np


def predict(Gs, nf):
    nf = np.asarray(nf, float); return np.column_stack([np.ones(len(nf)), nf, nf ** 2, nf ** 3]) @ np.asarray(Gs["coef"])


if __name__ == "__main__":
    import e12a_validate as V, e12c_analyze as A, e12c_run as R, luppi_r4e_lib as Lib
    rows = [dict(src=x["group"].split(":")[0], nf=x["nf"], fp=x["fp"]) for x in V.rows_v3()[0] if x.get("ttype") == "COOP"]
    D, rej, fsha = A.load(); mg = Lib.support(np.load(R.OUT + "inp/SC_group.npy"))
    for (cell, st, tag), d in D.items():
        if st == "S1" and cell.startswith("COOP_") and "signed" in d:
            j, r = d["signed"]; nf, gs = V.inp_measures(Lib.bandpass(np.load(R.job_files(j)[0])))
            rows.append(dict(src="e12c", nf=nf, fp=float((r["_C"][mg] < 0).mean())))
    nf = np.array([x["nf"] for x in rows]); fp = np.array([x["fp"] for x in rows])
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(nf)), nf, nf ** 2, nf ** 3]), fp, rcond=None)[0]
    P = DATA_ROOT + "results/p33/"; srcf = {k: P + v for k, v in (("e10c", "r4e_e10c/r4e_e10c_freeze.json"), ("e11", "r4e_e11/r4e_e11_freeze.json"), ("e12c", "e12c/e12c_freeze.json"))}
    out = dict(model="OLS FP ~ 1 + nf + nf^2 + nf^3 (cooperative simulations only)", coef=coef.tolist(), n=len(rows),
               n_by_source={s: int(sum(x["src"].startswith(s) or x["src"] == "luppi_r4e_" + s for x in rows)) for s in ("luppi_r4e_e10c", "luppi_r4e_e11", "e12c")},
               train_range_nf=[float(nf.min()), float(nf.max())], in_sample_mae=float(np.mean(np.abs(fp - predict(dict(coef=coef), nf)))),
               source_freezes={k: hashlib.sha256(open(v, "rb").read()).hexdigest() for k, v in srcf.items()}, script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    json.dump(out, open(sys.argv[1], "x"), indent=1); print({k: v for k, v in out.items() if k != "source_freezes"}, flush=True)
