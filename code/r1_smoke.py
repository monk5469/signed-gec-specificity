"""Simulation 6, truth selection and input check (Section 2.8.3): draws the 30 truths and checks, without estimator fits, the
negative-FC level of their simulated FC against the condition of the original cooperative test.
Usage: python r1_smoke.py OUT_JSON"""
import sys, json, time
import numpy as np


def stats(x):
    import luppi_hopf as L
    F = np.corrcoef(L.prep_emp(x)); iu = np.triu_indices(100, 1)
    return dict(nf=float((F[iu] < 0).mean()), gs=float(np.var(x.mean(0)) / np.mean(x.var(1))), mean_fc=float(F[iu].mean()))


if __name__ == "__main__":
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib, r5_groupfc as R5
    X, pairs, rej, fsha = R5.admitted("acc")
    pick = sorted(np.random.default_rng(20261017).choice(len(pairs), 30, replace=False).tolist())
    f = np.load(X.OUT + "e5/freq_REST1.npy"); sim = L.make_simulator(100, Lib.T_LEN, 3000.0); cfg = dict(L.HEADER); rows = []
    for i in pick:
        tag, idx, d = pairs[i]; key = jax.random.PRNGKey(31_000_000 + idx); row = dict(tag=tag)
        for mdl in ("coop", "signed"):
            x = np.asarray(sim(d[mdl][2], f, cfg["a"], cfg["sigma"], key))
            row[mdl] = dict(raw=stats(x), bp=stats(Lib.bandpass(x)), C_neg=float((d[mdl][2][Lib.support(np.load(X.OUT + "e5/SC.npy"))] < 0).mean()))
        row["real_input"] = stats(np.load(X.OUT + f"e5/{tag}_REST1_input.npy")); rows.append(row)
    med = lambda path: float(np.median([eval("r" + path) for r in rows]))
    summ = {p: med(p) for p in ('["coop"]["raw"]["nf"]', '["coop"]["bp"]["nf"]', '["coop"]["raw"]["gs"]', '["coop"]["C_neg"]',
                                '["signed"]["raw"]["nf"]', '["signed"]["bp"]["nf"]', '["real_input"]["nf"]', '["real_input"]["gs"]')}
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), freeze=fsha, picked=[pairs[i][0] for i in pick], rows=rows, medians=summ), open(sys.argv[1], "x"), indent=1)
    for k, v in summ.items():
        print(f"{k:28s} median {v:.3f}", flush=True)
    print("coop raw nf range", round(min(r["coop"]["raw"]["nf"] for r in rows), 3), "-", round(max(r["coop"]["raw"]["nf"] for r in rows), 3), flush=True)
