"""Simulation 5 analysis (Sections 2.8.1, 2.8.2, 3.4.2; Supplementary Note 10): FP2alpha and the other thresholded measures, the
positive-control and counterexample criteria, and the descriptive ratios.
Usage: python e14b_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, json, hashlib
import numpy as np
import e14_analyze as A

P33 = DATA_ROOT + "results/p33/"; PREREG = P33 + "e14b_analyze_prereg.json"; ALPHA = 1e-4


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def thresholds(C, SC):
    w = C[A.supp(SC)]; mx = float(np.abs(w).max()); q95 = float(np.quantile(w[w > 0], 0.95)) if np.any(w > 0) else np.nan
    return dict(maxabs=mx, q95pos=q95, FP2a=float(np.mean(w < -2 * ALPHA)), FP1e3=float(np.mean(w < -1e-3)),
                FProb=float(np.mean(w < -0.05 * q95)) if np.isfinite(q95) else np.nan, FPm=float(np.mean(w < -max(0.05 * mx, 2 * ALPHA))),
                FP5=float(np.mean(w < -0.05 * mx)), FP0=float(np.mean(w < 0)))


def recovery(C, Ct, SC, inp):
    import luppi_hopf as L, luppi_r4e_e11_analyze as A11
    m = A.supp(SC); F = np.corrcoef(L.prep_emp(inp)); C0 = 0.2 * SC / SC.max(); w, t = C[m], Ct[m]
    pred = w < -2 * ALPHA; tn = t < 0; base = float(tn.mean())
    prec = float(tn[pred].mean()) if pred.any() else np.nan; rec = float(pred[tn].mean()) if tn.any() else np.nan
    a, af, a0 = A11.edge_auc(-C, Ct, SC), A11.edge_auc(-F, Ct, SC), A11.edge_auc(-C0, Ct, SC)
    return dict(AUC=a, AUC_FC=af, AUC_C0=a0, dAUC_FC=a - af, dAUC_C0=a - a0, precision=prec, recall=rec, base=base,
                enrichment=prec / base if base > 0 and np.isfinite(prec) else np.nan, no_pred=bool(not pred.any()), no_trueneg=bool(not tn.any()))


def ratio(num, den):
    if "median" not in num or "median" not in den:
        return "not analysable"
    if den["median"] == 0:
        return "undefined (denominator median 0)"
    nb, db = np.array(num["_boot"]), np.array(den["_boot"]); z = int((db == 0).sum())
    out = dict(median=float(num["median"] / den["median"]), zero_denominators=z)
    out["ci95"] = "unbounded" if z else [float(np.quantile(nb / db, 0.025)), float(np.quantile(nb / db, 0.975))]
    return out


def paired_ratio(xa, xb, key, rng):
    """ COOP / CMPH ratio of medians on the truth tags admitted in both cells, jointly resampled within each SC stratum."""
    a = {x["tag"]: x for x in xa}; b = {x["tag"]: x for x in xb}
    t = sorted(k for k in set(a) & set(b) if np.isfinite(a[k][key]) and np.isfinite(b[k][key])); st = {k: a[k]["stratum"] for k in t}
    g = [[k for k in t if st[k] == s] for s in (0, 1, 2)]; info = dict(n=len(t), excluded_a=len(a) - len(t), excluded_b=len(b) - len(t), per_stratum_n=[len(x) for x in g])
    if len(t) < A.MIN_N or min(len(x) for x in g) < A.MIN_STRATUM:
        return dict(info, label="not analysable")
    da = np.median([a[k][key] for k in t])
    db = np.median([b[k][key] for k in t])
    if db == 0:
        return dict(info, label="undefined (denominator median 0)")
    reps = []; z = 0
    for _ in range(A.B):
        ii = [k for x in g for k in np.array(x)[rng.integers(0, len(x), len(x))]]; den = np.median([b[k][key] for k in ii])
        if den == 0:
            z += 1; continue
        reps.append(np.median([a[k][key] for k in ii]) / den)
    return dict(info, median=float(da / db), zero_denominators=z, ci95="unbounded" if z else [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))])


def compute(rows, truth_root, rng):
    import luppi_r4e_e11 as E11
    cells = {}; per_fit = []
    for (cell, tag), d in rows.items():
        if "signed" not in d:
            continue
        s = d["signed"]; r = s["r"]; x = dict(tag=tag, stratum=d["stratum"], **thresholds(s["C"], s["SC"]), it=r["iterations"], fc=r["fc_corr"], P2=np.nan, dFC=np.nan)
        if "coop" in d:
            c = d["coop"]["r"]; x["P2"] = r["mse_ho"] - c["mse_ho"]; x["dFC"] = r["fc_corr"] - c["fc_corr"]
        if r["ttype"] == "CMPH":
            x.update(recovery(s["C"], E11.truth_c(int(tag.split("_")[1]), "CMPH", truth_root)[0], s["SC"], s["inp"]))
        cells.setdefault(cell, []).append(x); per_fit.append(dict(cell=cell, **x))
    keys = ("FP2a", "FP1e3", "FProb", "FPm", "FP5", "FP0", "maxabs", "q95pos", "P2", "dFC", "it", "fc",
            "AUC", "AUC_FC", "AUC_C0", "dAUC_FC", "dAUC_C0", "precision", "recall", "base", "enrichment")
    summ = {}
    for cell, xs in cells.items():
        strata = None if cell == "L-REAL-acc" else [x["stratum"] for x in xs]
        summ[cell] = {k: A.boot([x.get(k, np.nan) for x in xs], strata, rng) for k in keys}
    a = {x["tag"]: x for x in cells.get("L-COOP-d136", [])}; b = {x["tag"]: x for x in cells.get("L-COOP-raw", [])}; t = sorted(set(a) & set(b))
    dFP = {k: A.boot([a[s][k] - b[s][k] for s in t], [a[s]["stratum"] for s in t], rng) for k in ("FP2a", "FP1e3", "FProb")}
    c = summ.get("L-CMPH-d136", {}); P = "not analysable"; P_strict = "not analysable"
    if A.has(c, "AUC", "FP2a", "enrichment", "recall", "dAUC_C0"):
        P = bool(c["AUC"]["median"] >= 0.80 and c["AUC"]["ci95"][0] >= 0.75 and c["FP2a"]["ci95"][0] > 0.02 and c["enrichment"]["ci95"][0] > 2
                 and c["recall"]["ci95"][0] > 0.10 and c["dAUC_C0"]["ci95"][0] > 0)
        if A.has(c, "dAUC_FC"):
            P_strict = bool(P and c["dAUC_FC"]["ci95"][0] > 0)
    q = summ.get("L-COOP-d136", {}); q_num = "not analysable"
    if A.has(q, "FP2a") and "median" in dFP["FP2a"]:
        q_num = bool(q["FP2a"]["ci95"][0] > 0.02 and dFP["FP2a"]["ci95"][0] > 0.02)
    Q1 = q_num if P is True else "not interpreted (P not established)"
    re = summ.get("L-REAL-acc", {})
    R = {k: dict(coop_over_real=ratio(q.get(k, {}), re.get(k, {})), coop_over_cmph=ratio(q.get(k, {}), c.get(k, {})),
                 coop_over_cmph_paired=paired_ratio(cells.get("L-COOP-d136", []), cells.get("L-CMPH-d136", []), k, rng)) for k in ("FP2a", "FP1e3", "FProb")}
    cm = cells.get("L-CMPH-d136", [])
    counts = dict(n_cmph=len(cm), no_prediction=sum(bool(x.get("no_pred")) for x in cm), no_true_negative=sum(bool(x.get("no_trueneg")) for x in cm),
                  enrichment_n=int(np.isfinite([x.get("enrichment", np.nan) for x in cm]).sum()))
    if P is True and q_num is True:
        reading = ("under the released L1 configuration, with the L1-scale threshold, a cooperative construction without negative coupling still yields "
                   "negative weights that survive L1 after removal (cooperative construction only; no comparison with the real data is implied)")
    elif P is True and q_num is False:
        reading = "the joint counterexample criterion was not established under the released L1 configuration"
    elif P is False:
        reading = "the positive-control criterion was not established; Q1-L1 is not interpreted"
    else:
        reading = "not analysable"
    return dict(summary={k: {kk: A.strip(vv) for kk, vv in v.items()} for k, v in summ.items()}, paired={k: A.strip(v) for k, v in dFP.items()},
                P=P, P_strict=P_strict, Q1_L1_numeric=q_num, Q1_L1=Q1, R=R, cmph_counts=counts, reading=reading, per_fit=per_fit)


def show(res):
    for cell, v in res["summary"].items():
        print(cell, {k: (round(x["median"], 4), [round(c, 4) for c in x["ci95"]]) for k, x in v.items() if "median" in x}, flush=True)
    print("paired", {k: (round(v["median"], 4), [round(c, 4) for c in v["ci95"]]) if "median" in v else v for k, v in res["paired"].items()}, flush=True)
    print("P", res["P"], "| P_strict", res["P_strict"], "| Q1-L1 numeric", res["Q1_L1_numeric"], "| Q1-L1", res["Q1_L1"], flush=True)
    print("R", json.dumps(res["R"], default=float), flush=True); print("CMPH counts", res["cmph_counts"], flush=True)
    print("READING", res["reading"], flush=True)


def main():
    import e14b_run as R
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"] or sha("e14_analyze.py") != pre["helper_sha256"]:
        raise SystemExit("e14b_analyze.py or its helper differs from the pre-registration")
    rows, den, fsha = A.load(R); res = compute(rows, R.OUT, np.random.default_rng(141))
    res.update(prereg=pre, freeze=fsha, denominators=den)
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float); show(res)


if __name__ == "__main__":
    main()
