"""Analysis of the initial exploratory L1 run (Supplementary Note 10), which used a threshold relative to max|C|. Its functions
are also used by the analyses of Simulations 5, 6 and 6b.
Usage: python e14_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

P33 = DATA_ROOT + "results/p33/"; PREREG = P33 + "e14_analyze_prereg.json"; MIN_N = 15; MIN_STRATUM = 5; B = 2000; ALPHA = 1e-4


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def supp(SC):
    return (SC != 0) & ~np.eye(100, dtype=bool)


def thresholds(C, SC):
    w = C[supp(SC)]; mx = float(np.abs(w).max()); cut5 = 0.05 * mx
    return dict(maxabs=mx, cut5=cut5, ratio_cut5_2a=cut5 / (2 * ALPHA), FP0=float(np.mean(w < 0)), FP2a=float(np.mean(w < -2 * ALPHA)),
                FP5=float(np.mean(w < -cut5)), FPm=float(np.mean(w < -max(cut5, 2 * ALPHA))))


def recovery(C, Ct, SC, inp):
    import luppi_hopf as L, luppi_r4e_e11_analyze as A11
    m = supp(SC); F = np.corrcoef(L.prep_emp(inp)); C0 = 0.2 * SC / SC.max(); w, t = C[m], Ct[m]
    pred = w < -max(0.05 * np.abs(w).max(), 2 * ALPHA); tn = t < 0; base = float(tn.mean())
    prec = float(tn[pred].mean()) if pred.any() else np.nan; rec = float(pred[tn].mean()) if tn.any() else np.nan
    a, af, a0 = A11.edge_auc(-C, Ct, SC), A11.edge_auc(-F, Ct, SC), A11.edge_auc(-C0, Ct, SC)
    return dict(AUC=a, AUC_FC=af, AUC_C0=a0, dAUC_FC=a - af, dAUC_C0=a - a0, precision=prec, recall=rec, base=base,
                enrichment=prec / base if base > 0 and np.isfinite(prec) else np.nan)


def load(R):
    import luppi_r4e_lib as Lib
    fsha, fr = R.verify_freeze(); exp = {j["key"]: j for j in R.job_list()}; rej = dict(malformed=0, unexpected=0, field=0, state=0, prov=0, dup=0, matrix=0)
    seen = set(); out = {}
    for p in sorted(glob.glob(R.OUT + "jobs/*.json")):
        try:
            r = json.load(open(p)); assert isinstance(r, dict)
        except Exception:
            rej["malformed"] += 1; continue
        j = exp.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if any(r.get(k) != v for k, v in j.items()):
            rej["field"] += 1; continue
        if r.get("state") != "done":
            rej["state"] += 1; continue
        if r.get("prov") != R.job_prov(j, fsha, fr):
            rej["prov"] += 1; continue
        if j["key"] in seen:
            rej["dup"] += 1; continue
        seen.add(j["key"])
        try:
            C = np.load(j["file"].replace(".json", "_C.npy"))
        except Exception:
            C = None
        if C is None or C.shape != (100, 100) or not np.all(np.isfinite(C)) or sha_arr(C) != r.get("C_sha"):
            rej["matrix"] += 1; continue
        if j["kind"] == "sim":
            SC = Lib.demo_subject(j["stratum"])[1]; inp = Lib.bandpass(np.load(R.job_files(j)[0]))
        else:
            SC = np.load(R.job_files(j)[0]); inp = np.load(R.job_files(j)[2])
        out.setdefault((j["cell"], j["tag"]), dict(stratum=j.get("stratum")))[j["model"]] = dict(r=r, C=C, SC=SC, inp=inp)
    return out, dict(expected=len(exp), admitted=len(seen) - rej["matrix"], rejected=rej), fsha


def header_ref():
    """FP5 / FPm recomputed from the frozen HEADER signed fits of the same inputs (descriptive pairing)."""
    import luppi_r4e_e11 as E11, luppi_r4e_ppc as W, luppi_r4e_lib as Lib
    ref = {}; fsha, fr = E11.verify_freeze()
    for j in E11.job_list():
        if j["model"] != "signed" or not os.path.exists(j["file"]) or (j["ttype"], j["cond"]) not in {("COOP", "N49-d136"), ("COOP", "N49-raw"), ("CMPH", "N49-d136")}:
            continue
        r = json.load(open(j["file"])); C = np.load(j["file"].replace(".json", "_C.npy"))
        if r.get("state") == "done" and r.get("prov") == E11.job_prov(j, fsha, fr) and r.get("key") == j["key"] and sha_arr(C) == r.get("C_sha"):
            ref[(j["ttype"], j["cond"], j["tag"])] = thresholds(C, Lib.demo_subject(j["stratum"])[1])
    X = W.X; W.point("acc"); asha, afr = X.verify_freeze("e5"); SC = np.load(X.OUT + "e5/SC.npy")
    for j in X.job_list("e5"):
        if j["model"] != "signed" or not os.path.exists(j["file"]):
            continue
        r = json.load(open(j["file"])); C = np.load(j["file"].replace(".json", "_C.npy"))
        if r.get("state") == "done" and r.get("prov") == X.job_prov(j, asha, afr) and r.get("key") == j["key"] and sha_arr(C) == r.get("C_sha"):
            ref[("REAL", "acc", j["tag"])] = thresholds(C, SC)
    return ref


def boot(vals, strata, rng):
    v = np.asarray(vals, float); ok = np.isfinite(v); v = v[ok]
    if strata is None:
        if len(v) < MIN_N:
            return dict(label="not analysable", n=int(len(v)))
        bs = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
        return dict(n=int(len(v)), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], _boot=bs)
    st = np.asarray(strata)[ok]; groups = [v[st == k] for k in (0, 1, 2)]
    if len(v) < MIN_N or min(len(g) for g in groups) < MIN_STRATUM:
        return dict(label="not analysable", n=int(len(v)), per_stratum_n=[int(len(g)) for g in groups])
    bs = [np.median(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(B)]
    return dict(n=int(len(v)), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                per_stratum=[dict(n=int(len(g)), median=float(np.median(g))) for g in groups], _boot=bs)


def strip(d):
    return {k: v for k, v in d.items() if k != "_boot"}


def has(d, *ks):
    return all("median" in d.get(k, {}) for k in ks)


def compute(rows, ref, rng):
    import luppi_r4e_e11 as E11
    cells = {}; per_fit = []
    for (cell, tag), d in rows.items():
        if "signed" not in d:
            continue
        s = d["signed"]; r = s["r"]; th = thresholds(s["C"], s["SC"])
        hr = ref.get((r["ttype"], r["cond"], tag), {})
        x = dict(tag=tag, stratum=d["stratum"], **th, it=r["iterations"], fc=r["fc_corr"], P2=np.nan, dFC=np.nan,
                 FP5_header=hr.get("FP5", np.nan), FPm_header=hr.get("FPm", np.nan))
        if "coop" in d:
            c = d["coop"]["r"]; x["P2"] = r["mse_ho"] - c["mse_ho"]; x["dFC"] = r["fc_corr"] - c["fc_corr"]
        if r["ttype"] == "CMPH":
            x.update(recovery(s["C"], E11.truth_c(int(tag.split("_")[1]), "CMPH")[0], s["SC"], s["inp"]))
        cells.setdefault(cell, []).append(x); per_fit.append(dict(cell=cell, **{k: v for k, v in x.items()}))
    keys = ("FPm", "FP5", "FP2a", "FP0", "maxabs", "cut5", "ratio_cut5_2a", "P2", "dFC", "it", "fc", "FP5_header", "FPm_header",
            "AUC", "AUC_FC", "AUC_C0", "dAUC_FC", "dAUC_C0", "precision", "recall", "base", "enrichment")
    summ = {}
    for cell, xs in cells.items():
        strata = None if cell == "L-REAL-acc" else [x["stratum"] for x in xs]
        summ[cell] = {k: boot([x.get(k, np.nan) for x in xs], strata, rng) for k in keys}
    a = {x["tag"]: x for x in cells.get("L-COOP-d136", [])}; b = {x["tag"]: x for x in cells.get("L-COOP-raw", [])}; t = sorted(set(a) & set(b))
    dFPm = boot([a[k]["FPm"] - b[k]["FPm"] for k in t], [a[k]["stratum"] for k in t], rng)
    c = summ.get("L-CMPH-d136", {}); P = "not analysable"; P_strict = "not analysable"
    if has(c, "AUC", "FPm", "enrichment", "recall", "dAUC_C0"):
        P = bool(c["AUC"]["median"] >= 0.80 and c["AUC"]["ci95"][0] >= 0.75 and c["FPm"]["ci95"][0] > 0.02 and c["enrichment"]["ci95"][0] > 2
                 and c["recall"]["ci95"][0] > 0.10 and c["dAUC_C0"]["ci95"][0] > 0)
        if has(c, "dAUC_FC"):
            P_strict = bool(P and c["dAUC_FC"]["ci95"][0] > 0)
    q = summ.get("L-COOP-d136", {}); q_num = "not analysable"
    if has(q, "FPm") and "median" in dFPm:
        q_num = bool(q["FPm"]["ci95"][0] > 0.02 and dFPm["ci95"][0] > 0.02)
    Q1 = q_num if P is True else "not interpreted (P not established)"
    Rr = dict(descriptive=True); re = summ.get("L-REAL-acc", {}).get("FPm", {})
    if "median" in re and "median" in q.get("FPm", {}):
        Rr.update(real_FPm=strip(re), coop_FPm=strip(q["FPm"]))
        if re["median"] == 0:
            Rr["ratio"] = "undefined (real median 0)"
        else:
            rb = np.array(re["_boot"]); qb = np.array(q["FPm"]["_boot"]); z = int((rb == 0).sum())
            ratio = dict(median=float(q["FPm"]["median"] / re["median"]), zero_denominators=z)
            ratio["ci95"] = "unbounded" if z else [float(np.quantile(qb / rb, 0.025)), float(np.quantile(qb / rb, 0.975))]
            Rr["ratio"] = ratio
    if P is True and q_num is True:
        reading = ("under the released L1 configuration, the cooperative construction without negative coupling still yields above-threshold negative "
                   "weights after removal (cooperative construction only; no comparison with the real data is implied)")
    elif P is True and q_num is False:
        reading = "the joint counterexample criterion was not established under the released L1 configuration"
    elif P is False:
        reading = "the positive-control criterion was not established; Q1-L1 is not interpreted"
    else:
        reading = "not analysable"
    return dict(summary={k: {kk: strip(vv) for kk, vv in v.items()} for k, v in summ.items()}, paired=dict(dFPm=strip(dFPm)), P=P, P_strict=P_strict,
                Q1_L1_numeric=q_num, Q1_L1=Q1, R=Rr, reading=reading, per_fit=per_fit)


def main():
    import e14_run as R
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("e14_analyze.py differs from the pre-registration")
    rows, den, fsha = load(R); res = compute(rows, header_ref(), np.random.default_rng(14))
    res.update(prereg=pre, freeze=fsha, denominators=den)
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float)
    show(res)


def show(res):
    for cell, v in res["summary"].items():
        print(cell, {k: (round(x["median"], 4), [round(c, 4) for c in x["ci95"]]) for k, x in v.items() if "median" in x}, flush=True)
    print("dFPm", res["paired"]["dFPm"].get("median"), res["paired"]["dFPm"].get("ci95"), "| P", res["P"], "| P_strict", res["P_strict"],
          "| Q1-L1 numeric", res["Q1_L1_numeric"], "| Q1-L1", res["Q1_L1"], "| R", res["R"].get("ratio"), "| READING", res["reading"], flush=True)


if __name__ == "__main__":
    main()
