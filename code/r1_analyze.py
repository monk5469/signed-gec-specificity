"""Simulation 6 analysis (Sections 2.8.3, 3.4.3; Supplementary Note 10): the three sequential criteria (low-negative baseline,
positive control, removal effect) and the sensitivity analyses.
Usage: python r1_analyze.py OUT_JSON"""
import sys, os, json, glob
import numpy as np
import r1_run as R

ALPHA = 1e-4; B = 2000; MIN_N = 20; MIN_STAB = 6


def admit():
    fsha, fr = R.verify_freeze(); J = R.job_list(); rows = {}; rej = {}
    for j in J:
        why = None
        try:
            r = json.load(open(j["file"]))
        except Exception:
            why = "missing"
        else:
            Cp = j["file"].replace(".json", "_C.npy")
            why = ("field_mismatch" if any(r.get(q) != v for q, v in j.items()) else "not_done" if r.get("state") != "done" else
                   "wrong_prov" if r.get("prov") != R.job_prov(j, fsha, fr) else None)
            if why is None:
                try:
                    C = np.load(Cp)
                    if C.shape != (100, 100) or not np.all(np.isfinite(C)) or R.sha_arr(C) != r.get("C_sha"):
                        why = "bad_matrix"
                except Exception:
                    why = "bad_matrix"
        if why:
            rej.setdefault((j["sg"], j["g"], j["seed"], j["arm"], j["model"]), {}).setdefault(why, 0)
            rej[(j["sg"], j["g"], j["seed"], j["arm"], j["model"])][why] += 1; continue
        rows[(j["k"], j["sg"], j["g"], j["seed"], j["arm"], j["model"])] = (j, r, C)
    return rows, rej, fsha, J


def metrics(rows):
    import e14_analyze as A, e14b_analyze as A14, luppi_r4e_lib as Lib
    from scipy.stats import spearmanr
    SC = R.group_sc(); m = A.supp(SC); C0 = 0.2 * SC / SC.max(); M = {}
    for (k, sg, g, seed, arm, model), (j, r, C) in rows.items():
        if model != "signed":
            continue
        tt = "CMPH" if arm == "Pplus" else "COOP"; Ct = np.load(R.OUT + f"truth/k{k}_{tt}_C.npy"); th = A.thresholds(C, SC)
        rec = dict(FP2a=th["FP2a"], FP0=th["FP0"], FP1e3=float(np.mean(C[m] < -1e-3)),
                   drho=float(spearmanr(C[m], Ct[m])[0] - spearmanr(C0[m], Ct[m])[0]), rho=float(spearmanr(C[m], Ct[m])[0]))
        co = rows.get((k, sg, g, seed, arm, "coop"))
        rec["P2"] = float(r["mse_ho"] - co[1]["mse_ho"]) if co else np.nan
        if arm == "Pplus":
            rec.update({q: v for q, v in A14.recovery(C, Ct, SC, Lib.bandpass(np.load(R.job_files(j)[0]))).items() if isinstance(v, (int, float))})
        M[(k, sg, g, seed, arm)] = rec
    return M


def boot(vals, rng):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    bs = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(B)]
    return dict(n=int(v.size), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def cell(M, ks, sg, g, seed, arm, key, rng):
    return boot([M[(k, sg, g, seed, arm)][key] for k in ks if (k, sg, g, seed, arm) in M], rng)


def paired(M, ks, sg, g, seed, a, b, key, rng):
    d = [M[(k, sg, g, seed, a)][key] - M[(k, sg, g, seed, b)][key] for k in ks if (k, sg, g, seed, a) in M and (k, sg, g, seed, b) in M]
    return boot(d, rng)


def criteria(M, ks, sg, g, seed, rng, nmin):
    out = dict(A0=cell(M, ks, sg, g, seed, "A0", "FP2a", rng), A0_drho=cell(M, ks, sg, g, seed, "A0", "drho", rng),
               A1=cell(M, ks, sg, g, seed, "A1", "FP2a", rng), A2=cell(M, ks, sg, g, seed, "A2", "FP2a", rng),
               A2_minus_A1=paired(M, ks, sg, g, seed, "A2", "A1", "FP2a", rng), A2_minus_A0=paired(M, ks, sg, g, seed, "A2", "A0", "FP2a", rng),
               P2_A2=cell(M, ks, sg, g, seed, "A2", "P2", rng), P2_A0=cell(M, ks, sg, g, seed, "A0", "P2", rng),
               Pplus={q: cell(M, ks, sg, g, seed, "Pplus", q, rng) for q in ("FP2a", "AUC", "recall", "enrichment", "dAUC_C0", "dAUC_FC", "drho")})
    ok = lambda *c: all(x.get("n", 0) >= nmin for x in c)
    P = out["Pplus"]
    out["R1a"] = (bool(out["A0"]["ci95"][1] < 0.02 and out["A0_drho"]["ci95"][0] > 0) if ok(out["A0"], out["A0_drho"]) else "not analysable")
    out["R1c"] = (bool(P["AUC"]["median"] >= 0.80 and P["AUC"]["ci95"][0] >= 0.75 and P["FP2a"]["ci95"][0] > 0.02 and P["recall"]["ci95"][0] > 0.10
                       and P["enrichment"]["ci95"][0] > 2 and P["dAUC_C0"]["ci95"][0] > 0) if ok(*P.values()) else "not analysable")
    out["R1b"] = (bool(out["A2"]["ci95"][0] > 0.02 and out["A2_minus_A1"]["ci95"][0] > 0.02) if ok(out["A2"], out["A2_minus_A1"]) else "not analysable")
    return out


def main():
    R.check_rules()
    rows, rej, fsha, J = admit(); M = metrics(rows); P = json.load(open(R.PANEL)); man = json.load(open(R.MAN))
    valid = {tuple(v) for v in P["valid_units"]}; ks = sorted(k for k in range(1, R.N_TRUTH + 1) if (k, "s01", 0) in valid)
    tier0 = [j for j in J if j["tier"] == 0]; incomplete = [j["key"] for j in tier0 if (j["k"], j["sg"], j["g"], j["seed"], j["arm"], j["model"]) not in rows]
    rng = np.random.default_rng(20261018); res = dict(freeze=fsha, n_valid_truths=len(ks), invalid_units=P["invalid"],
                                                      rejected={"|".join(map(str, q)): v for q, v in rej.items()}, incomplete_tier0=len(incomplete))
    main_ = criteria(M, ks, "s01", 0, 11, rng, MIN_N); res["main_s01"] = main_
    if incomplete:
        res["verdict"] = "incomplete; no confirmatory verdict"
    elif main_["R1a"] is True and main_["R1c"] is True:
        res["verdict"] = "R1b holds" if main_["R1b"] is True else "R1b does not hold"
    else:
        res["verdict"] = "R1b not interpreted (" + ("R1a " + str(main_["R1a"]) if main_["R1a"] is not True else "") + (" R1c " + str(main_["R1c"]) if main_["R1c"] is not True else "") + ")"
    ks001 = sorted(k for k in range(1, R.N_TRUTH + 1) if (k, "s001", 0) in valid); res["s001"] = criteria(M, ks001, "s001", 0, 11, rng, MIN_N)
    res["secondary"] = {a: dict(FP2a=cell(M, ks, "s01", 0, 11, a, "FP2a", rng), minus_A1=paired(M, ks, "s01", 0, 11, a, "A1", "FP2a", rng),
                                minus_A0=paired(M, ks, "s01", 0, 11, a, "A0", "FP2a", rng), drho=cell(M, ks, "s01", 0, 11, a, "drho", rng))
                        for a in ("A0x", "A3cal", "A3cal_svd", "A3ind", "A4")}
    nf = {c["k"]: c["input"]["A0"]["neg_fc"] for c in man["calibration"] if c.get("valid") and c["sg"] == "s01" and c["g"] == 0}
    sub = sorted(k for k in ks if nf.get(k, 0) >= 0.20); res["subset_nf020"] = dict(truths=sub, a0_input_nf={str(k): nf[k] for k in sub},
                                                                                    stats=criteria(M, sub, "s01", 0, 11, rng, 8) if len(sub) >= 8 else "descriptive only (< 8 truths)")
    stab = P["stability_truths"]; combos = [(0, 11)] + list(R.STAB_COMBOS); st = {}
    for g, seed in combos:
        kk = [k for k in stab if (k, "s01", g) in valid]; c = criteria(M, kk, "s01", g, seed, rng, MIN_STAB)
        st[f"g{g}_seed{seed}"] = dict(n=len(kk), R1b=c["R1b"], R1c=c["R1c"], A2=c["A2"], A2_minus_A1=c["A2_minus_A1"])
    agree = sum(v["R1b"] == main_["R1b"] for v in st.values()); res["stability"] = dict(combos=st, agree_with_main=agree, of=len(st),
                                                                                       label="depends on random stream" if agree < 5 else "stable")
    res["input_nf_A0_median"] = float(np.median(list(nf.values()))) if nf else np.nan
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float)
    print("verdict:", res["verdict"], "| R1a", main_["R1a"], "R1c", main_["R1c"], "R1b", main_["R1b"], "| stability", res["stability"]["label"], agree, "/", len(st), flush=True)
    for q in ("A0", "A0_drho", "A1", "A2", "A2_minus_A1", "A2_minus_A0"):
        print(q, main_[q], flush=True)
    print("Pplus", {q: (v.get("median"), v.get("ci95")) for q, v in main_["Pplus"].items()}, flush=True)


if __name__ == "__main__":
    main()
