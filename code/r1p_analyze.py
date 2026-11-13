"""Simulation 6b analysis (Sections 2.8.4, 3.4.3): paired contrasts for the three candidate explanations, and the Simulation 6
criteria applied to each variant.
Usage: python r1p_analyze.py OUT_JSON"""
import sys, json
import numpy as np
import r1p_run as R
import r1_analyze as R1A

B = 2000; MIN_N = 20


def admit():
    fsha, fr = R.verify_freeze(); J = R.job_list(); rows = {}; rej = {}
    for j in J:
        why = None
        try:
            r = json.load(open(j["file"]))
        except Exception:
            why = "missing"
        else:
            why = ("field_mismatch" if any(r.get(q) != v for q, v in j.items()) else "not_done" if r.get("state") != "done" else
                   "wrong_prov" if r.get("prov") != R.job_prov(j, fsha, fr) else None)
            if why is None:
                try:
                    C = np.load(j["file"].replace(".json", "_C.npy"))
                    if C.shape != (100, 100) or not np.all(np.isfinite(C)) or R.sha_arr(C) != r.get("C_sha"):
                        why = "bad_matrix"
                except Exception:
                    why = "bad_matrix"
        if why:
            key = f"{j['variant']}|{j['arm']}|{j['model']}"; rej.setdefault(key, {}).setdefault(why, 0); rej[key][why] += 1; continue
        rows[(j["k"], j["variant"], j["arm"], j["model"])] = (j, r, C)
    return rows, rej, fsha, J


def metrics(rows):
    import e14_analyze as A, e14b_analyze as A14, luppi_r4e_lib as Lib
    from scipy.stats import spearmanr
    SC = R.R.group_sc(); m = A.supp(SC); C0 = 0.2 * SC / SC.max(); M = {}
    coop = {k: np.load(R.OUT + f"truth/k{k}_COOP_C.npy") for k in {key[0] for key in rows}}
    pooled = np.concatenate([c[m][c[m] > 0] for c in coop.values()]); edges = np.quantile(pooled, [0.25, 0.5, 0.75])   # strata fixed on the truths
    for (k, var, arm, model), (j, r, C) in rows.items():
        if model != "signed":
            continue
        tt = "CMPH" if arm == "Pplus" else "COOP"; Ct = np.load(R.OUT + f"truth/k{k}_{tt}_C.npy"); w = C[m]
        rec = dict(FP2a=float(np.mean(w < -2e-4)), FP_001=float(np.mean(w < -0.01)), FP0=float(np.mean(w < 0)),
                   drho=float(spearmanr(w, Ct[m])[0] - spearmanr(C0[m], Ct[m])[0]), iterations=float(r.get("iterations", np.nan)),
                   early_stop=float(r.get("stop") == "error_increase"))
        tc = coop[k][m]; lab = np.where(tc == 0, 0, 1 + np.searchsorted(edges, tc))          # strata by the COOP truth weight
        for i, s in enumerate(("zero", "Q1", "Q2", "Q3", "Q4")):
            sel = lab == i
            rec[f"within_{s}"] = float(np.mean(w[sel] < -2e-4)) if sel.any() else np.nan; rec[f"contrib_{s}"] = float(np.sum(w[sel] < -2e-4) / w.size)
        co = rows.get((k, var, arm, "coop")); rec["P2"] = float(r["mse_ho"] - co[1]["mse_ho"]) if co else np.nan
        if arm == "Pplus":
            rec.update({q: v for q, v in A14.recovery(C, Ct, SC, Lib.bandpass(np.load(R.job_files(j)[0]))).items() if isinstance(v, (int, float))})
        M[(k, var, arm)] = rec
    return M


def cell(M, ks, var, arm, key, rng):
    return R1A.boot([M[(k, var, arm)][key] for k in ks if (k, var, arm) in M], rng)


def paired(M, ks, va, aa, vb, ab, key, rng):
    return R1A.boot([M[(k, va, aa)][key] - M[(k, vb, ab)][key] for k in ks if (k, va, aa) in M and (k, vb, ab) in M], rng)


def criteria(M, ks, var, rng):
    out = {a: {q: cell(M, ks, var, a, q, rng) for q in ("FP2a", "FP_001", "FP0", "drho", "P2")} for a in R.ARMS}
    out["A2_minus_A1"] = paired(M, ks, var, "A2", var, "A1", "FP2a", rng); out["A2_minus_A0"] = paired(M, ks, var, "A2", var, "A0", "FP2a", rng)
    out["A2_minus_A1_FP001"] = paired(M, ks, var, "A2", var, "A1", "FP_001", rng)
    out["Pplus_rec"] = {q: cell(M, ks, var, "Pplus", q, rng) for q in ("AUC", "recall", "enrichment", "dAUC_C0", "dAUC_FC")}
    ok = lambda *c: all(x.get("n", 0) >= MIN_N for x in c)
    a0, P = out["A0"], out["Pplus_rec"]
    out["R1pa"] = bool(a0["FP2a"]["ci95"][1] < 0.02 and a0["drho"]["ci95"][0] > 0) if ok(a0["FP2a"], a0["drho"]) else "not analysable"
    out["R1pc"] = (bool(P["AUC"]["median"] >= 0.80 and P["AUC"]["ci95"][0] >= 0.75 and out["Pplus"]["FP2a"]["ci95"][0] > 0.02 and P["recall"]["ci95"][0] > 0.10
                        and P["enrichment"]["ci95"][0] > 2 and P["dAUC_C0"]["ci95"][0] > 0) if ok(*P.values(), out["Pplus"]["FP2a"]) else "not analysable")
    out["R1pb"] = bool(out["A2"]["FP2a"]["ci95"][0] > 0.02 and out["A2_minus_A1"]["ci95"][0] > 0.02) if ok(out["A2"]["FP2a"], out["A2_minus_A1"]) else "not analysable"
    return out


def main():
    R.check_rules(); rows, rej, fsha, J = admit(); M = metrics(rows); P = json.load(open(R.PANEL))
    valid = {tuple(v) for v in P["valid_units"]}; ks = sorted(int(k) for k in P["truths"] if (int(k), "short") in valid)
    kl = sorted(int(k) for k in P["truths"] if (int(k), "long") in valid)
    inc = [j["key"] for j in J if j["variant"] == "ftrue" and (j["k"], j["variant"], j["arm"], j["model"]) not in rows]
    rng = np.random.default_rng(20261022)
    res = dict(freeze=fsha, n_short=len(ks), n_long=len(kl), invalid=P["invalid"], rejected=rej, incomplete_ftrue=len(inc))
    res["ftrue"] = criteria(M, ks, "ftrue", rng); res["fest"] = criteria(M, ks, "fest", rng); res["ftrue_long"] = criteria(M, kl, "ftrue_long", rng)
    res["freq_effect_A0"] = paired(M, ks, "ftrue", "A0", "fest", "A0", "FP2a", rng)
    res["freq_effect_A2"] = paired(M, ks, "ftrue", "A2", "fest", "A2", "FP2a", rng)
    res["ftrue_full"] = {a: {q: cell(M, ks, "ftrue_full", a, q, rng) for q in ("FP2a", "FP_001", "drho", "iterations")} for a in R.FULL_ARMS}
    res["stop_effect"] = {a: paired(M, ks, "ftrue_full", a, "ftrue", a, "FP2a", rng) for a in R.FULL_ARMS}
    res["full_A2_minus_A0"] = paired(M, ks, "ftrue_full", "A2", "ftrue_full", "A0", "FP2a", rng)
    res["strata"] = {v: {a: {s: dict(within=cell(M, ks if v != "ftrue_long" else kl, v, a, f"within_{s}", rng), contrib=cell(M, ks if v != "ftrue_long" else kl, v, a, f"contrib_{s}", rng))
                             for s in ("zero", "Q1", "Q2", "Q3", "Q4")} for a in (R.FULL_ARMS if v == "ftrue_full" else ("A0", "A1", "A2"))}
                     for v in R.VARIANTS}
    res["stopping"] = {v: {a: dict(iterations=cell(M, ks if v != "ftrue_long" else kl, v, a, "iterations", rng), early_stop_share=cell(M, ks if v != "ftrue_long" else kl, v, a, "early_stop", rng))
                           for a in (R.FULL_ARMS if v == "ftrue_full" else R.ARMS)} for v in R.VARIANTS}
    res["length_effect_A0"] = paired(M, sorted(set(ks) & set(kl)), "ftrue_long", "A0", "ftrue", "A0", "FP2a", rng)
    m = res["ftrue"]
    if inc:
        res["verdict"] = "incomplete; no confirmatory verdict"
    elif m["R1pa"] is True and m["R1pc"] is True:
        res["verdict"] = "R1'b holds" if m["R1pb"] is True else "R1'b does not hold"
    else:
        res["verdict"] = f"R1'b not interpreted (R1'a {m['R1pa']}, R1'c {m['R1pc']})"
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float)
    print("verdict:", res["verdict"], flush=True)
    for v in ("ftrue", "fest", "ftrue_long"):
        x = res[v]; print(v, "| A0 FP2a", x["A0"]["FP2a"], "| A2 FP2a", x["A2"]["FP2a"], "| A2-A1", x["A2_minus_A1"], "| A0 FP_001", x["A0"]["FP_001"], "| R1'a/c/b", x["R1pa"], x["R1pc"], x["R1pb"], flush=True)
    print("freq effect A0 (ftrue - fest)", res["freq_effect_A0"], "| A2", res["freq_effect_A2"], "| length effect A0 (long - short)", res["length_effect_A0"], flush=True)
    print("full (no early stop):", {a: res["ftrue_full"][a]["FP2a"] for a in R.FULL_ARMS}, "| stop effect", res["stop_effect"], "| full A2 - A0", res["full_A2_minus_A0"], flush=True)


if __name__ == "__main__":
    main()
