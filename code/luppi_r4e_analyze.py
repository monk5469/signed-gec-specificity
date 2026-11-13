"""Shared admission functions of the simulation analyses: freeze verification, record admission and truth tables."""
import sys, os, json, glob
import numpy as np
from scipy import stats

W = (1 / 3, 1 / 3, 1 / 3)
P2_BOUNDS = (-8.0, 8.0)
E0_RULE = dict(max_neg_abs2alpha=0.01, min_spearman=0.6)
ALPHA2 = 0.0125 / 2


def load(jobs_dir, group):
    """Admission: the group's freeze must verify (code incl. this script,
 fixture result, raw inputs, manifests, every job input unchanged). A record is admitted only if its state is "done", its
 provenance equals the provenance the frozen job would carry (freeze hash, group, input hashes), its key is an expected
 key of the frozen group, its file name matches that key, and it is not a duplicate. Rejections are counted by reason."""
    import luppi_r4e_run as RUN
    fsha, fr = RUN.verify_freeze(group)
    J, _ = RUN.job_list(fr["exps"]); expected = {j["key"]: j for j in J}
    R, seen, rej = [], set(), dict(not_done=0, wrong_prov=0, unexpected=0, duplicate=0)
    for p in sorted(glob.glob(os.path.join(jobs_dir, "*.json"))):
        r = json.load(open(p))
        j = expected.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != RUN.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if r["key"] in seen:
            rej["duplicate"] += 1; continue
        seen.add(r["key"]); R.append(r)
    return R, rej, fsha


def load_C(r):
    """Fitted GEC of a record, verified against its recorded C_sha."""
    import luppi_r4e_run as RUN
    C = np.load(r["file"].replace(".json", "_C.npy"))
    if RUN.sha_arr(C) != r["C_sha"]:
        raise SystemExit(f"fitted matrix of {r['key']} does not match its C_sha")
    return C


def strat_interval(vals_by_stratum, level_two_sided_alpha=ALPHA2 * 2):
    n = [len(v) for v in vals_by_stratum]
    if any(k < 2 for k in n):
        return dict(theta=np.nan, lo=np.nan, hi=np.nan, degenerate=True, n=n)
    m = [float(np.mean(v)) for v in vals_by_stratum]; s2 = [float(np.var(v, ddof=1)) for v in vals_by_stratum]
    if any(x == 0 for x in s2):
        return dict(theta=float(sum(w * x for w, x in zip(W, m))), lo=np.nan, hi=np.nan, degenerate=True, n=n)
    V = sum(w ** 2 * s / k for w, s, k in zip(W, s2, n)); nu = V ** 2 / sum((w ** 2 * s / k) ** 2 / (k - 1) for w, s, k in zip(W, s2, n))
    th = sum(w * x for w, x in zip(W, m)); t = stats.t.ppf(1 - level_two_sided_alpha / 2, nu)
    return dict(theta=th, lo=th - t * np.sqrt(V), hi=th + t * np.sqrt(V), nu=nu, n=n, per_stratum_mean=m, degenerate=False)


def truth_table(R, exp, family, branch, extra=lambda r: True):
    """tag -> {seed -> {model -> record}}"""
    T = {}
    for r in R:
        if r["exp"] == exp and r["family"] == family and r["branch"] == branch and extra(r):
            T.setdefault(r["tag"], {}).setdefault(r["seed"], {})[r["model"]] = r
    return T


def p1_values(T):
    """Truth-level P1; only seeds with a complete signed + coop pair count (same completeness rule as P2 / the gate)."""
    out = {}
    for tag, seeds in T.items():
        pairs = [x for x in seeds.values() if "signed" in x and "coop" in x]
        if len(pairs) >= 2:
            out[tag] = (pairs[0]["signed"]["stratum"], float(np.mean([x["signed"]["metrics"]["frac_neg"] for x in pairs])))
    return out


def p2_values(T):
    out = {}
    for tag, seeds in T.items():
        d = [x["signed"]["mse_ho"] - x["coop"]["mse_ho"] for x in seeds.values() if "signed" in x and "coop" in x]
        if len(d) >= 2:
            st = next(x["signed"]["stratum"] for x in seeds.values() if "signed" in x)
            out[tag] = (st, float(np.mean(d)))
    return out


def by_stratum(vals):
    S = [[], [], []]
    for st, v in vals.values():
        S[st].append(v)
    return S


def gate(vals, n_planned=(10, 10, 10)):
    S = by_stratum(vals); return all(len(S[h]) >= 0.8 * n_planned[h] for h in range(3)), [len(x) for x in S]


def verdict_p1(iv):
    if iv["degenerate"]:
        return "no verdict (degenerate)"
    if iv["lo"] >= 0.15:
        return "substantial false negatives (provisional)"
    if iv["hi"] <= 0.05:
        return "low observed false positives (provisional)"
    return "intermediate / indeterminate (provisional)"


def verdict_p2(iv):
    if iv["degenerate"]:
        return "no verdict (degenerate)"
    if iv["hi"] < 0:
        return "signed fits held-out better under cooperative truth: non-diagnostic (provisional)"
    if iv["lo"] > 0:
        return "cooperative fits held-out better (provisional)"
    return "indeterminate (provisional)"


def worst_case(vals, lo_fill, hi_fill, n_planned=(10, 10, 10)):
    S = by_stratum(vals)
    lo = [x + [lo_fill] * (n_planned[h] - len(x)) for h, x in enumerate(S)]; hi = [x + [hi_fill] * (n_planned[h] - len(x)) for h, x in enumerate(S)]
    return dict(fill_low=strat_interval(lo), fill_high=strat_interval(hi))


def primary(R):
    out = {}
    for br in "RF":
        T = truth_table(R, "E1", "QM1", br)
        v1, v2 = p1_values(T), p2_values(T)
        for name, vals, vf, fills in (("P1", v1, verdict_p1, (0.0, 1.0)), ("P2", v2, verdict_p2, None)):
            ok, n = gate(vals); iv = strat_interval(by_stratum(vals))
            rec = dict(interval=iv, gate_ok=ok, n_by_stratum=n, verdict=vf(iv) if ok else "not analysable")
            if fills is None:
                fills = P2_BOUNDS          # theoretical: each MSE term in [0, 4] -> MSE in [0, 8] -> difference in [-8, 8]
            if fills and not all(k == 10 for k in n):
                rec["worst_case"] = worst_case(vals, *fills)
            out[f"{name}-{br}"] = rec
    return out


def secondary(R):
    out = {}
    for fam in ("SG", "QM1", "QM3"):
        for br in "RF":
            T = truth_table(R, "E1", fam, br); recs = [d["signed"] for s in T.values() for d in s.values() if "signed" in d]
            if not recs:
                continue
            M = lambda k: float(np.nanmean([r["metrics"][k] for r in recs]))
            q = np.nanmean([r["metrics"]["frac_neg_by_true_quartile"] for r in recs], 0).tolist()
            it = [r["iterations"] for r in recs]; neg = [r["metrics"]["frac_neg"] for r in recs]
            stab = []
            for tag, s in T.items():
                negs = [load_C(d["signed"]) < 0 for d in s.values() if "signed" in d]
                if len(negs) >= 2:
                    stab.append(float(np.mean(np.all(negs, 0)[np.any(negs, 0)])) if np.any(np.any(negs, 0)) else np.nan)
            out[f"E1_{fam}_{br}"] = dict(n_fits=len(recs), frac_neg=M("frac_neg"), thr01=M("frac_neg_thr0.01"), thr05=M("frac_neg_thr0.05"), thr10=M("frac_neg_thr0.1"),
                                         neg_true_zero=M("frac_neg_true_zero"), neg_true_pos=M("frac_neg_true_pos"), by_true_quartile=q,
                                         spearman_truth=M("spearman"), sign_stability_all_seeds=float(np.nanmean(stab)) if stab else np.nan,
                                         neg_vs_stop_iter_spearman=float(stats.spearmanr(it, neg)[0]) if len(set(it)) > 1 else np.nan,
                                         fc_corr_signed=float(np.mean([r["fc_corr"] for r in recs])))
    out.update(e0_rule(R))
    for exp in ("E1w", "E1cpp"):
        sub = [r for r in R if r["exp"] == exp and r["model"] == "signed"]
        if sub:
            out[exp] = {br: dict(n=len([r for r in sub if r["branch"] == br]),
                                 frac_neg=float(np.mean([r["metrics"]["frac_neg"] for r in sub if r["branch"] == br] or [np.nan])),
                                 thr05=float(np.mean([r["metrics"]["frac_neg_thr0.05"] for r in sub if r["branch"] == br] or [np.nan])),
                                 spearman_truth=float(np.mean([r["metrics"]["spearman"] for r in sub if r["branch"] == br] or [np.nan])))
                        for br in "RF"}
    return out


def e0_rule(R):
    """E0 / E0m, descriptive. Unit: truth (T-SG, 3), mean over estimator seeds of the signed
 fit. Reported per branch: raw negative fraction, negative fraction beyond the absolute 2 alpha = 2e-4, tie-aware Spearman
 truth vs fit on SC-support edges, target FC negative fraction. Classification: "Fig-7-type outcome observed" iff the
 truth-mean frac_neg_abs2alpha <= 0.01 AND truth-mean Spearman >= 0.6, else "not observed". Labels: E0 = public L1
 configuration on HEADER-generated data (sigma 0.01 vs 0.001, declared mismatch); E0m = matched sigma 0.001. Neither is
 a replication of Ext Data Fig 7 nor an effect attributable to L1 alone."""
    out = {}
    for exp, label in (("E0", "public L1 config on HEADER-generated data (10x sigma mismatch)"), ("E0m", "public L1 config, matched sigma 0.001")):
        for br in "RF":
            per = {}
            for r in R:
                if r["exp"] == exp and r["branch"] == br and r["model"] == "signed":
                    per.setdefault(r["tag"], []).append(r)
            if not per:
                continue
            complete = len(per) == 3 and all(len({x["seed"] for x in v}) == 3 for v in per.values())
            if not complete:                                   #: no classification on incomplete data
                out[f"{exp}_{br}"] = dict(label=label, classification="not classifiable (incomplete: needs 3 truths x 3 signed seeds)",
                                          n_truths=len(per), seeds_per_truth={t: len({x["seed"] for x in v}) for t, v in per.items()})
                continue
            agg = lambda f: float(np.mean([np.mean([f(x) for x in v]) for v in per.values()]))
            m = dict(n_truths=len(per), frac_neg=agg(lambda x: x["metrics"]["frac_neg"]), frac_neg_abs2alpha=agg(lambda x: x["metrics"]["frac_neg_abs2alpha"]),
                     spearman=agg(lambda x: x["metrics"]["spearman"]), target_fc_neg_frac=agg(lambda x: x["target_fc_neg_frac"]))
            m["classification"] = ("Fig-7-type outcome observed" if m["frac_neg_abs2alpha"] <= E0_RULE["max_neg_abs2alpha"] and m["spearman"] >= E0_RULE["min_spearman"]
                                   else "Fig-7-type outcome not observed")
            out[f"{exp}_{br}"] = dict(label=label, **m)
    return out


if __name__ == "__main__":
    R, rej, fsha = load(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else "primary")
    res = dict(freeze=fsha, group=sys.argv[3] if len(sys.argv) > 3 else "primary", n_admitted=len(R), rejected=rej, primary=primary(R), secondary=secondary(R))
    json.dump(res, open(sys.argv[2], "w"), indent=1, default=float)
    for k, v in res["primary"].items():
        iv = v["interval"]; print(k, v["verdict"], "theta %.4f [%.4f, %.4f]" % (iv["theta"], iv["lo"], iv["hi"]), "n", v["n_by_stratum"])
