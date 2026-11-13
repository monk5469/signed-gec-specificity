"""Preprocessing comparison (PPC) analysis (Sections 2.3, 3.2): admission of the PPC fits, the confirmatory tests T1-T3 with Holm
correction, and the descriptive endpoints.
Usage: python luppi_ppc_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np
from scipy import stats

P33 = DATA_ROOT + "results/p33/"
PREREG = P33 + "ppc_analysis_prereg.json"; EXPFREEZE = P33 + "ppc_experiment_freeze.json"
FIX_OUT = P33 + "r4e_e5fix/"; NETS = P33 + "r4e_e8/networks.json"
NEW = ("vol0", "mot", "acc", "acc1", "acc3", "accero", "gsr", "null10"); CONDS = NEW + ("fix",)
DEMO = {"100307", "100408", "101107"}; MIN_N = 86; DOSE = (("mot", 0), ("acc1", 1), ("acc3", 3), ("acc", 5))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def extras(C, SC, inp, nets):
    """E_bn_fc (SC + input-FC quartile covariates), RD and the separation reason of the between x negative table."""
    import luppi_hopf as L, luppi_r4e_lib as Lib, luppi_r4e_e8_analyze as A8
    m = Lib.support(SC); y = C[m] < 0; nn = np.asarray(nets); b = (nn[:, None] != nn[None, :])[m]; F = np.corrcoef(L.prep_emp(inp))
    q = np.searchsorted(np.quantile(F[m], [0.25, 0.5, 0.75]), F[m]); Xf = np.column_stack([A8.design(SC, m, b)] + [(q == k).astype(float) for k in (1, 2, 3)])
    nb, nw = int((y & b).sum()), int((y & ~b).sum())
    sep = "none_negative" if y.sum() == 0 else "pos_sep" if nw == 0 else "neg_sep" if nb == 0 else None
    return dict(E_bn_fc=A8.logor(y, Xf), RD=float(y[b].mean() - y[~b].mean()), sep=sep)


def load(cond):
    import luppi_r4e_ppc as W, luppi_r4e_run as RUN, luppi_hopf as L, luppi_r4e_lib as Lib, luppi_r4e_e8_analyze as A8
    X = W.X
    if cond == "fix":
        X.OUT = FIX_OUT; X.HCP = DATA_ROOT + "work/p33_ts_fix/"
    else:
        W.point(cond)
    fsha, fr = X.verify_freeze("e5"); expected = {j["key"]: j for j in X.job_list("e5")}
    rej = dict(malformed=0, unexpected=0, field_mismatch=0, not_done=0, wrong_prov=0, duplicate=0, bad_matrix=0); seen = set(); R = {}
    for p in sorted(glob.glob(X.OUT + "jobs_e5/*.json")):
        try:
            r = json.load(open(p)); assert isinstance(r, dict)
        except Exception:
            rej["malformed"] += 1; continue
        j = expected.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if any(r.get(k) != v for k, v in j.items()):
            rej["field_mismatch"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != X.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if j["key"] in seen:
            rej["duplicate"] += 1; continue
        seen.add(j["key"]); R.setdefault(j["tag"], {})[j["model"]] = (j, r)
    SC = np.load(X.OUT + "e5/SC.npy"); nets = json.load(open(NETS))["networks"]; m = Lib.support(SC); rows = {}
    for tag, d in R.items():
        if "signed" not in d:
            continue
        j, r = d["signed"]; Cp = j["file"].replace(".json", "_C.npy")
        try:
            C = np.load(Cp)
        except Exception:
            C = None
        if C is None or C.shape != (100, 100) or not np.all(np.isfinite(C)) or RUN.sha_arr(C) != r.get("C_sha"):
            rej["bad_matrix"] += 1; continue
        inp = np.load(X.OUT + f"e5/{tag}_REST1_input.npy"); F = np.corrcoef(L.prep_emp(inp)); iu = np.triu_indices(100, 1)
        try:                                                             # optional endpoints: failure marks only them
            dsc = A8.descriptors(C, SC, inp, nets); ex = extras(C, SC, inp, nets); why = None
        except Exception as e:
            dsc = dict(E_bn=np.nan); ex = dict(E_bn_fc=np.nan, RD=np.nan, sep=None); why = repr(e)[:120]
        p2 = np.nan
        if "coop" in d and np.isfinite(r.get("mse_ho", np.nan)) and np.isfinite(d["coop"][1].get("mse_ho", np.nan)):
            p2 = r["mse_ho"] - d["coop"][1]["mse_ho"]
        rows[tag] = dict(fp=float((C[m] < 0).mean()), thr5=float((C[m] < -0.05 * np.abs(C[m]).max()).mean()), p2=float(p2),
                         fc_corr=float(r["fc_corr"]) if np.isfinite(r.get("fc_corr", np.nan)) else np.nan, E_bn=dsc["E_bn"], E_bn_fc=ex["E_bn_fc"], RD=ex["RD"], sep=ex["sep"], descriptor_error=why,
                         gs=float(np.var(inp.mean(0)) / np.mean(inp.var(1))), mean_fc=float(F[iu].mean()), neg_fc=float((F[iu] < 0).mean()))
    return rows, rej, fsha


def paired(a, b, key, alt="greater"):
    tags = sorted((set(a) & set(b)) - DEMO); d = np.array([a[t][key] - b[t][key] for t in tags], float); d = d[np.isfinite(d)]
    if len(d) < 6:
        return dict(n=int(len(d)), available=False, p=1.0)
    w = stats.wilcoxon(d, alternative=alt) if np.any(d != 0) else None
    rng = np.random.default_rng(20261009); bs = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
    return dict(n=int(len(d)), available=True, median_diff=float(np.median(d)), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                p=float(w.pvalue) if w is not None else 1.0, share_positive=float(np.mean(d > 0)))


def summ(rows, key, tags):
    v = np.array([rows[t][key] for t in tags if t in rows], float); v = v[np.isfinite(v)]
    return dict(n=int(v.size), median=float(np.median(v)) if v.size else np.nan, q25=float(np.quantile(v, .25)) if v.size else np.nan, q75=float(np.quantile(v, .75)) if v.size else np.nan)


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("analysis script differs from the pre-registered version")
    ef = json.load(open(EXPFREEZE))
    bad = [p for p, h in ef["pinned"].items() if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"experiment freeze: {len(bad)} pinned items changed, e.g. {bad[:3]}")
    G = {c: load(c) for c in CONDS}; rows = {c: G[c][0] for c in CONDS}
    for c in NEW:
        if G[c][2] != ef["group_freezes"][c]:
            raise SystemExit(f"{c}: group freeze differs from the experiment freeze")
    T = {"T1": paired(rows["acc"], rows["mot"], "fp"), "T2": paired(rows["acc"], rows["null10"], "fp")}
    for k in ("T1", "T2"):
        T[k]["available"] = T[k]["available"] and T[k]["n"] >= MIN_N
        if not T[k]["available"]:
            T[k]["p"] = 1.0
    tags = sorted(set.intersection(*[set(rows[c]) for c, _ in DOSE]) - DEMO); rho = []
    for t in tags:
        fp = [rows[c][t]["fp"] for c, _ in DOSE]
        rho.append(stats.spearmanr([k for _, k in DOSE], fp).statistic if np.ptp(fp) > 0 else 0.0)
    rho = np.array(rho, float)
    if len(rho) >= MIN_N and np.any(rho != 0):
        T["T3"] = dict(n=int(len(rho)), available=True, median_rho=float(np.median(rho)), share_positive=float(np.mean(rho > 0)), p=float(stats.wilcoxon(rho, alternative="greater").pvalue))
    else:
        T["T3"] = dict(n=int(len(rho)), available=False, p=1.0)
    order = sorted(T, key=lambda k: T[k]["p"]); holm, stop = {}, False
    for i, k in enumerate(order):
        holm[k] = bool(not stop and T[k]["available"] and T[k]["p"] <= 0.05 / (3 - i)); stop = stop or not holm[k]
    for k in T:
        T[k]["holm_significant"] = holm[k]
    for k, fl in (("T1", 0.10), ("T2", 0.10)):
        if T[k]["available"]:
            T[k]["floor"] = fl; T[k]["ci_lower_clears_floor"] = bool(T[k]["ci"][0] >= fl)
    R1 = "not analysable" if not (T["T1"]["available"] and T["T2"]["available"]) else \
         ("confirmed" if holm["T1"] and holm["T2"] and T["T1"]["median_diff"] >= 0.10 and T["T2"]["median_diff"] >= 0.10 else "not confirmed")
    R2 = "not analysable" if not T["T3"]["available"] else ("confirmed" if holm["T3"] else "not confirmed")
    elig = {c: sorted(set(rows[c]) - DEMO) for c in CONDS}
    sec = dict(gsr_vs_acc=paired(rows["gsr"], rows["acc"], "fp", "two-sided"), accero_vs_acc=paired(rows["accero"], rows["acc"], "fp", "two-sided"),
               vol0_vs_mot=paired(rows["vol0"], rows["mot"], "fp", "two-sided"), acc_vs_fix_surface=paired(rows["acc"], rows["fix"], "fp", "two-sided"))
    com = sorted((set(rows["gsr"]) & set(rows["acc"])) - DEMO)
    sec["spearman_fp_gsr_vs_acc"] = float(stats.spearmanr([rows["gsr"][t]["fp"] for t in com], [rows["acc"][t]["fp"] for t in com]).statistic) if len(com) > 2 else np.nan
    sec["observational_spearman_gs_fp_acc"] = float(stats.spearmanr([rows["acc"][t]["gs"] for t in elig["acc"]], [rows["acc"][t]["fp"] for t in elig["acc"]]).statistic) if len(elig["acc"]) > 2 else np.nan
    pg = [rows[c][t]["gs"] for c in NEW for t in elig[c]]; pf = [rows[c][t]["fp"] for c in NEW for t in elig[c]]
    sec["observational_spearman_gs_fp_pooled"] = float(stats.spearmanr(pg, pf).statistic) if len(pg) > 2 else np.nan
    sec["per_condition"] = {c: dict(n=len(elig[c]), **{k: summ(rows[c], k, elig[c]) for k in ("fp", "thr5", "p2", "E_bn", "E_bn_fc", "RD", "gs", "mean_fc", "neg_fc", "fc_corr")},
                                    p2_negative_share=float(np.mean([rows[c][t]["p2"] < 0 for t in elig[c] if np.isfinite(rows[c][t]["p2"])])) if elig[c] else np.nan,
                                    fp_ge_020_share=float(np.mean([rows[c][t]["fp"] >= 0.20 for t in elig[c]])) if elig[c] else np.nan,
                                    sep_counts={x: sum(rows[c][t]["sep"] == x for t in elig[c]) for x in ("pos_sep", "neg_sep", "none_negative")}) for c in CONDS}
    sec["demo_subjects"] = {t: {c: {k: rows[c][t][k] for k in ("fp", "gs", "neg_fc")} for c in CONDS if t in rows[c]} for t in sorted(DEMO)}
    res = dict(prereg=pre, experiment_freeze=sha(EXPFREEZE), groups={c: dict(freeze=G[c][2], admitted=len(rows[c]), eligible=len(elig[c]), rejected=G[c][1]) for c in CONDS},
               primary=T, holm=holm, reading=dict(R1=R1, R2=R2), secondary=sec, rows=rows)
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float)
    for c in CONDS:
        pc = sec["per_condition"][c]
        print(c, "n", pc["n"], "| FP %.3f [%.3f, %.3f]" % (pc["fp"]["median"], pc["fp"]["q25"], pc["fp"]["q75"]), "| gs %.3f | negFC %.3f" % (pc["gs"]["median"], pc["neg_fc"]["median"]),
              "| P2<0 %.2f | FP>=0.2 %.2f" % (pc["p2_negative_share"], pc["fp_ge_020_share"]), flush=True)
    print("PRIMARY", T, flush=True); print("READING", res["reading"], flush=True)
    print("SECONDARY gsr-acc", sec["gsr_vs_acc"], "| accero-acc", sec["accero_vs_acc"], "| rho(FP_gsr, FP_acc) %.3f" % sec["spearman_fp_gsr_vs_acc"], flush=True)


if __name__ == "__main__":
    main()
