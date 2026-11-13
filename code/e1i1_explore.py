"""Exploratory edge-level analysis (Sections 2.3.4, 3.2; Supplementary Note 5): edge signs across operators, cross-participant
consistency, and prediction of acc-negative edges (edge AUC) from pre-removal FC and the removed component.
Usage: python e1i1_explore.py OUT_JSON"""
from paths import DATA_ROOT
import sys, json, glob
import numpy as np

P33 = DATA_ROOT + "results/p33/"; DEMO = {"100307", "100408", "101107"}
CONDS = ("vol0", "mot", "null10", "acc1", "acc3", "acc", "gsr")


def load_cond(cond):
    import luppi_r4e_ppc as W, luppi_r4e_run as RUN
    X = W.X; W.point(cond); fsha, fr = X.verify_freeze("e5"); exp = {j["key"]: j for j in X.job_list("e5")}; out = {}
    for p in sorted(glob.glob(X.OUT + "jobs_e5/*.json")):
        r = json.load(open(p)); j = exp.get(r.get("key"))
        if j is None or j["model"] != "signed" or j["tag"] in DEMO or r.get("state") != "done" or r.get("prov") != X.job_prov(j, fsha, fr):
            continue
        C = np.load(j["file"].replace(".json", "_C.npy"))
        if C.shape == (100, 100) and np.all(np.isfinite(C)) and RUN.sha_arr(C) == r.get("C_sha"):
            out[j["tag"]] = C
    return out, X.OUT


def auc(score, y):
    from scipy.stats import rankdata
    a, b = score[y], score[~y]
    if a.size == 0 or b.size == 0:
        return np.nan
    r = rankdata(np.r_[a, b]); return float((r[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size))


def logit_auc(X, y):
    """In-sample AUC of a 2-predictor logistic fit (descriptive upper bound of the combination)."""
    from numpy.linalg import solve
    Xd = np.c_[np.ones(len(y)), (X - X.mean(0)) / X.std(0)]; b = np.zeros(Xd.shape[1]); yf = y.astype(float)
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(Xd @ b, -30, 30))); H = Xd.T @ (Xd * (p * (1 - p))[:, None]) + 1e-6 * np.eye(Xd.shape[1])
        st = solve(H, Xd.T @ (yf - p)); b += st
        if np.max(np.abs(st)) < 1e-8:
            break
    return auc(Xd @ b, y)


def main():
    import luppi_hopf as L
    fits, outs = {}, {}
    for c in CONDS:
        fits[c], outs[c] = load_cond(c)
    tags = sorted(set.intersection(*[set(v) for v in fits.values()]))
    SC = np.load(outs["acc"] + "e5/SC.npy"); m = (SC != 0) & ~np.eye(100, dtype=bool); iu = np.triu_indices(100, 1)
    h = np.array(json.load(open(P33 + "e13_hier.json"))["h_A2"]); scs = SC.sum(1)
    res = dict(n_subjects=len(tags), n_support=int(m.sum()))
    # ---------------- (1) operator dose-response per edge
    neg = {c: np.stack([fits[c][t][m] < 0 for t in tags]) for c in CONDS}                    # (subjects, edges)
    W = {c: np.stack([fits[c][t][m] for t in tags]) for c in CONDS}
    e1 = {c: float(np.median(neg[c].mean(1))) for c in CONDS}
    pers = [(neg["mot"][k] & neg["acc"][k]).sum() / max(neg["mot"][k].sum(), 1) for k in range(len(tags))]
    pre = [(neg["mot"][k] & neg["acc"][k]).sum() / max(neg["acc"][k].sum(), 1) for k in range(len(tags))]
    allc = np.logical_and.reduce([neg[c] for c in CONDS])
    grp = {c: int((neg[c].mean(0) >= 0.5).sum()) for c in CONDS}
    gs_med = {}
    for c in CONDS:
        g = []
        for t in tags:
            x = np.load(outs[c] + f"e5/{t}_REST1_input.npy"); g.append(np.var(x.mean(0)) / np.mean(x.var(1)))
        gs_med[c] = float(np.median(g))
    from scipy.stats import spearmanr
    order = sorted(CONDS, key=lambda c: -gs_med[c]); med = np.stack([np.median(W[c], 0) for c in order]); dose = -np.array([gs_med[c] for c in order])
    rho = np.array([spearmanr(dose, med[:, e])[0] for e in range(med.shape[1])])
    acc_grp_neg = np.median(W["acc"], 0) < 0
    res["E1"] = dict(FP_median=e1, persistence_mot_into_acc=float(np.median(pers)), share_acc_neg_already_neg_in_mot=float(np.median(pre)),
                     dose_invariant_edges_per_subject=dict(median=float(np.median(allc.sum(1))), mean_share=float(allc.mean())),
                     group_consistent_neg_edges=grp, gs_median=gs_med,
                     group_acc_negative_edges=int(acc_grp_neg.sum()),
                     rho_weight_vs_dose_on_group_acc_negatives=dict(median=float(np.nanmedian(rho[acc_grp_neg])), share_below_minus_0p8=float(np.mean(rho[acc_grp_neg] < -0.8))))
    # ---------------- (2) prediction of acc-negative edges
    for tgt in ("acc", "gsr"):
        A = {k: [] for k in ("a_negFCmot", "b_removal", "d_rank1", "c_negFCpost", "ab_logit")}; G = []
        for t in tags:
            Ym = np.load(outs["mot"] + f"e5/{t}_REST1_input.npy"); Yt = np.load(outs[tgt] + f"e5/{t}_REST1_input.npy")
            Em, Et = L.prep_emp(Ym), L.prep_emp(Yt); Z = Em - Et
            Cm, Ct = np.cov(Em), np.cov(Et); d = np.sqrt(np.diag(Cm)); Fm = Cm / np.outer(d, d); dt = np.sqrt(np.diag(Ct)); Ft = Ct / np.outer(dt, dt)
            Delta = (Cm - Ct) / np.outer(d, d)
            U, S, Vt = np.linalg.svd(Z - Z.mean(1, keepdims=True), full_matrices=False); z1 = Vt[0]
            g = np.array([np.corrcoef(Em[i], z1)[0, 1] for i in range(100)]); g = g * np.sign(g.mean()); G.append(g)
            y = fits[tgt][t][m] < 0
            A["a_negFCmot"].append(auc(-Fm[m], y)); A["b_removal"].append(auc(Delta[m], y)); A["d_rank1"].append(auc(np.outer(g, g)[m], y))
            A["c_negFCpost"].append(auc(-Ft[m], y)); A["ab_logit"].append(logit_auc(np.c_[-Fm[m], Delta[m]], y))
        Gm = np.mean(G, 0)
        res[f"I1_{tgt}"] = dict(auc_median={k: float(np.nanmedian(v)) for k, v in A.items()},
                               auc_q25_q75={k: [float(np.nanquantile(v, 0.25)), float(np.nanquantile(v, 0.75))] for k, v in A.items()},
                               topography_corr_myelin=float(np.corrcoef(Gm, h)[0, 1]), topography_corr_SCstrength=float(np.corrcoef(Gm, scs)[0, 1]),
                               topography_mean=Gm.tolist())
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("I1_acc", "I1_gsr")}, indent=1))
    for tgt in ("acc", "gsr"):
        r = res[f"I1_{tgt}"]; print(tgt, "AUC", {k: round(v, 3) for k, v in r["auc_median"].items()}, "| topography~myelin %.3f ~SC %.3f" % (r["topography_corr_myelin"], r["topography_corr_SCstrength"]))


if __name__ == "__main__":
    main()
