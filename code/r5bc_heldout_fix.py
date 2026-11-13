"""Group-level held-out comparison and ICA-FIX transport of the readout law (descriptive; Sections 2.3.3, 3.2, 3.3)."""
import sys, os, json, time, hashlib
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; R5 = P33 + "r5/"; PREREG = R5 + "r5bc_prereg.json"; GJ = P33 + "e12a_g.json"; PPC = P33 + "ppc_analysis.json"
CONDS = ("vol0", "mot", "null10", "acc1", "acc3", "acc", "accero", "gsr", "fix"); FIX_OUT = P33 + "r4e_e5fix/"
DEMO = {"100307", "100408", "101107"}; MARGIN = 0.023; B = 2000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def hashes():
    return dict(script=sha(os.path.abspath(__file__)), r5_run=sha(R5 + "r5_run.json"), g=sha(GJ), ppc=sha(PPC))


def outdir(cond):
    return FIX_OUT if cond == "fix" else P33 + f"r4e_ppc_{cond}/"


def gc(A, Bm, idx):
    from scipy.stats import spearmanr
    iu = np.triu_indices(A.shape[1], 1); a = A[idx].mean(0)[iu]; b = Bm[idx].mean(0)[iu]
    return float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b)[0])


def r5b(run):
    import luppi_hopf as L
    out = {}
    for cond in CONDS:
        if run["conds"][cond]["npz_sha"] != sha(R5 + f"{cond}.npz"):
            sys.exit(f"{cond}.npz changed after the R5 run")
        z = np.load(R5 + f"{cond}.npz"); tags = [str(t) for t in z["tags"]]; n = len(tags); all_ = np.arange(n)
        FH = np.stack([np.corrcoef(L.prep_emp(np.load(outdir(cond) + f"e5/{t}_REST1_heldout.npy"))) for t in tags])
        S, Co, FE = z["FCs_signed"], z["FCs_coop"], z["FCe"]
        ps, ss = gc(FH, S, all_); pc, sc = gc(FH, Co, all_); ce = gc(FH, FE, all_)
        rng = np.random.default_rng(20261019); bs = []
        for _ in range(B):
            i = rng.integers(0, n, n); a = gc(FH, S, i)[0]; b = gc(FH, Co, i)[0]; bs.append((a, b, a - b))
        bs = np.array(bs); ci = lambda k: [float(np.quantile(bs[:, k], 0.025)), float(np.quantile(bs[:, k], 0.975))]
        out[cond] = dict(n=n, heldout_pearson=dict(signed=ps, coop=pc, diff=ps - pc, ci_signed=ci(0), ci_coop=ci(1), ci_diff=ci(2)),
                         heldout_spearman=dict(signed=ss, coop=sc, diff=ss - sc), ceiling_input_vs_heldout=dict(pearson=ce[0], spearman=ce[1]),
                         label="does not generalise" if ci(2)[0] <= 0 else "generalises")
        h = out[cond]["heldout_pearson"]
        print(f"R5b {cond:7s} held-out r signed {ps:.3f} coop {pc:.3f} diff {ps - pc:+.3f} {np.round(ci(2), 3).tolist()} | ceiling {ce[0]:.3f} | {out[cond]['label']}", flush=True)
    return out


def r5c():
    import e12a_g_freeze as GF
    Gd = json.load(open(GJ)); a = json.load(open(PPC))["rows"]["fix"]; tr = Gd["train_range"]
    rows = [dict(unit=s, **{k: v[k] for k in ("fp", "p2", "neg_fc", "gs")}) for s, v in sorted(a.items())
            if s not in DEMO and all(np.isfinite(v.get(k, np.nan)) for k in ("fp", "p2", "neg_fc", "gs"))]
    sup = [r for r in rows if tr["nf"][0] <= r["neg_fc"] <= tr["nf"][1] and tr["gs"][0] <= r["gs"] <= tr["gs"][1]]
    res = dict(n_rows=len(rows), n_supported=len(sup), caveat="fix differs in space (surface, MSMAll) and parcellation as well as operator")
    if len(sup) < 20:
        res["label"] = "not analysable"; return res
    nf = [r["neg_fc"] for r in sup]; gs = [r["gs"] for r in sup]; rng = np.random.default_rng(20261020); boots = [rng.integers(0, len(sup), len(sup)) for _ in range(B)]
    for model in GF.MODELS:
        res[model] = {}
        for key in ("fp", "p2", "neg"):
            y = np.array([r[key] if key != "neg" else float(r["p2"] < 0) for r in sup]); e = y - GF.predict(Gd, model, key, nf, gs)
            Bm = np.array([(np.mean(np.abs(e[ii])), np.median(e[ii])) for ii in boots])
            res[model][key] = dict(mae=float(np.mean(np.abs(e))), mae_ci95=[float(np.quantile(Bm[:, 0], 0.025)), float(np.quantile(Bm[:, 0], 0.975))],
                                   median_residual=float(np.median(e)), median_ci95=[float(np.quantile(Bm[:, 1], 0.025)), float(np.quantile(Bm[:, 1], 0.975))],
                                   r2=float(1 - np.mean(e ** 2) / np.var(y)) if np.var(y) > 0 else np.nan)
    f = res["g"]["fp"]
    res["label"] = "consistent" if (f["mae_ci95"][1] <= MARGIN and -MARGIN <= f["median_ci95"][0] and f["median_ci95"][1] <= MARGIN) else "not consistent"
    res["input_medians"] = dict(nf=float(np.median(nf)), gs=float(np.median(gs)), fp=float(np.median([r["fp"] for r in sup])))
    print("R5c fix | supported", len(sup), "of", len(rows), "| g FP MAE %.4f %s | median residual %.4f %s | R2 %.3f | %s" % (
        f["mae"], np.round(f["mae_ci95"], 4).tolist(), f["median_residual"], np.round(f["median_ci95"], 4).tolist(), f["r2"], res["label"]), flush=True)
    return res


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "prereg":
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), hashes=hashes(), spec="held-out group-level FC and ICA-FIX transport"), open(PREREG, "x"), indent=1)
        print("prereg", hashes()["script"][:12])
    elif st == "run":
        pre = json.load(open(PREREG))
        if pre["hashes"] != hashes():
            sys.exit("script or inputs differ from the registration")
        res = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), prereg=pre, r5b=r5b(json.load(open(R5 + "r5_run.json"))), r5c=r5c())
        json.dump(res, open(R5 + "r5bc_result.json", "x"), indent=1, default=float)
