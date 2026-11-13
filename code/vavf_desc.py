"""Descriptive checks without new fits (Supplementary Notes 5 and 6): filter cross-terms (vc), operator-only baseline of the
readout law (ve) and random-placement null of edge consistency (vf).
Usage: python vavf_desc.py vc|ve|vf OUT_JSON [--fixture]"""
import sys, json, glob, hashlib, argparse
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; WORK = DATA_ROOT + "work/"; DEMO = {"100307", "100408", "101107"}
FIG = P33 + "vavf/figdata/"; DROP = 11


def subjects():
    return [x["subject"] for x in json.load(open(P33 + "p33_hcp_s200_v13.json"))["subjects"] if x["subject"] not in DEMO]


def e5(R):
    import luppi_r4e_lib as Lib
    return Lib.bandpass(R.T.astype(float)[:, DROP:])


def cov(A, B=None):
    A = A - A.mean(1, keepdims=True); B = A if B is None else B - B.mean(1, keepdims=True); return A @ B.T / (A.shape[1] - 1)


def stage_vc(fixture):
    subs = subjects()[:3] if fixture else subjects(); out = {}
    for a in ("acc", "gsr", "acc1"):
        rows = []
        for s in subs:
            Ym = np.load(WORK + f"p33_ts_mot/{s}_REST1_LR_s100.npy").T.astype(float); Ya = np.load(WORK + f"p33_ts_{a}/{s}_REST1_LR_s100.npy").T.astype(float)
            D = Ym - Ya; orth = np.linalg.norm(Ya @ D.T) / (np.linalg.norm(Ya) * np.linalg.norm(D))
            raw_id = np.linalg.norm(cov(Ym) - cov(Ya) - cov(D)) / np.linalg.norm(cov(D))
            Hm, Ha = e5(Ym.T), e5(Ya.T); HD = Hm - Ha; CHD = cov(HD); X = cov(Ha, HD); dS = cov(Hm) - cov(Ha)
            off = ~np.eye(len(dS), dtype=bool); ev = np.linalg.eigvalsh((dS + dS.T) / 2)
            sd = np.sqrt(np.diag(cov(Hm))); nrm = np.outer(sd, sd)
            rows.append(dict(s=s, orth_raw=float(orth), raw_identity_rel=float(raw_id), cross_rel=float(np.linalg.norm(X + X.T) / np.linalg.norm(CHD)),
                             sign_disagree=float((np.sign(dS[off]) != np.sign(CHD[off])).mean()), eig_ratio=float(ev[0] / ev[-1]),
                             cross_rel_corr=float(np.linalg.norm((X + X.T) / nrm) / np.linalg.norm(CHD / nrm))))
        med = {q: float(np.median([r[q] for r in rows])) for q in rows[0] if q != "s"}
        q90 = {q: float(np.quantile([r[q] for r in rows], 0.9)) for q in rows[0] if q != "s"}
        out[f"mot->{a}"] = dict(n=len(rows), median=med, q90=q90, rows=rows)
        print(f"V-C mot->{a}", {k: round(v, 4) for k, v in med.items()}, flush=True)
    return out


def stage_ve(fixture):
    import csv, e12a_g_freeze as GF
    files = sorted(glob.glob(FIG + "fig3_readout_*.csv")); R = [r for f in files for r in csv.DictReader(open(f))]
    if fixture:
        R = [r for st in ("train", "V1", "V2", "accS") for r in [x for x in R if x["set"] == st][:30]]
    st_, cond = np.array([r["set"] for r in R]), np.array([r["cond"] for r in R])
    nf, gs, fp, gp = (np.array([float(r[q]) for r in R]) for q in ("nf", "gs", "fp", "g_pred"))
    G = json.load(open(P33 + "e12a_g.json")); rn, rg = G["train_range"]["nf"], G["train_range"]["gs"]
    pred = dict(g=GF.predict(G, "g", "fp", nf, gs), g_nf=GF.predict(G, "g_nf", "fp", nf, gs))
    opm = {c: float(np.median(fp[(st_ == "train") & (cond == c)])) for c in set(cond[st_ == "train"])}
    pred["op"] = np.array([opm.get(c, np.nan) for c in cond])
    inside = (nf >= rn[0]) & (nf <= rn[1]) & (gs >= rg[0]) & (gs <= rg[1])
    def score(sel, m):
        sel = sel & np.isfinite(pred[m])
        if sel.sum() < 3:
            return dict(n=int(sel.sum()))
        e = fp[sel] - pred[m][sel]; return dict(n=int(sel.sum()), mae=float(np.abs(e).mean()), r2=float(1 - (e ** 2).sum() / ((fp[sel] - fp[sel].mean()) ** 2).sum()))
    out = dict(files={f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}, gpred_max_absdiff_vs_csv=float(np.abs(pred["g"] - gp).max()))
    for rows in ("inside", "all"):
        for st in ("train", "V1", "V2", "accS"):
            base = (st_ == st) & (inside if rows == "inside" else True)
            out[f"{rows}|{st}"] = dict(overall={m: score(base, m) for m in ("op", "g_nf", "g")},
                                      per_operator={c: {m: score(base & (cond == c), m) for m in ("op", "g_nf", "g")} for c in sorted(set(cond[base]))})
    for st in ("V1", "V2", "accS"):
        print("V-E", st, {m: out[f"inside|{st}"]["overall"][m] for m in ("op", "g_nf", "g")}, flush=True)
    return out


def stage_vf(fixture):
    import e1i1_explore as E
    rng = np.random.default_rng(20261026); nd = 20 if fixture else 1000; out = {}
    fits = {c: E.load_cond(c) for c in E.CONDS}; tags = sorted(set.intersection(*[set(v[0]) for v in fits.values()]))
    if fixture:
        tags = tags[:10]
    SC = np.load(fits["acc"][1] + "e5/SC.npy"); m = (SC != 0) & ~np.eye(100, dtype=bool); M = int(m.sum())
    for c in E.CONDS:
        neg = np.stack([fits[c][0][t][m] < 0 for t in tags]); obs = int((neg.mean(0) >= 0.5).sum()); k = neg.sum(1)
        null = []
        for _ in range(nd):
            Z = np.zeros((len(tags), M), bool)
            for i, ki in enumerate(k):
                Z[i, rng.choice(M, ki, replace=False)] = True
            null.append(int((Z.mean(0) >= 0.5).sum()))
        out[c] = dict(n=len(tags), median_negatives=float(np.median(k)), observed=obs, null_mean=float(np.mean(null)), null_q95=float(np.quantile(null, 0.95)),
                      null_max=int(max(null)))
        print("V-F", c, out[c], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("out"); ap.add_argument("--fixture", action="store_true"); a = ap.parse_args()
    if not a.fixture:
        import vavf_rules
        vavf_rules.check("desc")
    res = dict(stage=a.stage, fixture=a.fixture, result={"vc": stage_vc, "ve": stage_ve, "vf": stage_vf}[a.stage](a.fixture))
    json.dump(res, open(a.out, "w" if a.fixture else "x"), indent=1, default=float)
