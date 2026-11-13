"""Readout law (Section 2.5): fits and freezes g (separate cubic terms in nf and gs on standardised predictors) and the nf-only
comparator g_nf on the 760 PPC participant-condition rows, for the outcomes FP, P2 and 1[P2 < 0]. The frozen output is
provenance/frozen_laws/e12a_g.json.
Usage: python e12a_g_freeze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, json, hashlib
import numpy as np

P33 = DATA_ROOT + "results/p33/"; SRC = P33 + "ppc_analysis.json"; VOL = ("vol0", "mot", "null10", "acc1", "acc3", "acc", "accero", "gsr")
DEMO = {"100307", "100408", "101107"}; MODELS = ("g", "g_nf")


def features(model, nf, gs):
    nf, gs = np.asarray(nf, float), np.asarray(gs, float); cols = [nf, nf ** 2, nf ** 3]
    if model == "g":
        cols += [gs, gs ** 2, gs ** 3]
    return np.column_stack(cols)


def predict(G, model, key, nf, gs):
    m = G[model]; Z = (features(model, nf, gs) - np.asarray(m["mu"])) / np.asarray(m["sd"])
    return np.column_stack([np.ones(len(Z)), Z]) @ np.asarray(m["coef"][key])


if __name__ == "__main__":
    a = json.load(open(SRC)); R = []
    for c in VOL:
        for s, v in a["rows"][c].items():
            if s not in DEMO and all(np.isfinite(v.get(k, np.nan)) for k in ("fp", "neg_fc", "gs", "p2")):
                R.append(dict(nf=v["neg_fc"], gs=v["gs"], fp=v["fp"], p2=v["p2"], neg=float(v["p2"] < 0)))
    if len(R) != 95 * len(VOL):
        raise SystemExit(f"expected 760 training rows, got {len(R)}")
    nf = [r["nf"] for r in R]; gs = [r["gs"] for r in R]
    G = dict(source=SRC, source_sha=hashlib.sha256(open(SRC, "rb").read()).hexdigest(), n=len(R), conditions=VOL, excluded_subjects=sorted(DEMO),
             train_range=dict(nf=[float(min(nf)), float(max(nf))], gs=[float(min(gs)), float(max(gs))]))
    for model in MODELS:
        Z = features(model, nf, gs); mu, sd = Z.mean(0), Z.std(0) + 1e-12; X = np.column_stack([np.ones(len(R)), (Z - mu) / sd])
        G[model] = dict(mu=mu.tolist(), sd=sd.tolist(), coef={k: np.linalg.lstsq(X, np.array([r[k] for r in R]), rcond=None)[0].tolist() for k in ("fp", "p2", "neg")})
        G[model]["in_sample_mae"] = {k: float(np.mean(np.abs(np.array([r[k] for r in R]) - predict(G, model, k, nf, gs)))) for k in ("fp", "p2", "neg")}
    json.dump(G, open(sys.argv[1], "x"), indent=1)
    print("frozen on", len(R), "rows;", {m: {k: round(v, 4) for k, v in G[m]["in_sample_mae"].items()} for m in MODELS}, flush=True)
