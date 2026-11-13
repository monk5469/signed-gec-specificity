"""Simulation 3: freezes the joint (nf, gs) training support of the readout law (k-nearest-neighbour criterion on the 760 PPC
rows) before any simulated input is generated.
Usage: python e12c_support_freeze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, json, hashlib
import numpy as np

SRC = DATA_ROOT + "results/p33/ppc_analysis.json"; VOL = ("vol0", "mot", "null10", "acc1", "acc3", "acc", "accero", "gsr")
DEMO = {"100307", "100408", "101107"}


def in_support(S, nf, gs):
    """Distance of (nf, gs) to the nearest frozen training row in standardised coordinates, and membership (<= threshold)."""
    z = (np.array([nf, gs], float) - np.asarray(S["mu"])) / np.asarray(S["sd"]); d = float(np.sqrt(((np.asarray(S["train_z"]) - z) ** 2).sum(1)).min())
    return d, bool(d <= S["threshold"])


if __name__ == "__main__":
    a = json.load(open(SRC)); T = []
    for c in VOL:
        for s, v in sorted(a["rows"][c].items()):
            if s not in DEMO and all(np.isfinite(v.get(k, np.nan)) for k in ("fp", "neg_fc", "gs", "p2")):
                T.append((v["neg_fc"], v["gs"]))
    T = np.array(T, float)
    if len(T) != 760:
        raise SystemExit(f"expected 760 training rows, got {len(T)}")
    mu, sd = T.mean(0), T.std(0, ddof=0); Z = (T - mu) / sd
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf); nn = D.min(1)
    thr = float(np.quantile(nn, 0.95, method="linear"))
    out = dict(source=SRC, source_sha=hashlib.sha256(open(SRC, "rb").read()).hexdigest(), script_sha=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               n=len(T), mu=mu.tolist(), sd=sd.tolist(), ddof=0, quantile=0.95, quantile_method="linear", threshold=thr, train_z=Z.tolist(),
               note="frozen before any Simulation 3 input; membership: Euclidean distance to the nearest training row in standardised (nf, gs) <= threshold")
    json.dump(out, open(sys.argv[1], "x"), indent=1); print("frozen joint support: n", len(T), "threshold %.17g" % thr, flush=True)
