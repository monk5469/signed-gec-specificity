"""Shared observation components of the known-truth simulations (Section 2.6.2): K = 7 cooperative spatial modes B >= 0 from
symmetric non-negative matrix factorisation of the real mot group FC.
Usage: python e10c_sources.py OUT_NPZ"""
from paths import DATA_ROOT
import sys, os, glob, json, hashlib
import numpy as np
import luppi_hopf as L, luppi_r4e_lib as Lib

W = DATA_ROOT + "work/"; DEMO = {"100307", "100408", "101107"}; KS = (7, 10, 14, 17, 20, 30); DROP = 11
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def symnmf(R, k, iters=3000, seed=0):
    A = np.clip(R, 0, None); off = ~np.eye(len(A), dtype=bool); B = np.random.default_rng(seed).uniform(0.1, 0.5, (len(A), k))
    for _ in range(iters):
        P = B @ B.T; A2 = np.where(off, A, P)
        B = B * (0.5 + 0.5 * (A2 @ B) / np.maximum(P @ B, 1e-12))
    P = B @ B.T; return B, float(np.linalg.norm((R - P)[off]) / np.linalg.norm(R[off]))


if __name__ == "__main__":
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit("exists")
    subs = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(W + "p33_ts_mot/*_REST1_LR_s100.npy")} - DEMO)
    assert len(subs) == 95, len(subs)
    files = [W + f"p33_ts_mot/{s}_REST1_LR_s100.npy" for s in subs]
    R = np.mean([np.corrcoef(L.prep_emp(Lib.bandpass(np.load(p).T.astype(float)[:, DROP:]))) for p in files], 0)
    fits = {k: symnmf(R, k) for k in KS}; res = {k: fits[k][1] for k in KS}
    k = min([k for k in KS if res[k] <= 0.05] or [max(KS)]); B = fits[k][0]; B = B / np.sqrt(np.mean(np.sum(B ** 2, 1)))
    np.savez(out, B=B, k=k, ks=np.array(KS), residuals=np.array([res[x] for x in KS]))
    json.dump(dict(subjects=subs, series_sha={p: sha(p) for p in files}, k=k, residuals=res, rule="smallest k with residual <= 0.05",
                   B_summary=dict(min=float(B.min()), max=float(B.max()), zero_share=float(np.mean(B < 1e-6))), npz_sha=sha(out)),
              open(out + ".json", "x"), indent=1)
    print("k", k, "residuals", {x: round(v, 3) for x, v in res.items()}, "B max %.3f zero share %.2f" % (B.max(), np.mean(B < 1e-6)), flush=True)
