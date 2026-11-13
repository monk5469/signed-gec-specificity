"""Numerical check of the equivalence construction for symmetric networks with isotropic noise (Supplementary Note 1).
Usage: python ldsec_t14_theorem.py OUT_JSON"""
import sys, json
import numpy as np

def spec(J, q, G, tau, nu):
    N = J.shape[0]; R = np.linalg.inv(1j * nu[:, None, None] * np.eye(N)[None] - J[None])
    inner = q * np.eye(N)[None].astype(complex)
    if G is not None:
        inner = inner + ((2 / tau) / (nu ** 2 + 1 / tau ** 2))[:, None, None] * (G @ G.T)[None]
    return R @ inner @ np.conj(np.transpose(R, (0, 2, 1)))

out = []
for seed in range(6):
    rng = np.random.default_rng(seed); N = 10
    pos = rng.random((N, 2)); D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    W = np.exp(-D / 0.3) * (D < 0.5); np.fill_diagonal(W, 0); W = 0.5 * (W + W.T); W *= 0.7 / np.max(np.abs(np.linalg.eigvalsh(W)))
    JD = W - np.eye(N); ev, V = np.linalg.eigh(JD)
    for mode in (N - 1, N - 2):          # Perron (slowest, positive) and second (mixed-sign) eigenmode
        a = -ev[mode]; ghat = V[:, mode]
        for tau in (20.0, 50.0):
            b = 1 / tau; q = 1.0
            if a <= b:
                continue
            g = ghat * np.sqrt(q * (a ** 2 - b ** 2) / (2 * b))
            JS = JD + (a - b) * np.outer(ghat, ghat)
            nu = 2 * np.pi * np.geomspace(1e-3, 0.69, 300)
            S1, S2 = spec(JD, q, g[:, None], tau, nu), spec(JS, q, None, 1, nu)
            off = ~np.eye(N, dtype=bool); neg = (JS < 0) & off
            out.append(dict(seed=seed, mode="perron" if mode == N - 1 else "second", tau=tau, a=float(a),
                            max_rel_diff=float(np.max(np.abs(S1 - S2)) / np.max(np.abs(S2))),
                            JS_stable=bool(np.max(np.linalg.eigvalsh(JS)) < 0), n_neg_offdiag=int(neg.sum()),
                            frac_neg_with_opposite_loading=float(np.mean((np.outer(ghat, ghat) < 0)[neg])) if neg.any() else None,
                            frac_opposite_pairs_negative=float(np.mean(neg[(np.outer(ghat, ghat) < 0)])) ))
json.dump(out, open(sys.argv[1], "w"), indent=1)
for r in out:
    print(r)
