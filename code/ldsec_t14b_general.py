"""Numerical check of the general equivalence construction (non-symmetric coupling, anisotropic noise; Eq. (5), Supplementary
Note 1), including the sparse, spatially embedded examples.
Usage: python ldsec_t14b_general.py OUT_JSON"""
import sys, json
import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize_scalar


def spec(J, Sig, g, b, nu):
    N = J.shape[0]; R = np.linalg.inv(1j * nu[:, None, None] * np.eye(N)[None] - J[None])
    inner = np.broadcast_to(Sig, (len(nu), N, N)).astype(complex)
    if g is not None:
        inner = inner + (2 * b / (nu ** 2 + b ** 2))[:, None, None] * np.outer(g, g)[None]
    return R @ inner @ np.conj(np.transpose(R, (0, 2, 1)))


def network(rng, N, kind):
    pos = rng.random((N, 2)); D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    W = np.exp(-D / 0.3) * rng.random((N, N)) * (D < 0.6); np.fill_diagonal(W, 0)
    W *= 0.7 / np.max(np.abs(np.linalg.eigvals(W)))
    gx = pos[:, 0] - pos[:, 0].mean()
    if kind == "comp_random":
        W = W * np.where(rng.random((N, N)) < 0.3, -1, 1)
    elif kind == "comp_hier":
        W = W * np.where(np.outer(gx, gx) < -0.02, -1, 1)
    tsc = rng.uniform(0.5, 2, N)
    J = (W - np.eye(N)) / tsc[:, None]
    lam = np.max(np.linalg.eigvals(J).real)
    if lam > -0.05:
        J -= (lam + 0.05) * np.eye(N)
    return J, gx


def real_left_modes(J):
    w, vl = eig(J, left=True, right=False)
    out = []
    for i in np.argsort(-w.real):
        if abs(w[i].imag) < 1e-10 and w[i].real < 0:
            out.append((-w[i].real, np.real(vl[:, i])))
    return out


def min_offdiag(M):
    off = ~np.eye(M.shape[0], dtype=bool); return float(M[off].min())


def reduce_best(JS, Sig, a_grid_factor=np.geomspace(1.001, 50, 200)):
    """Best one-driver cooperative reduction: maximise the most negative off-diagonal of J_D over modes and a."""
    best = dict(min_off=min_offdiag(JS), mode=None, a=None, b=None)
    for b, u in real_left_modes(JS):
        Su = Sig @ u; den = u @ Su
        for f in a_grid_factor:
            a = b * f; JD = JS - (a - b) * np.outer(Su, u) / den
            m = min_offdiag(JD)
            if m > best["min_off"]:
                best = dict(min_off=m, b=float(b), a=float(a), mode_pos_frac=float(np.mean(u > 0)), _u=u, _den=den, _Su=Su)
    if best.get("_u") is not None:          # exact refinement: min off-diagonal is concave piecewise-linear in a
        u, den, Su, b = best.pop("_u"), best.pop("_den"), best.pop("_Su"), best["b"]
        f = lambda a: -min_offdiag(JS - (a - b) * np.outer(Su, u) / den)
        r = minimize_scalar(f, bounds=(b * 1.0000001, b * 60), method="bounded", options=dict(xatol=1e-13 * b))
        if -r.fun > best["min_off"]:
            best["min_off"], best["a"] = float(-r.fun), float(r.x)
    best["reducible"] = bool(best["min_off"] >= -1e-8 * np.max(np.abs(JS)))
    return best


if __name__ == "__main__":
    out = dict(forward=[], reverse=[]); nu = 2 * np.pi * np.geomspace(1e-3, 0.69, 300)
    for seed in range(8):
        rng = np.random.default_rng(seed); N = 12
        J, gx = network(rng, N, "coop"); Sig = np.diag(rng.uniform(0.5, 2, N))
        for k, (a, u) in enumerate(real_left_modes(J)[:3]):
            for tau in (20.0, 50.0):
                b = 1 / tau
                if a <= b:
                    continue
                g = Sig @ u; g *= np.sqrt(((a ** 2 - b ** 2) / (2 * b)) / (g @ np.linalg.solve(Sig, g)))
                uu = np.linalg.solve(Sig, g)
                JS = J + (a - b) * np.outer(g, uu) / (g @ uu)
                S1, S2 = spec(J, Sig, g, b, nu), spec(JS, Sig, None, b, nu)
                off = ~np.eye(N, dtype=bool); neg = (JS < 0) & off
                out["forward"].append(dict(seed=seed, mode_rank=k, mixed_sign=bool((u > 0).any() and (u < 0).any()), tau=tau,
                                           max_rel_diff=float(np.max(np.abs(S1 - S2)) / np.max(np.abs(S2))),
                                           JS_max_eig=float(np.max(np.linalg.eigvals(JS).real)), frac_neg_offdiag=float(neg.sum() / off.sum()),
                                           frac_neg_opposite=float(np.mean((np.outer(g, uu) < 0)[neg])) if neg.any() else None))
                if k >= 1 and tau == 20.0:          # (i) constructed equivalent must be reducible back
                    r = reduce_best(JS, Sig); r.update(case="constructed_from_driver", seed=seed, mode_rank=k); out["reverse"].append(r)
        for kind in ("comp_random", "comp_hier"):
            JSc, _ = network(np.random.default_rng(100 + seed), N, kind)
            r = reduce_best(JSc, Sig); r.update(case=kind, seed=seed, frac_neg=float(np.mean(JSc[~np.eye(N, dtype=bool)] < 0)))
            out["reverse"].append(r)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for r in out["forward"]:
        print("FWD", r)
    for r in out["reverse"]:
        print("REV", r)
