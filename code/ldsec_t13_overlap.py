"""Numerical verification of the linear observational equivalence (Supplementary Note 1): exact spectral overlap of a
cooperative network with hidden drive and a signed network (NumPy/SciPy only).
Usage: python ldsec_t13_overlap.py OUT_JSON"""
import sys, json, itertools, multiprocessing as mp
import numpy as np
from scipy.optimize import minimize

BANDS = {"full": (0.001, 0.69), "bold": (0.01, 0.08)}     # Hz


def spec(J, Sig, G, tau, eta, nu):
    N = J.shape[0]; I = np.eye(N)
    R = np.linalg.inv(1j * nu[:, None, None] * I[None] - J[None])
    inner = np.broadcast_to(Sig, (len(nu), N, N)).astype(complex)
    if G is not None:
        sd = (2.0 / tau) / (nu ** 2 + 1.0 / tau ** 2)
        inner = inner + sd[:, None, None] * (G @ G.T)[None]
    return R @ inner @ np.conj(np.transpose(R, (0, 2, 1))) + eta * I[None]


def kl(S0, S):
    X = np.linalg.solve(S, S0)
    return float(np.mean(np.real(np.trace(X, axis1=1, axis2=2)) - np.real(np.log(np.linalg.det(X))) - S0.shape[1]))


def nu_grid(band, n=400):
    return 2 * np.pi * np.geomspace(band[0], band[1], n)


def fit_MS(S0, eta, nu, starts=8, seed=0):
    rng = np.random.default_rng(seed); best = (np.inf, None)
    def f(p):
        J = p[:4].reshape(2, 2); Sig = np.diag(np.exp(p[4:6]))
        if np.max(np.linalg.eigvals(J).real) >= -1e-6:
            return 1e6
        return kl(S0, spec(J, Sig, None, 1.0, eta, nu))
    for s in range(starts):
        p0 = np.concatenate([(-np.eye(2) * rng.uniform(0.05, 2) + rng.normal(0, 0.1, (2, 2))).ravel(), rng.normal(0, 1, 2)])
        r = minimize(f, p0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-10, fatol=1e-14))
        r = minimize(f, r.x, method="BFGS", options=dict(gtol=1e-12, maxiter=5000))
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best


def fit_MD(S0, eta, nu, starts=8, seed=0):
    """Cooperative (Metzler) 2x2 J + one driver (free loadings, free tau) + diagonal Sigma."""
    rng = np.random.default_rng(seed); best = (np.inf, None)
    def f(p):
        J = np.array([[-np.exp(p[0]), np.exp(p[1])], [np.exp(p[2]), -np.exp(p[3])]])
        if np.max(np.linalg.eigvals(J).real) >= -1e-6:
            return 1e6
        return kl(S0, spec(J, np.diag(np.exp(p[4:6])), p[6:8][:, None], np.exp(p[8]), eta, nu))
    for s in range(starts):
        p0 = np.concatenate([rng.normal(0, 0.5, 1), rng.normal(-3, 1, 2), rng.normal(0, 0.5, 1), rng.normal(0, 1, 2), rng.normal(0, 1, 2), [np.log(rng.uniform(5, 60))]])
        r = minimize(f, p0, method="Nelder-Mead", options=dict(maxiter=40000, xatol=1e-10, fatol=1e-14))
        r = minimize(f, r.x, method="BFGS", options=dict(gtol=1e-12, maxiter=5000))
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best


def job_D(args):
    gr, tau, eta, band, a, q = args
    b = 1.0 / tau; gstar = np.sqrt(q * (a ** 2 - b ** 2) / (4 * b)); g = gr * gstar
    nu = nu_grid(BANDS[band])
    S0 = spec(-a * np.eye(2), q * np.eye(2), np.array([[g], [-g]]), tau, eta, nu)
    v, _ = fit_MS(S0, eta, nu)
    return dict(dir="MD_truth_fit_MS", g_ratio=gr, tau=tau, eta=eta, band=band, kl_rate=v)


def job_S(args):
    off, eta, band, a = args
    J = np.array([[-a, off], [off, -a]]); nu = nu_grid(BANDS[band])
    S0 = spec(J, np.eye(2), None, 1.0, eta, nu)
    v, _ = fit_MD(S0, eta, nu)
    return dict(dir="MS_truth_fit_MD", offdiag=off, eta=eta, band=band, kl_rate=v)


if __name__ == "__main__":
    out = {}
    # (1) analytic identity
    a, q = 1.0, 1.0; res = []
    for tau in (5.0, 20.0, 50.0):
        b = 1 / tau; g = np.sqrt(q * (a ** 2 - b ** 2) / (4 * b)); nu = nu_grid(BANDS["full"])
        SD = spec(-a * np.eye(2), q * np.eye(2), np.array([[g], [-g]]), tau, 0.0, nu)
        JS = np.array([[-(a + b) / 2, -(a - b) / 2], [-(a - b) / 2, -(a + b) / 2]])
        SS = spec(JS, q * np.eye(2), None, 1.0, 0.0, nu)
        res.append(dict(tau=tau, g_star=float(g), JS_offdiag=float(JS[0, 1]), max_rel_diff=float(np.max(np.abs(SD - SS)) / np.max(np.abs(SS)))))
    out["identity"] = res
    # (2) landscapes
    jobsD = list(itertools.product((0.0, 0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0), (5.0, 20.0, 50.0), (0.0, 0.1, 1.0), ("full", "bold"), (1.0,), (1.0,)))
    jobsS = list(itertools.product((-0.6, -0.3, -0.1, 0.0, 0.1, 0.3), (0.0, 0.1, 1.0), ("full", "bold"), (1.0,)))
    with mp.Pool(48) as p:
        out["MD_truth"] = p.map(job_D, jobsD)
        out["MS_truth"] = p.map(job_S, jobsS)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out["identity"], indent=1))
    for r in out["MD_truth"]:
        if r["eta"] == 0.0:
            print("MD", r["band"], "tau", r["tau"], "g/g*", r["g_ratio"], "KL %.3e" % r["kl_rate"])
    for r in out["MS_truth"]:
        print("MS", r["band"], "eta", r["eta"], "offdiag", r["offdiag"], "KL %.3e" % r["kl_rate"])
