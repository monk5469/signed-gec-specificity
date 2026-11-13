"""Shared library: known-truth matrices, target and held-out simulations (optionally with a hidden Ornstein-Uhlenbeck input),
band-pass filtering, the held-out endpoint P2 and edge-level metrics."""
from paths import DATA_ROOT
import numpy as np, scipy.io as sio
from scipy.signal import butter, filtfilt, detrend
import jax, jax.numpy as jnp
from jax import lax
import luppi_hopf as L

DEMO = DATA_ROOT + "raw/luppi_demo/gec_3/"
T_LEN = 1189


def demo_subject(k):
    ts = sio.loadmat(DEMO + "FMRI.mat")["BOLD_timeseries_HCP"].ravel()[k].astype(float)
    SC = sio.loadmat(DEMO + "SC.mat")["DTI_fibers_VolNorm_HCP"].ravel()[k].astype(float)
    return ts, SC, L.regional_frequencies([ts])


def support(SC):
    return (SC != 0) & ~np.eye(len(SC), dtype=bool)


def truth_lognormal(seed, sigma_ln):
    """T-LN family: SC_k * lognormal(0, sigma_ln) on the SC support, k = (seed - 1) mod 3, max = 0.1."""
    k = (seed - 1) % 3; _, SC, f = demo_subject(k)
    m = support(SC); rng = np.random.default_rng(seed)
    C = np.zeros_like(SC); C[m] = SC[m] * rng.lognormal(0.0, sigma_ln, m.sum())
    return C / C.max() * 0.1, SC, f, dict(family=f"LN{sigma_ln}", seed=seed, subject=k)


R4V2_DIR = DATA_ROOT + "results/p33/r4v2/"


def truth_qm(seed, sigma_r):
    """T-QM family: realistic coupling scale. Stratum k = (seed - 1) mod 3. The multiset of support-edge weights is
 taken from the stratum's T-SG truth (the cooperative-only main-configuration fit to that example subject, reference fit in R4V2, seed 11;
 includes its exact zeros); the weights are assigned to edges in the order of log SC + N(0, sigma_r^2)
 (default_rng(seed)). The weight multiset is INHERITED from the tested estimator's fit: QM truths are
 randomised re-assignments of estimator-derived weights, not estimator-independent truths. Max weight = 0.1 (inherited)."""
    k = (seed - 1) % 3; _, SC, f = demo_subject(k); m = support(SC)
    ref = np.sort(np.load(R4V2_DIR + f"C_s{k}_header_coop_matlab_seed11.npy")[m])
    rng = np.random.default_rng(seed); score = np.log(SC[m]) + rng.normal(0.0, sigma_r, m.sum())
    w = np.empty(m.sum()); w[np.argsort(score)] = ref
    C = np.zeros_like(SC); C[m] = w
    return C / C.max() * 0.1, SC, f, dict(family=f"QM{sigma_r:g}", seed=seed, subject=k, n_zero=int(np.sum(w == 0)))



def fiedler(SC):
    """Fiedler vector of the SC graph Laplacian, scaled to max |g| = 1 (sign fixed: first nonzero entry positive)."""
    A = (SC + SC.T) / 2; Lap = np.diag(A.sum(1)) - A
    w, V = np.linalg.eigh(Lap); g = V[:, 1]
    g = g / np.abs(g).max(); return g if g[np.flatnonzero(g)[0]] > 0 else -g


def truth_competitive(seed, aligned=False):
    """T-CMP (random 25 % of support edges negated) / T-CMP-H (only edges whose endpoints have opposite Fiedler sign).
 Weights as T-QM1 with the same seed (only nonzero edges can be negated) (stratum (seed - 1) mod 3); signs flipped with default_rng([seed, 1]); then the
 whole matrix is rescaled so that the largest POSITIVE weight is 0.1."""
    C, SC, f, meta = truth_qm(seed, 1.0)
    m = support(SC) & (C > 0); rng = np.random.default_rng([seed, 1]); idx = np.flatnonzero(m.ravel()); n_neg = int(round(0.25 * idx.size))
    if aligned:
        g = fiedler(SC); opp = (np.sign(g)[:, None] * np.sign(g)[None, :] < 0).ravel()
        pool = idx[opp[idx]]; short = pool.size < n_neg
        pick = pool if short else rng.choice(pool, n_neg, replace=False)
    else:
        short = False; pick = rng.choice(idx, n_neg, replace=False)
    Cf = C.ravel().copy(); Cf[pick] = -Cf[pick]; Cf = Cf.reshape(C.shape)
    return Cf / Cf.max() * 0.1, SC, f, dict(meta, family="CMP-H" if aligned else "CMP", n_neg=int(len(pick)), short=bool(short))


def make_driven_simulator(N, T, burnin, TR=L.TR):
    """As luppi_hopf.make_simulator, plus an additive hidden input s * g_i * d(t) in the x equation; d is a unit-variance
 OU process with time constant tau_d (exact AR(1) update per Euler step), started from its stationary law and run
 through the burn-in. s = 0 reproduces the undriven model exactly for the same key (the OU draws use a separate key)."""
    dt = 0.1 * TR / 2; spt = int(round(TR / dt)); nb = L.burn_steps(burnin, dt, spt)

    @jax.jit
    def sim(C, f, a, sigma, key, g, s, tau_d, key_d):
        om = 2 * jnp.pi * f; sumC = C.sum(1); dsig = jnp.sqrt(dt) * sigma
        rho = jnp.exp(-dt / tau_d); q = jnp.sqrt(1 - rho ** 2)
        def step(state, kk):
            (x, y, d), (k, kd) = state, kk
            e = jax.random.normal(k, (2, N), dtype=jnp.float64); r2 = x * x + y * y
            nx = x + dt * (a * x - om * y - x * r2 + C @ x - sumC * x + s * g * d) + dsig * e[0]
            ny = y + dt * (a * y + om * x - y * r2 + C @ y - sumC * y) + dsig * e[1]
            nd = rho * d + q * jax.random.normal(kd, (), dtype=jnp.float64)
            return (nx, ny, nd), None
        k1, k2 = jax.random.split(key); kd0, kd1, kd2 = jax.random.split(key_d, 3)
        st = (0.1 * jnp.ones(N), 0.1 * jnp.ones(N), jax.random.normal(kd0, (), dtype=jnp.float64))
        st, _ = lax.scan(step, st, (jax.random.split(k1, nb), jax.random.split(kd1, nb)))
        def tr_step(st, ks):
            st, _ = lax.scan(step, st, ks); return st, st[0]
        ks = (jax.random.split(k2, T * spt).reshape(T, spt, 2), jax.random.split(kd2, T * spt).reshape(T, spt, 2))
        _, xs = lax.scan(tr_step, st, ks)
        return xs.T
    return sim


def simulate(C, f, seed, sim, g=None, s=0.0, tau_d=50.0, seed_d=0, cfg=L.HEADER):
    N = len(C); g = np.zeros(N) if g is None else g
    return np.asarray(sim(jnp.asarray(C), jnp.asarray(f), cfg["a"], cfg["sigma"], jax.random.PRNGKey(seed),
                          jnp.asarray(g), float(s), float(tau_d), jax.random.PRNGKey(seed_d)))


def bandpass(ts, TR=L.TR, band=(0.008, 0.09)):
    """F branch: the same 2nd-order Butterworth filtfilt (MATLAB padlen) as the authors' frequency extraction."""
    b, a = butter(2, [band[0] / (0.5 / TR), band[1] / (0.5 / TR)], btype="band")
    return filtfilt(b, a, detrend(ts - ts.mean(1, keepdims=True), axis=1), axis=1, padlen=3 * (max(len(a), len(b)) - 1))


def branch(ts, br):
    return L.prep_emp(bandpass(ts) if br == "F" else ts)


def heldout_mse(FC_ho, LAG_ho, FCm, LAGm):
    """P2 endpoint: off-diagonal mean squared FC error + lag-FC error."""
    off = ~np.eye(len(FC_ho), dtype=bool)
    return float(np.mean((FC_ho - FCm)[off] ** 2) + np.mean((LAG_ho - LAGm)[off] ** 2))


def edge_metrics(C_fit, C_true, SC):
    """Sign and magnitude metrics on SC-support edges (diagonal excluded)."""
    m = support(SC); w, t = C_fit[m], C_true[m]; mx = np.abs(w).max()
    q = np.quantile(np.abs(t), [0.25, 0.5, 0.75]); qi = np.searchsorted(q, np.abs(t))
    out = dict(frac_neg=float(np.mean(w < 0)), **{f"frac_neg_thr{th}": float(np.mean(w < -th * mx)) for th in (0.01, 0.05, 0.1)},
               frac_neg_by_true_quartile=[float(np.mean(w[qi == k] < 0)) for k in range(4)],
               spearman=float(_spearman(w, t)), pearson=float(np.corrcoef(w, t)[0, 1]),
               frac_neg_abs2alpha=float(np.mean(w < -2e-4)),          # beyond 2 alpha (alpha = 1e-4), absolute
               frac_neg_true_zero=float(np.mean(w[t == 0] < 0)) if np.any(t == 0) else np.nan,
               frac_neg_true_pos=float(np.mean(w[t > 0] < 0)) if np.any(t > 0) else np.nan)
    if np.any(t < 0):
        tn, tp = t < 0, t > 0
        out.update(sensitivity=float(np.mean(w[tn] < 0)), specificity=float(np.mean(w[tp] >= 0)),
                   precision=float(np.mean(t[w < 0] < 0)) if np.any(w < 0) else np.nan, false_neg_sign_rate=float(np.mean(w[tp] < 0)))
    return out


def _spearman(a, b):
    """Tie-aware Spearman (average ranks, scipy.stats.rankdata); NaN if either vector is constant."""
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
