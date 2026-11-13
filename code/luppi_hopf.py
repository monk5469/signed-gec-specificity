"""Independent reimplementation of the Luppi et al. (2026) generative-effective-connectivity (GEC) estimator.
Written from the authors' released source (github.com/Hana-Ali/competitive-cooperative-hopf @da592aa). Dependencies: JAX for the compiled Euler loop, NumPy/SciPy elsewhere.

Model (per region i, z_i = (x_i, y_i)), Euler forward with dt = 0.1 * TR / 2 (released C++/MATLAB form):
 dz = [a z + zz.* [-w, +w] - z (x^2 + y^2) + C z - sumC.* z] dt + sigma sqrt(dt) N(0, 1), zz = (y, x), w = 2 pi f
 i.e. dx = a x - w y - x r^2 + (Cx - sumC x), dy = a y + w x - y r^2 + (Cy - sumC y)
 "BOLD" = x sampled every TR (no HRF, no filtering). Burn-in of `burnin` seconds before every simulation
 (MATLAB fcn_Hopf_simulate_BOLD_from_GEC.m: `for t=0:dt:3000`). Initial state 0.1.
The default fit follows the MATLAB fitting function fcn_Hopf_NonLinear_cooperative_competitive.m:
 targets from the demeaned + detrended empirical series; lag-FC in MATLAB xcov orientation, L[i, j] = cov(x_i[t+tau], x_j[t])
 (the C++ port uses the transpose); start 0.2 SC / max SC; diagonal not updated; error checked every 100
 iterations BEFORE the update, stop if (old - now) / now < conv (includes any increase).
 Update (SC != 0, i != j): C += eta_FC dFC + eta_tau dLag - alpha sign(C); cooperative clip; if alpha == 0: C = C / max(C) * max_C.
fit(..., legacy=True) selects the C++-port conventions.
Regional frequencies (fcn_extract_frequencies.m): 2nd-order Butterworth band-pass of demeaned + detrended series, FFT power
pw(1:T/2)^2 / (T/TR), averaged over regions' subjects, Gaussian smoothing (0.01 Hz), peak frequency (Hz).
"""
import time
import numpy as np
from scipy.signal import butter, filtfilt, detrend

import jax, jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)
RELEASED = dict(sigma=0.001, max_iterations=2000, eta_fc=1e-3, eta_tau=1e-4, alpha=1e-4, max_C=0.15, conv=5e-4, tau=1, a=-0.02, burnin=3000.0)
HEADER = dict(sigma=0.01, max_iterations=5000, eta_fc=2e-4, eta_tau=4e-5, alpha=0.0, max_C=0.1, conv=1e-3, tau=1, a=-0.02, burnin=3000.0)
L1P = dict(HEADER, alpha=1e-4)   # paper-stated settings (eps 2e-4 / 4e-5, beta 0.01) + the optional L1 step; no max_C normalisation when alpha > 0 (code)
TR = 0.72


def gaussfilt(t, z, sigma):
    """Port of the authors' gaussfilt.m (uniform-grid branch): kernel dt*a*exp(-0.5 (t - mean t)^2 / sigma^2) over the
 whole grid, entries < dt*a*1e-6 removed, MATLAB conv(z, kernel, 'same') (zero padding; central part)."""
    dt = t[1] - t[0]; a = 1 / (np.sqrt(2 * np.pi) * sigma)
    k = dt * a * np.exp(-0.5 * (t - t.mean()) ** 2 / sigma ** 2); k = k[k >= dt * a * 1e-6]
    full = np.convolve(z, k); s = len(k) // 2                  # MATLAB 'same': c(floor(m/2)+1 : floor(m/2)+n)
    return full[s:s + len(z)]


def regional_frequencies(ts_list, TR=TR, band=(0.008, 0.09), smooth_hz=0.01):
    """Port of fcn_extract_frequencies.m."""
    b, a = butter(2, [band[0] / (0.5 / TR), band[1] / (0.5 / TR)], btype="band")
    T = ts_list[0].shape[1]; freqs = np.arange(int(np.ceil(T / 2 - 1)) + 1)[: T // 2] / (T * TR)
    P = np.mean([[np.abs(np.fft.fft(filtfilt(b, a, detrend(x - x.mean()), padlen=3 * (max(len(a), len(b)) - 1))))[:T // 2] ** 2 / (T / TR) for x in ts] for ts in ts_list], axis=0)
    Ps = np.array([gaussfilt(freqs, p, smooth_hz) for p in P])
    f = freqs[np.argmax(Ps, axis=1)]
    f[f == 0] = f[f != 0].mean()
    return f


def fc(ts):
    return np.corrcoef(ts)


def lag_fc(ts, tau=1, convention="matlab"):
    """Lagged FC with sigratio from cov (ddof=1) and the lag sum divided by T, as in the authors' code.
 convention="matlab" (fcn_Hopf_NonLinear_cooperative_competitive.m, MATLAB xcov(x_i, x_j, tau)):
 L[i, j] = sum_t x_i[t + tau] x_j[t] / T / (sd_i sd_j) (i in the future of j)
 convention="cpp" (cpp/hopf_optimization.hpp compute_lagged_FC): the transpose,
 L[i, j] = sum_t x_i[t] x_j[t + tau] / T / (sd_i sd_j)."""
    x = ts - ts.mean(1, keepdims=True); T = x.shape[1]; sd = x.std(1, ddof=1)
    L = (x[:, :-tau] @ x[:, tau:].T / T) / np.outer(sd, sd)
    return L.T if convention == "matlab" else L


def burn_steps(burnin, dt, spt, phase="matlab"):
    """Euler steps before the first 20-step sampling block.
 phase="matlab": MATLAB runs len(0:dt:burnin) = floor(burnin/dt) + 1 burn-in steps, then samples x after modelling steps
 1, 1 + spt, 1 + 2 spt,... (`abs(mod(t,TR))<0.01` with t = 0, dt,...). Sampling at the end of each spt-step block therefore
 needs floor(burnin/dt) + 1 + 1 - spt burn-in steps. phase="v1": ceil(burnin/dt)."""
    if phase == "v1":
        return int(np.ceil(burnin / dt))
    return int(np.floor(burnin / dt + 1e-9)) + 2 - spt


def make_simulator(N, T, burnin, TR=TR, phase="matlab"):
    dt = 0.1 * TR / 2; spt = int(round(TR / dt)); nb = burn_steps(burnin, dt, spt, phase)

    @jax.jit
    def sim(C, f, a, sigma, key):
        om = 2 * jnp.pi * f; sumC = C.sum(1); dsig = jnp.sqrt(dt) * sigma
        def step(state, k):
            x, y = state; e = jax.random.normal(k, (2, N), dtype=jnp.float64); r2 = x * x + y * y
            nx = x + dt * (a * x - om * y - x * r2 + C @ x - sumC * x) + dsig * e[0]
            ny = y + dt * (a * y + om * x - y * r2 + C @ y - sumC * y) + dsig * e[1]
            return (nx, ny), None
        k1, k2 = jax.random.split(key)
        st = (0.1 * jnp.ones(N), 0.1 * jnp.ones(N))
        st, _ = lax.scan(step, st, jax.random.split(k1, nb))
        def tr_step(st, ks):
            st, _ = lax.scan(step, st, ks); return st, st[0]
        _, xs = lax.scan(tr_step, st, jax.random.split(k2, T * spt).reshape(T, spt, 2))
        return xs.T
    return sim



class Cancelled(Exception):
    pass


def check_deadline(stop_at, where):
    """Hard deadline: raise Cancelled if stop_at (epoch s) has passed. An already running simulator call may
 finish, but no further simulation is started and the job cannot be accepted as done."""
    if stop_at is not None and time.time() > stop_at:
        raise Cancelled(f"deadline reached ({where})")


def update_step(C, mask, dFC, dLag, c, cooperative_only, l1_order="matlab"):
    """One GEC update (MATLAB loop body + post-loop normalisation). l1_order "matlab": sign of the pre-update weight;
 "cpp": sign after the FC / lag increments (C++ port; legacy option)."""
    pre = np.sign(C)
    C = C + mask * (c["eta_fc"] * dFC + c["eta_tau"] * dLag)
    if c["alpha"] > 0:
        C = C - c["alpha"] * (pre if l1_order == "matlab" else np.sign(C)) * mask
    if cooperative_only:
        C = np.where(mask > 0, np.maximum(C, 0.0), C)
    if c["alpha"] == 0:
        C = C / C.max() * c["max_C"]
    return C


def prep_emp(ts):
    """MATLAB: TS(seed,:) = detrend(TS(seed,:) - nanmean(TS(seed,:))) before FC / lag-FC targets (no filtering)."""
    return detrend(ts - ts.mean(1, keepdims=True), axis=1)


def fit(emp_ts, SC, f, cfg, seed, cooperative_only=False, C0=None, record_every=100, sim=None, legacy=False,
        lag="matlab", eval_sims=0, emp_prepared=False, trace=None, stop_at=None):
    """MATLAB-faithful fit, following fcn_Hopf_NonLinear_cooperative_competitive.m:
 - targets from the demeaned + detrended empirical series (prep_emp), lag convention `lag`;
 - GEC initialised as 0.2 * SC / max(SC) (also the C++ setup_for_optimization); diagonal never updated;
 - each iteration: one simulation -> at iterations divisible by 100 the error is checked BEFORE the update:
 stop if (olderror - errornow) / errornow < conv (this includes any increase; reason recorded); olderror starts at 1e5;
 - update on SC != 0 off-diagonal: + eta_FC dFC + eta_tau dLag - alpha sign(GEC); cooperative clip; if alpha == 0,
 GEC = GEC / max(GEC) * max_C. Returned GEC = the matrix that produced the last checked simulation if stopped,
 else the last iterate.
 L1 order: the sign is taken from the PRE-update weight, as in the MATLAB expression
 GEC(i,j) = GEC(i,j) + eta dFC + eta_tau dLag - alphaL1 * sign(GEC(i,j)); the C++ port (and legacy) uses the post-update sign.
 trace(it, FC_sim, LAG_sim, C_after_update): optional per-iteration callback (fixtures). stop_at: epoch seconds; checked before and
 after every simulation, including the post-fit one, so a passed deadline raises Cancelled.
 legacy=True selects the C++-port conventions: no detrend, unscaled SC start, cpp lag convention, check after update.
 eval_sims > 0: also returns FC / lag-FC averaged over that many fresh simulations of the returned GEC."""
    if legacy:
        lag = "cpp"
    T = emp_ts.shape[1]; N = SC.shape[0]; c = dict(cfg)
    sim = sim or make_simulator(N, T, c["burnin"])
    key = jax.random.PRNGKey(seed)
    E = emp_ts if (legacy or emp_prepared) else prep_emp(emp_ts)
    FCe, LFe = fc(E), lag_fc(E, c["tau"], lag)
    offd = ~np.eye(N, dtype=bool)
    mask = ((SC != 0) & (offd | legacy)).astype(float)
    if C0 is not None:
        C = np.asarray(C0, float).copy()
    else:
        C = SC.astype(float).copy() if legacy else 0.2 * SC / SC.max()
    old_err = 1.0 if legacy else 1e5; hist = []; reason = "max_iterations"; it = 0; FCs = LFs = None
    for it in range(1, c["max_iterations"] + 1):
        check_deadline(stop_at, f"before simulation, iteration {it}")
        key, k = jax.random.split(key)
        s = np.asarray(sim(jnp.asarray(C), jnp.asarray(f), c["a"], c["sigma"], k))
        check_deadline(stop_at, f"after simulation, iteration {it}")
        FCs, LFs = fc(s), lag_fc(s, c["tau"], lag)
        if not legacy and it % record_every == 0:
            err = float(np.mean((FCe - FCs) ** 2) + np.mean((LFe - LFs) ** 2)); hist.append((it, err))
            if (old_err - err) / err < c["conv"]:
                reason = "error_increase" if err > old_err else "converged"; break
            old_err = err
        C = update_step(C, mask, FCe - FCs, LFe - LFs, c, cooperative_only, "cpp" if legacy else "matlab")
        if trace is not None:
            trace(it, FCs, LFs, C)
        if legacy and it % record_every == 0:
            err = float(np.mean((FCe - FCs) ** 2) + np.mean((LFe - LFs) ** 2)); hist.append((it, err))
            if old_err < err:
                reason = "error_increase"; break
            imp = (old_err - err) / err
            if 0 <= imp < c["conv"]:
                reason = "converged"; break
            old_err = err
    iu = np.triu_indices(N, 1)
    info = dict(history=hist, stop=reason, iterations=it, lag=lag, legacy=legacy,
                fc_corr_last=float(np.corrcoef(FCe[iu], FCs[iu])[0, 1]))
    check_deadline(stop_at, "before the post-fit simulation")
    key, k = jax.random.split(key)
    s = np.asarray(sim(jnp.asarray(C), jnp.asarray(f), c["a"], c["sigma"], k))
    check_deadline(stop_at, "after the post-fit simulation")
    info["fc_corr"] = float(np.corrcoef(FCe[iu], fc(s)[iu])[0, 1])
    if eval_sims:
        FCm, LFm = evaluate(C, f, c, sim, key, eval_sims, lag)
        info.update(eval_FC=FCm, eval_LAG=LFm)
    return C, info


def evaluate(C, f, c, sim, key, n, lag="matlab"):
    """FC and lag-FC averaged over n independent fresh simulations of C (keys derived from `key`)."""
    F, L = [], []
    for k in jax.random.split(jax.random.fold_in(key, 7919), n):
        s = np.asarray(sim(jnp.asarray(C), jnp.asarray(f), c["a"], c["sigma"], k)); F.append(fc(s)); L.append(lag_fc(s, c["tau"], lag))
    return np.mean(F, 0), np.mean(L, 0)


def neg_stats(C, SC, thresholds=(0.0, 0.01, 0.05, 0.1)):
    """Negative-weight fractions among SC-present off-diagonal edges, raw and with |w| > thr * max|w|."""
    m = (SC != 0) & ~np.eye(C.shape[0], dtype=bool); w = C[m]; mx = np.max(np.abs(w))
    out = {f"frac_neg_thr{t}": float(np.mean(w < -t * mx)) for t in thresholds}
    out.update(n_edges=int(m.sum()), max_abs=float(mx), median_abs_neg=float(np.median(np.abs(w[w < 0]))) if np.any(w < 0) else None,
               median_abs_pos=float(np.median(w[w > 0])) if np.any(w > 0) else None)
    return out
