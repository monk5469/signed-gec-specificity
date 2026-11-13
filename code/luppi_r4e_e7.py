"""Ground-truth generation and observation functions (structurally constrained truths, observation noise, linear and Hopf
simulators), used by Simulations 2-5."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e_e7/"
R4E = DATA_ROOT + "results/p33/r4e/"
DEMO = DATA_ROOT + "raw/luppi_demo/"
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_e7.py", "luppi_r4e_e7_analyze.py", "luppi_r4e_analyze.py", "luppi_r4e_run.py"]
RAW = [DEMO + "gec_3/FMRI.mat", DEMO + "gec_3/SC.mat", R4E + "r4e_manifest.json"]
CELLS = ("G0", "G1", "G2", "G3", "G4", "G5", "G6")
SCG_SEEDS = tuple(range(5001, 5031)); QM1_SEEDS = tuple(range(1001, 1031))
EXTRA = 45                       # frames simulated before the kept window (HRF warm-up)
T_KEEP = 1189
GLOBAL_SHARE, GLOBAL_TAU = 0.30, 20.0
SHARE_TOL = 0.05                 #: |realised broadband median share - 0.30| <= 0.05
LIN_LAMBDA = 0.05
DEADLINE_H = 72.0
EMP_MEAN_FC = None               # filled in prepare from the example data (R branch)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------- observation models
def hrf(TR):
    """SPM canonical double-gamma HRF sampled at TR over 32 s, unit sum."""
    from scipy.stats import gamma
    t = np.arange(0, 32, TR); h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6.0
    return h / h.sum()


def apply_hrf(x, TR):
    from scipy.signal import lfilter
    return lfilter(hrf(TR), [1.0], x, axis=1)


def ou_series(rng, n, tau, TR):
    phi = np.exp(-TR / tau); e = rng.standard_normal(n) * np.sqrt(1 - phi ** 2); e[0] = rng.standard_normal()
    from scipy.signal import lfilter
    return lfilter([1.0], [1.0, -phi], e)


def add_global(x, rng, TR, stats=None):
    """Observation-level global component; GLOBAL_SHARE is the NOMINAL BROADBAND median regional variance share.
 Realised broadband and F-branch shares are recorded in `stats`."""
    import luppi_r4e_lib as Lib
    z = ou_series(rng, x.shape[1], GLOBAL_TAU, TR); z = (z - z.mean()) / z.std(); c = 1 + 0.2 * rng.standard_normal(x.shape[0])
    def share(s):                                        # realised median regional share, cross terms included
        g = s * c[:, None] * z[None, :]; return float(np.median(g.var(1) / (x + g).var(1)))
    lo, hi = -6.0, 6.0                                   # bisection on log10 s: exact realised median share
    for _ in range(60):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if share(10 ** mid * np.sqrt(np.median(x.var(1)))) < GLOBAL_SHARE else (lo, mid)
    s = 10 ** ((lo + hi) / 2) * np.sqrt(np.median(x.var(1))); g = s * c[:, None] * z[None, :]
    if stats is not None:
        stats["global_share_broadband"] = float(np.median(g.var(1) / (x + g).var(1)))
        bg, bx = Lib.bandpass(g), Lib.bandpass(x + g); stats["global_share_F"] = float(np.median(bg.var(1) / bx.var(1)))
    return x + g


def add_noise(x, rng, stats=None):
    """White observation noise scaled per region so that the IN-BAND (0.008-0.09 Hz, the estimator's F-branch input) noise
 variance equals the in-band signal variance: in-band SNR = 1."""
    import luppi_r4e_lib as Lib
    e = rng.standard_normal(x.shape); scale = np.sqrt(Lib.bandpass(x).var(1) / Lib.bandpass(e).var(1))
    n = scale[:, None] * e
    if stats is not None:
        stats["snr_F"] = float(np.median(Lib.bandpass(x).var(1) / Lib.bandpass(n).var(1))); stats["snr_broadband"] = float(np.median(x.var(1) / n.var(1)))
    return x + n


# ---------------------------------------------------------------- simulators
_SIMS = {}


def hopf_sim():
    import luppi_hopf as L
    if "hopf" not in _SIMS:
        _SIMS["hopf"] = L.make_simulator(100, T_KEEP + EXTRA, 3000.0)
    return _SIMS["hopf"]


def linear_sim():
    """Linear OU network dx = (-lam x + C x - sumC x) dt + sigma dW (Euler, same dt / sampling / burn-in as the Hopf)."""
    if "lin" in _SIMS:
        return _SIMS["lin"]
    import jax, jax.numpy as jnp
    from jax import lax
    import luppi_hopf as L
    TR = L.TR; dt = 0.1 * TR / 2; spt = int(round(TR / dt)); nb = L.burn_steps(3000.0, dt, spt); N, T = 100, T_KEEP + EXTRA

    @jax.jit
    def sim(C, lam, sigma, key):
        sumC = C.sum(1); dsig = jnp.sqrt(dt) * sigma
        def step(x, k):
            e = jax.random.normal(k, (N,), dtype=jnp.float64); return x + dt * (-lam * x + C @ x - sumC * x) + dsig * e, None
        k1, k2 = jax.random.split(key); x, _ = lax.scan(step, jnp.zeros(N), jax.random.split(k1, nb))
        def tr_step(x, ks):
            x, _ = lax.scan(step, x, ks); return x, x
        _, xs = lax.scan(tr_step, x, jax.random.split(k2, T * spt).reshape(T, spt, 2))
        return xs.T
    _SIMS["lin"] = sim
    return sim


def generate(C, f, s, cell, held, stats=None):
    """One target (held False) or held-out (True) series (100 x 1189) under the cell's observation model."""
    import jax, jax.numpy as jnp
    import luppi_hopf as L
    ci = CELLS.index(cell); key = jax.random.PRNGKey(11_000_000 + 100 * s + ci + (500_000 if held else 0))
    rng = np.random.default_rng([s, ci, 2 if held else 1]); cfg = L.HEADER; TR = L.TR
    if cell == "G6":
        x = np.asarray(linear_sim()(jnp.asarray(C), LIN_LAMBDA, cfg["sigma"], key))
    else:
        a = cfg["a"]
        if cell in ("G4", "G5"):
            a = np.random.default_rng([s, 4]).uniform(-0.04, -0.01, len(C))
        x = np.asarray(hopf_sim()(jnp.asarray(C), jnp.asarray(f), jnp.asarray(a), cfg["sigma"], key))
    if cell in ("G1", "G5", "G6"):
        x = apply_hrf(x, TR)
    x = x[:, EXTRA:]
    if cell in ("G3", "G5", "G6"):
        x = add_global(x, rng, TR, stats)
    if cell in ("G2", "G5"):
        x = add_noise(x, rng, stats)
    return x


# ---------------------------------------------------------------- truths and calibration
def scg_truth(s, gamma_k):
    import luppi_r4e_lib as Lib
    k = (s - 1) % 3; _, SC, f = Lib.demo_subject(k); m = Lib.support(SC); rng = np.random.default_rng(s)
    C = np.zeros_like(SC); C[m] = SC[m] ** gamma_k[k] * rng.lognormal(0.0, 0.5, m.sum())
    return C / C.max() * 0.1, SC, f, k


def mean_fc(x):
    import luppi_hopf as L
    F = np.corrcoef(L.prep_emp(x)); return float(F[~np.eye(len(F), dtype=bool)].mean())


def calibrate_gamma(k, target):
    """Bisection on gamma in [0.1, 1] (20 steps): mean FC of G0 targets (2 calibration seeds, first SCG truth of the stratum's
 lognormal draw) equals the stratum's empirical example mean FC. Mean FC decreases with gamma (more uneven weights)."""
    import jax, jax.numpy as jnp
    import luppi_hopf as L, luppi_r4e_lib as Lib
    s0 = min(s for s in SCG_SEEDS if (s - 1) % 3 == k); _, SC, f = Lib.demo_subject(k); m = Lib.support(SC)
    noise = np.random.default_rng(s0).lognormal(0.0, 0.5, m.sum()); lo, hi = 0.1, 1.0; trace = []
    for _ in range(20):
        g = (lo + hi) / 2; C = np.zeros_like(SC); C[m] = SC[m] ** g * noise; C = C / C.max() * 0.1
        v = float(np.mean([mean_fc(np.asarray(hopf_sim()(jnp.asarray(C), jnp.asarray(f), L.HEADER["a"], L.HEADER["sigma"],
                                                         jax.random.PRNGKey(12_000_000 + 10 * k + j)))[:, EXTRA:]) for j in range(2)]))
        trace.append((g, v)); lo, hi = (g, hi) if v > target else (lo, g)
    best = min(trace, key=lambda t: abs(t[1] - target))
    ends = {}
    for g in (0.1, 1.0):                                 #: both endpoints with the same draws and keys
        C = np.zeros_like(SC); C[m] = SC[m] ** g * noise; C = C / C.max() * 0.1
        ends[g] = float(np.mean([mean_fc(np.asarray(hopf_sim()(jnp.asarray(C), jnp.asarray(f), L.HEADER["a"], L.HEADER["sigma"],
                                                             jax.random.PRNGKey(12_000_000 + 10 * k + j)))[:, EXTRA:]) for j in range(2)]))
    bracket = bool(min(ends.values()) <= target <= max(ends.values()))
    return dict(stratum=k, target=target, gamma=best[0], mean_fc=best[1], trace=trace, endpoints=ends, endpoints_bracket_target=bracket,
                success=bool(abs(best[1] - target) <= 0.1 * target),
                note=None if bracket else "target not attained by the prescribed search (endpoints do not bracket it)")


def truth_job(args):
    fam, s, gamma_k = args
    import luppi_r4e_lib as Lib
    tries = [s] if fam == "QM1" else [s, s + 6000, s + 12000]; log = []
    for s2 in tries:
        if fam == "QM1":
            C = np.load(R4E + f"truth/QM1_{s}_C.npy"); f = np.load(R4E + f"truth/QM1_{s}_f.npy"); k = (s - 1) % 3
        else:
            C, _, f, k = scg_truth(s2, gamma_k)
        cells = [c for c in CELLS if not (fam == "QM1" and c == "G0")]; files, desc, ok = [], {}, True
        tag = f"{fam}_{s}"
        if fam == "SCG":
            for n, a in (("C", C), ("f", f)):
                p = f"truth/{tag}_{n}.npy"; np.save(OUT + p, a); files.append(dict(file=p, sha=sha(OUT + p)))
        for c in cells:
            st_t, st_h = {}, {}
            tgt = generate(C, f, s2, c, False, st_t); ho = generate(C, f, s2, c, True, st_h)
            ok &= bool(np.all(np.isfinite(tgt)) and np.all(np.isfinite(ho)) and np.all(tgt.std(1) > 0) and np.max(np.abs(tgt)) < 1e3)
            for st in (st_t, st_h):                           #: realised nominal share within tolerance
                if "global_share_broadband" in st:
                    ok &= abs(st["global_share_broadband"] - GLOBAL_SHARE) <= SHARE_TOL
            for n, a in (("target", tgt), ("heldout", ho)):
                p = f"truth/{tag}_{c}_{n}.npy"; np.save(OUT + p, a); files.append(dict(file=p, sha=sha(OUT + p)))
            F = np.corrcoef(Lib.bandpass(tgt))[~np.eye(100, dtype=bool)]
            desc[c] = dict(mean_fc_F=float(F.mean()), neg_fc_F=float(np.mean(F < 0)), target_stats=st_t, heldout_stats=st_h)
        log.append(dict(seed=s2, valid=ok))
        if ok:
            return dict(tag=tag, family=fam, seed=s, used_seed=s2, stratum=k, files=files, targets=desc, attempts=log)
    return dict(tag=f"{fam}_{s}", family=fam, seed=s, missing=True, attempts=log, files=[])


def stage_prepare(procs):
    import scipy.io as sio
    os.makedirs(OUT + "truth", exist_ok=True)
    ts = sio.loadmat(DEMO + "gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel()
    emp = [mean_fc(ts[k].astype(float)) for k in range(3)]
    with mp.Pool(3) as p:
        cal = p.starmap(calibrate_gamma, [(k, emp[k]) for k in range(3)])
    gamma_k = [c["gamma"] for c in cal]
    jobs = [("SCG", s, gamma_k) for s in SCG_SEEDS] + [("QM1", s, gamma_k) for s in QM1_SEEDS]
    with mp.Pool(procs) as p:
        res = p.map(truth_job, jobs, chunksize=1)
    files = [f for r in res for f in r["files"]]
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), emp_mean_fc=emp, calibration=cal, truths=res, files=files,
                   code={c: sha(c) for c in CODE if os.path.exists(c)}, namespaces=dict(system="11e6+100s+cell (+5e5 held-out)",
                   obs="default_rng([s, cell, 1|2])", hetero="default_rng([s, 4])", calib="12e6+10k+j")),
              open(OUT + "r4e_e7_manifest.json", "w"), indent=1, default=float)
    for c in cal:
        print("stratum", c["stratum"], "target mean FC %.3f -> gamma %.3f mean FC %.3f success %s" % (c["target"], c["gamma"], c["mean_fc"], c["success"]), flush=True)
    print("truths", len(res), "missing", sum(1 for r in res if r.get("missing")), "replaced", sum(1 for r in res if r.get("used_seed", r["seed"]) != r["seed"]), flush=True)
    for cell in CELLS:
        v = [r["targets"][cell]["mean_fc_F"] for r in res if not r.get("missing") and cell in r["targets"]]
        print(cell, "target mean FC (F) %.3f [%.3f, %.3f]" % (np.median(v), np.min(v), np.max(v)), flush=True)


# ---------------------------------------------------------------- jobs, freeze, run (same contract as luppi_r4e_ext)
def job_list():
    J = []
    for fam, seeds in (("SCG", SCG_SEEDS), ("QM1", QM1_SEEDS)):
        for s in seeds:
            for c in CELLS:
                if fam == "QM1" and c == "G0":
                    continue
                for mdl in ("signed", "coop"):
                    J.append(dict(exp="E7", family=fam, tag=f"{fam}_{s}", stratum=(s - 1) % 3, cell=c, branch="F", seed=11, model=mdl))
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def truth_path(j, n):
    return (R4E if j["family"] == "QM1" else OUT) + f"truth/{j['tag']}_{n}.npy"


def job_files(j):
    return [truth_path(j, "C"), truth_path(j, "f"), OUT + f"truth/{j['tag']}_{j['cell']}_target.npy", OUT + f"truth/{j['tag']}_{j['cell']}_heldout.npy"]


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if "fit" not in _SIMS:
        _SIMS["fit"] = L.make_simulator(100, T_KEEP, 3000.0)
    sim = _SIMS["fit"]; t0 = time.time(); cfg = dict(L.HEADER)
    C_true = np.load(truth_path(j, "C")); _, SC, _ = Lib.demo_subject(j["stratum"])
    tgt = np.load(OUT + f"truth/{j['tag']}_{j['cell']}_target.npy"); ho = np.load(OUT + f"truth/{j['tag']}_{j['cell']}_heldout.npy")
    inp = Lib.bandpass(tgt); f = L.regional_frequencies([inp]); hop = Lib.branch(ho, "F")
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    base = 13_000_000 + 1000 * int(j["tag"].split("_")[1]) + 10 * CELLS.index(j["cell"]); sims = []
    for r in range(8):                                    # estimator's own model, paired keys for signed / coop
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm); met = Lib.edge_metrics(C, C_true, SC)
    if not (np.all(np.isfinite(C)) and np.isfinite(mse) and np.isfinite(met["frac_neg"])):
        raise ValueError("non-finite fitted matrix or endpoint input")
    np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"],
                mse_ho=mse, metrics=met, C_sha=sha_arr(C))


FREEZE = OUT + "r4e_e7_freeze.json"
PREREG = OUT + "r4e_e7_analysis_prereg.json"


def eligible_tags():
    """Eligibility from the manifest's missingness record, not from leftover files."""
    return {t["tag"] for t in json.load(open(OUT + "r4e_e7_manifest.json"))["truths"] if not t.get("missing")}


def stage_freeze():
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    man = {OUT + r["file"]: r["sha"] for r in json.load(open(OUT + "r4e_e7_manifest.json"))["files"]}
    r4e = json.load(open(R4E + "r4e_manifest.json"))
    for t in r4e["truths"]:
        if not t.get("missing"):
            for n, h in t["sha"].items():
                man[R4E + f"truth/{t['tag']}_{n}.npy"] = h
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("luppi_r4e_e7_analyze.py"):
        raise SystemExit("analysis pre-registration missing or not matching luppi_r4e_e7_analyze.py")
    J = job_list(); files = {}; ok_tags = eligible_tags()
    for j in J:
        if j["tag"] not in ok_tags:
            continue
        for p in job_files(j):
            if p not in man or sha(p) != man[p]:
                raise SystemExit(f"{p} missing from manifests or changed")
            files[p] = man[p]
    fr = dict(group="e7", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE}, raw={p: sha(p) for p in RAW},
              manifest=sha(OUT + "r4e_e7_manifest.json"), prereg=sha(PREREG), eligible=sorted(ok_tags), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen e7", sha(FREEZE), "jobs", len(J), "files", len(files), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(OUT + "r4e_e7_manifest.json", fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="e7", inputs={p: fr["files"][p] for p in job_files(j)})


def run_job_safe(args):
    j, prov, deadline = args
    import luppi_hopf as L
    def w(rec):
        json.dump(rec, open(j["file"], "w"), indent=1, default=float); return rec["state"]
    if os.path.exists(j["file"]):
        old = json.load(open(j["file"]))
        if old.get("state") in ("done", "failed_twice", "cancelled") and old.get("prov") == prov and old.get("key") == j["key"]:
            return old["state"]
    errors = []
    for attempt in (1, 2):
        if time.time() > deadline:
            return w(dict(j, state="cancelled", prov=prov, reason="deadline before start" if attempt == 1 else "deadline before retry", attempts=attempt - 1, errors=errors))
        try:
            r = run_job(j, deadline)
            if time.time() > deadline:
                return w(dict(j, state="cancelled", prov=prov, reason="deadline passed before acceptance", attempts=attempt, errors=errors))
            r.update(state="done", prov=prov, attempts=attempt, errors=errors); return w(r)
        except L.Cancelled as e:
            return w(dict(j, state="cancelled", prov=prov, reason=str(e), attempts=attempt, errors=errors))
        except Exception:
            errors.append(traceback.format_exc())
    return w(dict(j, state="failed_twice", prov=prov, attempts=2, errors=errors))


def stage_run(procs):
    fsha, fr = verify_freeze(); os.makedirs(OUT + "jobs", exist_ok=True)
    dl_file = OUT + "r4e_e7_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    J = [j for j in job_list() if j["tag"] in set(fr["eligible"])]
    J.sort(key=lambda j: (j["cell"], j["tag"]))
    print("jobs", len(J), "of", fr["n_jobs"], flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"prepare": lambda: stage_prepare(a.procs), "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
