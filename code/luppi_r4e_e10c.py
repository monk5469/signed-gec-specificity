"""Simulation 1 runner (Sections 2.6, 3.4): cooperative and CMPN truths observed through shared sources and the surrogate global
operator at the matched removal doses (the dose map)."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e_e10c/"; P33 = DATA_ROOT + "results/p33/"; E7 = P33 + "r4e_e7/"; E8D = P33 + "r4e_e8/"
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e8_analyze.py", "luppi_r4e_run.py",
        "luppi_r4e_e10c.py", "luppi_r4e_e10c_analyze.py", "e10c_targets.py", "e10c_sources.py", "e10c_prereg.py"]
GEN_CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e10c.py"]
DEMO = DATA_ROOT + "raw/luppi_demo/gec_3/"; TARGETS = P33 + "e10c_targets.json"; NETS = E8D + "networks.json"; SOURCES = P33 + "e10c_sources.npz"
RAW = [E8D + "r4e_e8_freeze.json", E8D + "r4e_e8_manifest.json", NETS, E7 + "r4e_e7_manifest.json", E7 + "r4e_e7_freeze.json", TARGETS, SOURCES, SOURCES + ".json",
       P33 + "ppc_analysis.json", P33 + "ppc_experiment_freeze.json", P33 + "ppc_analysis_prereg.json", DEMO + "FMRI.mat", DEMO + "SC.mat"]
TTYPES = ("COOP", "CMPN", "CMPNp"); CELLS = ("Co",); DOSES = ("raw", "d284", "d189", "d136", "gsr")
CELL_DOSES = {"Co": DOSES}                     # Cn dropped before any fit: input realism
SEEDS = tuple(range(5001, 5031)); SNR = 15.0; NEG_SHARE = 0.25
CAL_KEYS = tuple(8_900_001 + i for i in range(10)); TOL = dict(raw_gs=0.05, truth_raw_gs=0.01, dose_gs=0.01)
RANGES = dict(Co=(-2.0, 2.5), Cn=(-4.0, 1.0)); STEPS = 30
DEADLINE_H = 72.0
FREEZE = OUT + "r4e_e10c_freeze.json"; PREREG = OUT + "r4e_e10c_analysis_prereg.json"; MAN = OUT + "r4e_e10c_manifest.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def targets():
    t = json.load(open(TARGETS))
    if t["n"] != 95 or t["source_sha"] != sha(t["source"]):
        raise SystemExit("target artefact does not match its PPC source")
    return t["targets"]


def gsr_gs_target():
    """Real GSR residual parcel-mean global share, from the same auditable artefact."""
    return float(json.load(open(TARGETS))["input_reference"]["gsr"]["gs"])


_NET = {}


def net_info():
    """(labels (100,), indicator matrix Inet (100, 7) in sorted label order, same-network mask on the upper triangle)."""
    if "x" not in _NET:
        nets = np.asarray(json.load(open(NETS))["networks"]); u = sorted(set(nets.tolist()))
        iu = np.triu_indices(100, 1); _NET["x"] = (nets, (nets[:, None] == np.asarray(u)[None, :]).astype(float), (nets[:, None] == nets[None, :])[iu])
    return _NET["x"]


# ---------------------------------------------------------------- truths and generation
def truth_c(s, ttype):
    import luppi_r4e_e8 as E8, luppi_r4e_lib as Lib
    C, f, k = E8.truth(s)
    if ttype == "COOP":
        return C, f, k, dict(n_neg=0)
    _, SC, _ = Lib.demo_subject(k); nets = net_info()[0]
    pos = Lib.support(SC) & (C > 0); pool = np.flatnonzero((pos & (nets[:, None] != nets[None, :])).ravel())
    n = int(round(NEG_SHARE * pos.sum())); short = pool.size < n
    pick = pool if short else np.random.default_rng([s, 7]).choice(pool, n, replace=False)
    Cf = C.ravel().copy(); Cf[pick] = -Cf[pick]
    return Cf.reshape(C.shape), f, k, dict(n_neg=int(pick.size), n_target=n, pool=int(pool.size), short=bool(short))


_SRC = {}


def sources():
    """Nonnegative source loadings B (100, K) from the auditable symmetric-NMF artefact (sha checked against its metadata)."""
    if "B" not in _SRC:
        meta = json.load(open(SOURCES + ".json"))
        if meta["npz_sha"] != sha(SOURCES):
            raise SystemExit("source artefact does not match its metadata")
        z = np.load(SOURCES); B = z["B"]
        if B.shape != (100, int(z["k"])) or np.any(B < 0):
            raise SystemExit("invalid source loadings")
        _SRC["B"] = B
    return _SRC["B"]


def ou(rng, T, tau, n=1):
    import luppi_hopf as L
    phi = np.exp(-L.TR / tau); e = rng.standard_normal((n, T)); d = np.empty((n, T)); d[:, 0] = e[:, 0]
    for t in range(1, T):
        d[:, t] = phi * d[:, t - 1] + np.sqrt(1 - phi ** 2) * e[:, t]
    return d


def neural(s, ttype, cell, held, c, key_s=None):
    """Neural Hopf series x (100, T); Cn: source drives in the x equation (multi-mode simulator, M = K, G = c B, scale 1)."""
    import luppi_hopf as L, luppi_r4e_ext as X, luppi_r4e_e8 as E8
    C, f, _, _ = truth_c(s, ttype); ci = CELLS.index(cell); B = sources()
    ks = (14_000_000 + s + (500_000 if held else 0)) if key_s is None else key_s
    kd = (22_000_000 + 100 * s + 10 * ci + (500_000 if held else 0)) if key_s is None else key_s + 3_000_000 + 10 * ci
    if cell == "Cn":
        return X.simulate(E8.msim(B.shape[1]), C, f, ks, c * B, 1.0, kd, L.HEADER)
    return X.simulate(E8.msim(1), C, f, ks, np.zeros((100, 1)), 0.0, kd, L.HEADER)


def observe(x, s, cell, held, c, seed_tag=None, parts=False):
    """Observed regional signals y (100, T) and the removal surrogate xi (T,); Co: observation-level sources c * sx * B D(t).
 parts=True also returns the additive components (neural x, sources, observation noise)."""
    import luppi_r4e_ext as X, luppi_r4e_e8 as E8
    ci = CELLS.index(cell); h = 1 if held else 0; tag = s if seed_tag is None else seed_tag; T = x.shape[1]
    common = np.zeros_like(x)
    if cell == "Co":
        B = sources(); common = (c * float(np.median(x.std(1)))) * (B @ ou(np.random.default_rng([tag, ci, h, 4]), T, X.TAU, B.shape[1]))
    y = E8.add_noise(x + common, SNR, np.random.default_rng([tag, 15, 2 if held else 1]))
    xi = surrogate(y.mean(0), np.random.default_rng([tag, ci, h, 3]))
    return (y, xi, (x, common, y - x - common)) if parts else (y, xi)


def surrogate(g, rng):
    """Phase-randomised copy of g (amplitude spectrum kept, DC / Nyquist real), scaled to the standard deviation of g."""
    g = g - g.mean(); F = np.fft.rfft(g); ph = rng.uniform(0, 2 * np.pi, F.size); ph[0] = 0.0
    if g.size % 2 == 0:
        ph[-1] = 0.0
    z = np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=g.size); return z * (g.std() / z.std())


def design(y, sigma, xi):
    T = y.shape[1]; t = np.arange(T, dtype=float); M = np.c_[np.ones(T), t]
    return M if sigma is None else np.c_[M, y.mean(0) + sigma * xi]


def remove(y, sigma, xi, A=None):
    """Unfiltered residual of y on [1, t] (sigma None = raw) or [1, t, g + sigma * xi] (sigma 0 = GSR). With A, the SAME
 regression operator (design built from y) is applied to A instead (component decomposition)."""
    M = design(y, sigma, xi); Z = y if A is None else A
    return (Z.T - M @ np.linalg.lstsq(M, Z.T, rcond=None)[0]).T


def var_fractions(y, sigma, xi, comps):
    """In-band variance fractions (median over regions) of neural / source / noise components after the realised removal operator;
 'cross' = 1 - their sum (cross-covariance terms)."""
    import luppi_r4e_lib as Lib
    vy = Lib.bandpass(remove(y, sigma, xi)).var(1)
    fr = {nm: float(np.median(Lib.bandpass(remove(y, sigma, xi, A)).var(1) / vy)) for nm, A in zip(("neural", "source", "noise"), comps)}
    fr["cross"] = 1.0 - sum(fr.values()); return fr


def desc(r):
    """Band-passed global share, FC mean / SD / negative share and Yeo-7 modularity of a (unfiltered) residual."""
    import luppi_hopf as L, luppi_r4e_lib as Lib
    x = Lib.bandpass(r); F = np.corrcoef(L.prep_emp(x))[np.triu_indices(100, 1)]; same = net_info()[2]
    return dict(gs=float(np.var(x.mean(0)) / np.mean(x.var(1))), mean_fc=float(F.mean()), sd=float(F.std()), neg_fc=float((F < 0).mean()),
                mod=float(F[same].mean() - F[~same].mean()))


def gshare(r):
    import luppi_r4e_lib as Lib
    x = Lib.bandpass(r); return float(np.var(x.mean(0)) / np.mean(x.var(1)))


# ---------------------------------------------------------------- calibration
def _bisect(f, lo, hi, target, increasing, steps):
    tr = []
    for _ in range(steps):
        mid = (lo + hi) / 2; v = f(mid); tr.append((mid, v))
        lo, hi = (mid, hi) if (v < target) == increasing else (lo, mid)
    b = min(tr, key=lambda t: abs(t[1] - target)); return b[0], b[1], tr


def calibrate(args):
    """Per (truth type, cell, stratum): source strength c (STEPS bisection steps, log10 scale) -> mean raw global share = target
 raw_gs; then sigma per dose (25 steps, log10 sigma in [-3, 3]) -> mean band-passed residual global share. Out of sample (recorded,
 never tuned): negative FC, FC SD, mean FC and Yeo-7 modularity per dose."""
    ttype, cell, k = args; t = targets(); s0 = SEEDS[k]; res = dict(ttype=ttype, cell=cell, stratum=k, s0=s0)
    x0 = None if cell == "Cn" else {q: neural(s0, ttype, cell, False, 0.0, key_s=q) for q in CAL_KEYS}
    def data(c):
        return {q: observe(x0[q] if x0 is not None else neural(s0, ttype, cell, False, c, key_s=q), s0, cell, False, c, seed_tag=q) for q in CAL_KEYS}
    m, v, tr = _bisect(lambda m: float(np.mean([gshare(remove(y, None, None)) for y, _ in data(10 ** m).values()])), *RANGES[cell], t["raw_gs"], True, STEPS)
    c = 10 ** m; D = data(c); res.update(c=c, trace=tr)
    res["raw_gs"] = float(np.mean([gshare(remove(y, None, None)) for y, _ in D.values()]))
    res["strength_ok"] = bool(abs(res["raw_gs"] - t["raw_gs"]) <= TOL["raw_gs"])
    res["sigma"] = dict(raw=None, gsr=0.0); res["dose_gs"] = {}; res["dose_ok"] = {}
    for d in CELL_DOSES[cell]:
        if d in ("raw", "gsr"):
            continue
        if t[d] >= res["raw_gs"] - 0.01:
            res["sigma"][d] = None; res["dose_gs"][d] = None; res["dose_ok"][d] = "unreachable"; continue
        g = lambda m: float(np.mean([gshare(remove(y, 10 ** m, xi)) for y, xi in D.values()]))
        m, v, _ = _bisect(g, -3.0, 3.0, t[d], True, 25); res["sigma"][d] = 10 ** m; res["dose_gs"][d] = v; res["dose_ok"][d] = bool(abs(v - t[d]) <= TOL["dose_gs"])
    res["out_of_sample"] = {d: {key: float(np.mean([desc(remove(y, res["sigma"][d], xi))[key] for y, xi in D.values()])) for key in ("gs", "mean_fc", "sd", "neg_fc", "mod")}
                            for d in CELL_DOSES[cell] if d in ("raw", "gsr") or res["sigma"][d] is not None}
    res["success"] = bool(res["strength_ok"] and all(v is True for v in res["dose_ok"].values()))
    return res


def cal_key(ttype, cell, k):
    return f"{ttype}|{cell}|{k}"


# ---------------------------------------------------------------- targets
def target_job(args):
    """Per-truth calibration: on THIS truth's target realisation,
 c (STEPS bisection steps, log10 range RANGES) -> raw global share = target raw_gs and sigma per dose (25 steps) -> residual share
 = dose target; the held-out realisation uses the same c and sigma. The stratum-level calibration is a reported check only."""
    s, ttype, cell, cal, out = args
    _, _, k, meta = truth_c(s, ttype); files = []; info = dict(meta=meta); t = targets()
    xs = {h: neural(s, ttype, cell, h, 0.0) for h in (False, True)}
    # CMPNp: the CMPN truth with the c and sigma calibrated on the paired COOP truth (same seed,
    # same source / noise / surrogate streams); its doses are NOT rematched and are reported as realised.
    xc = xs[False] if ttype != "CMPNp" else neural(s, "COOP", cell, False, 0.0)
    m, v, _ = _bisect(lambda m: gshare(remove(observe(xc, s, cell, False, 10 ** m)[0], None, None)), *RANGES[cell], t["raw_gs"], True, STEPS)
    c = 10 ** m; yc, xic = observe(xc, s, cell, False, c); sig = dict(raw=None, gsr=0.0); got = dict(raw=v)
    for d in CELL_DOSES[cell] + ("d042",):
        if d not in ("raw", "gsr"):
            mm, vv, _ = _bisect(lambda mm: gshare(remove(yc, 10 ** mm, xic)), -3.0, 3.0, gsr_gs_target() if d == "d042" else t[d], True, 25); sig[d] = 10 ** mm; got[d] = vv
    ok = abs(got["raw"] - t["raw_gs"]) <= TOL["truth_raw_gs"] and all(abs(got[d] - t[d]) <= TOL["dose_gs"] for d in got if d not in ("raw", "d042"))
    yt, xit, comps = observe(xs[False], s, cell, False, c, parts=True)
    info.update(c=c, sigma=sig, calibrated_gs=got, realised_gs={d: gshare(remove(yt, sig[d], xit)) for d in sig}, truth_cal_ok=bool(ok),
                calibrated_on="COOP" if ttype == "CMPNp" else ttype)
    if not ok:
        return dict(tag=f"SCG_{s}", ttype=ttype, cell=cell, valid=False, files=[], **info)
    info["input_only"] = dict(d042=desc(remove(yt, sig["d042"], xit)))                    #: matched-GSR-share input arm
    info["var_fractions"] = {d: var_fractions(yt, sig[d], xit, comps) for d in CELL_DOSES[cell]}   # in-band variance fractions per dose
    cal = dict(cal, c=c, sigma=sig)
    for held, nm in ((False, "target"), (True, "heldout")):
        y, xi = (yt, xit) if not held else observe(xs[True], s, cell, True, c)
        if held:
            info["heldout_measures"] = {d: desc(remove(y, sig[d], xi)) for d in CELL_DOSES[cell]}
        for d in CELL_DOSES[cell]:
            if d not in ("raw", "gsr") and cal["sigma"][d] is None:                 # unreachable dose: no file (the freeze requires success)
                info.setdefault("unreachable", []).append(d); continue
            r = remove(y, cal["sigma"][d], xi)
            if not (np.all(np.isfinite(r)) and np.all(r.std(1) > 0)):
                return dict(tag=f"SCG_{s}", ttype=ttype, cell=cell, valid=False, files=[], **info)
            if not held:
                info.setdefault("measures", {})[d] = desc(r)
            p = f"truth/SCG_{s}_{ttype}_{cell}_{d}_{nm}.npy"; np.save(out + p, r); files.append(dict(file=p, sha=sha(out + p)))
    return dict(tag=f"SCG_{s}", ttype=ttype, cell=cell, valid=True, files=files, **info)


def stage_prepare(procs, seeds=SEEDS, out=OUT):
    """Calibration of every (truth type, cell, stratum), then targets / held-out for the given seeds. A smoke run passes a seed
 subset and a temporary output directory; only the production call (all seeds, OUT) writes the manifest used by the freeze."""
    import luppi_r4e_e8 as E8
    E8.verify_freeze(); os.makedirs(out + "truth", exist_ok=True)
    keys = [(tt, c, k) for tt in TTYPES for c in CELLS for k in range(3)]
    with mp.Pool(min(procs, len(keys))) as p:
        cal = {cal_key(*a): r for a, r in zip(keys, p.map(calibrate, keys, chunksize=1))}
    for a in keys:
        r = cal[cal_key(*a)]
        print("cal", cal_key(*a), "c %.4g | raw_gs %.3f" % (r["c"], r["raw_gs"]),
              "| dose_gs", {d: (None if v is None else round(v, 3)) for d, v in r["dose_gs"].items()}, "| ok", r["strength_ok"], r["dose_ok"], r["success"],
              "| negFC", {d: round(v["neg_fc"], 3) for d, v in r["out_of_sample"].items()}, "| sd", {d: round(v["sd"], 3) for d, v in r["out_of_sample"].items()},
              "| mod", {d: round(v["mod"], 3) for d, v in r["out_of_sample"].items()}, flush=True)
    jobs = [(s, tt, c, cal[cal_key(tt, c, (s - 1) % 3)], out) for s in seeds for tt in TTYPES for c in CELLS]
    with mp.Pool(procs) as p:
        tg = p.map(target_job, jobs, chunksize=1)
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seeds=list(seeds), calibration=cal, targets=tg, files=[f for t in tg for f in t["files"]],
               code={c: sha(c) for c in CODE if os.path.exists(c)}, generation_code={c: sha(c) for c in GEN_CODE},
               constants=dict(SNR=SNR, NEG_SHARE=NEG_SHARE, TOL=TOL, CAL_KEYS=CAL_KEYS, RANGES=RANGES, STEPS=STEPS, CELL_DOSES=CELL_DOSES,
                              targets=targets(), targets_sha=sha(TARGETS), sources_sha=sha(SOURCES), K=int(sources().shape[1])))
    json.dump(man, open(out + "r4e_e10c_manifest.json", "w"), indent=1, default=float)
    for tt in TTYPES:
        for c in CELLS:
            M = [t for t in tg if t["ttype"] == tt and t["cell"] == c and t["valid"]]
            print(tt, c, "valid", len(M), {d: "gs %.3f negFC %.3f sd %.3f mod %.3f" % tuple(np.median([t["measures"][d][q] for t in M]) for q in ("gs", "neg_fc", "sd", "mod"))
                                           for d in CELL_DOSES[c] if all(d in t["measures"] for t in M)} if M else None, flush=True)


# ---------------------------------------------------------------- jobs / freeze / run
def job_list():
    J = [dict(exp="E10c", family="SCG", tag=f"SCG_{s}", stratum=(s - 1) % 3, ttype=tt, cell=c, dose=d, branch="F", seed=11, model=m)
         for s in SEEDS for tt in TTYPES for c in CELLS for d in CELL_DOSES[c] for m in ("signed", "coop")]
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    b = OUT + f"truth/{j['tag']}_{j['ttype']}_{j['cell']}_{j['dose']}"
    return [E7 + f"truth/{j['tag']}_C.npy", E7 + f"truth/{j['tag']}_f.npy", b + "_target.npy", b + "_heldout.npy"]


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib, luppi_r4e_e8 as E8
    if "fit" not in E8._SIM:
        E8._SIM["fit"] = L.make_simulator(100, E8.T, 3000.0)
    sim = E8._SIM["fit"]; t0 = time.time(); cfg = dict(L.HEADER); s = int(j["tag"].split("_")[1])
    C_true, _, k, _ = truth_c(s, j["ttype"]); _, SC, _ = Lib.demo_subject(k)
    tgt = np.load(job_files(j)[2]); ho = np.load(job_files(j)[3])
    inp = Lib.bandpass(tgt); f = L.regional_frequencies([inp]); hop = Lib.branch(ho, "F")
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    base = 24_000_000 + 1000 * s; sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm); met = Lib.edge_metrics(C, C_true, SC)
    if not (C.shape == (100, 100) and np.all(np.isfinite(C)) and np.isfinite(mse) and np.isfinite(met["frac_neg"])):
        raise ValueError("non-finite or mis-shaped fitted matrix or endpoint input")
    np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"], mse_ho=mse, metrics=met, C_sha=E8.sha_arr(C))


def stage_freeze():
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("luppi_r4e_e10c_analyze.py"):
        raise SystemExit("analysis pre-registration missing or not matching")
    import luppi_r4e_e8 as E8
    E8.verify_freeze(); mf = json.load(open(MAN))
    if tuple(mf["seeds"]) != SEEDS:
        raise SystemExit("manifest is not the production preparation")
    if len(mf["calibration"]) != len(TTYPES) * len(CELLS) * 3 or not all(v["success"] for v in mf["calibration"].values()):
        raise SystemExit("calibration incomplete or unsuccessful")
    if mf.get("generation_code") != {c: sha(c) for c in GEN_CODE}:
        raise SystemExit("generation code changed since the targets were prepared")
    if mf["constants"].get("targets_sha") != sha(TARGETS) or mf["constants"].get("sources_sha") != sha(SOURCES):
        raise SystemExit("target / source artefacts changed since preparation")
    if len(mf["targets"]) != len(SEEDS) * len(TTYPES) * len(CELLS) or not all(t["valid"] for t in mf["targets"]):
        raise SystemExit("targets incomplete or invalid")
    man = {OUT + r["file"]: r["sha"] for r in mf["files"]}; man.update({E7 + r["file"]: r["sha"] for r in json.load(open(E7 + "r4e_e7_manifest.json"))["files"]})
    J = job_list(); files = {}
    for j in J:
        for p in job_files(j):
            if p not in man or sha(p) != man[p]:
                raise SystemExit(f"{p} missing from manifests or changed")
            files[p] = man[p]
    fr = dict(group="e10c", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE}, raw={p: sha(p) for p in RAW},
              manifest=sha(MAN), prereg=sha(PREREG), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen e10c", sha(FREEZE), "jobs", len(J), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(MAN, fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="e10c", inputs={p: fr["files"][p] for p in job_files(j)})


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
    dl_file = OUT + "r4e_e10c_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    J = job_list(); J.sort(key=lambda j: (j["dose"] != "d136", j["ttype"], j["cell"], j["dose"], j["tag"]))       # acc-matched dose first
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56)
    ap.add_argument("--seeds", default=None); ap.add_argument("--out", default=None); a = ap.parse_args()
    if a.stage == "prepare":
        seeds = SEEDS if a.seeds is None else tuple(int(x) for x in a.seeds.split(",")); out = OUT if a.out is None else a.out.rstrip("/") + "/"
        if (seeds == SEEDS) != (out == OUT):
            raise SystemExit("a seed subset must use a non-production output directory, and vice versa")
        os.makedirs(out, exist_ok=True); stage_prepare(a.procs, seeds, out)
    else:
        os.makedirs(OUT, exist_ok=True); {"freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
