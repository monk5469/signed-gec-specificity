"""Simulation 6 runner (Section 2.8.3): extension of the original L1 ground-truth test (Luppi et al. 2026, Extended Data Fig. 7),
with cooperative-only acc fits of 30 participants as truths, observed with and without shared sources and removal, and the
CMPH positive control."""
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; OUT = P33 + "r1/"; ACC = P33 + "r4e_ppc_acc/"; SMOKE = P33 + "r1_smoke.json"
SPEC = DATA_ROOT + "docs_r1/164_r1_spec_v0.2.md"
RULES = OUT + "r1_rules.json"; PANEL = OUT + "r1_panel.json"; MAN = OUT + "r1_manifest.json"; FREEZE = OUT + "r1_freeze.json"
SIGMA_GEN = dict(s01=0.01, s001=0.001); N_TRUTH = 30; STAB_N = 8; LAMBDA = 0.3; NVOX = 500; NPC = 4
ARMS_MAIN = ("A0", "A1", "A2", "Pplus", "A0x", "A4", "A3cal", "A3cal_svd", "A3ind"); CORE = ("A0", "A1", "A2", "Pplus")
STAB_COMBOS = ((0, 12), (1, 11), (1, 12), (2, 11), (2, 12)); BUDGET_H = 24.0
CODE = ["r1_run.py", "r1_analyze.py", "luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e10c.py", "luppi_r4e_e11.py",
        "luppi_r4e_ppc.py", "r5_groupfc.py", "e14_analyze.py", "e14b_analyze.py", "luppi_r4e_e11_analyze.py"]
RAW = [SMOKE, P33 + "e10c_targets.json", P33 + "e10c_sources.npz", P33 + "e10c_sources.npz.json", ACC + "e5/SC.npy", ACC + "e5/freq_REST1.npy"]
FX = dict(dir=P33 + "r1_fixture/", n=2, iters=120)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_rules():
    r = json.load(open(RULES))
    now = dict(r1_run=sha("r1_run.py"), r1_analyze=sha("r1_analyze.py"), spec=sha(SPEC))
    if r["hashes"] != now:
        raise SystemExit(f"rules changed since registration: {[k for k in now if now[k] != r['hashes'][k]]}")


# ---------------------------------------------------------------- truths, generation, arms
def group_sc():
    return np.load(ACC + "e5/SC.npy")


def cmph(C, SC, k):
    import luppi_r4e_lib as Lib
    pos = Lib.support(SC) & (C > 0); g = Lib.fiedler(SC)
    pool = np.flatnonzero((pos & (np.sign(g)[:, None] * np.sign(g)[None, :] < 0)).ravel()); n = int(round(0.8 * pool.size))
    pick = np.random.default_rng([6000 + k, 8]).choice(pool, n, replace=False); Cf = C.ravel().copy(); Cf[pick] = -Cf[pick]
    return Cf.reshape(C.shape), dict(pool=int(pool.size), n_neg=int(n))


def neural(C, f, k, g, held, sg):
    import luppi_hopf as L, luppi_r4e_ext as X, luppi_r4e_e8 as E8
    key = (41_500_000 if held else 41_000_000) + 1000 * k + g; kd = (43_500_000 if held else 43_000_000) + 1000 * k + g
    return X.simulate(E8.msim(1), C, f, key, np.zeros((100, 1)), 0.0, kd, dict(L.HEADER, sigma=SIGMA_GEN[sg]))


def pools(x, s_obs, held, c, noise_rel):
    """Tissue pools (T x NVOX each) sharing the observation's source draws D (same rng as luppi_r4e_e11.observe)."""
    import luppi_r4e_ext as X, luppi_r4e_e10c as E10
    h = 1 if held else 0; T = x.shape[1]; B = E10.sources(); K = B.shape[1]; sd = float(np.median(x.std(1)))
    D = E10.ou(np.random.default_rng([s_obs, 0, h, 4]), T, X.TAU, K); gx = x.mean(0); gx = (gx - gx.mean()) / gx.std() * sd
    rng = np.random.default_rng([s_obs, 31, h]); P = []; bbar = B.mean(0)
    for p in range(2):
        W = rng.uniform(0.0, 1.0, (NVOX, K)) * bbar[None, :] * K / bbar.sum()     # voxel-specific mixtures of all K sources
        a = rng.uniform(0.5, 1.5, NVOX); S = (LAMBDA * gx)[:, None] * a[None, :] + c * sd * (D.T @ W.T)
        P.append(S + noise_rel * S.std() * rng.standard_normal(S.shape))
    return P


def pool_regressors(P, mode):
    out = []
    for Z in P:
        if mode == "svd5":
            U, _, _ = np.linalg.svd(Z, full_matrices=False); out.append(U[:, :5])
        else:
            Z = Z - Z.mean(0); m = Z.mean(1, keepdims=True); R = Z - m @ np.linalg.lstsq(m, Z, rcond=None)[0]
            U, _, _ = np.linalg.svd(R, full_matrices=False); out.append(np.c_[m, U[:, :NPC]])
    return np.concatenate(out, 1)


def regress(y, Xr):
    T = y.shape[1]; M = np.c_[np.ones(T), np.arange(T, dtype=float), Xr]
    return (y.T - M @ np.linalg.lstsq(M, y.T, rcond=None)[0]).T


def gen_unit(args):
    """One generation unit (k, sg, g): calibration on the COOP target, then target / held-out series of every arm needed."""
    k, sg, g, arms, out, C, f = args
    import luppi_r4e_e10c as E10, luppi_r4e_e11 as E11
    SC = group_sc(); t = E10.targets(); s_obs = 6000 + 100 * g + k; info = dict(k=k, sg=sg, g=g); files = {}
    try:
        Ch, meta = cmph(C, SC, k); info["cmph"] = meta
        xs = {(tt, h): neural(Ct, f, k, g, h, sg) for tt, Ct in (("COOP", C), ("CMPH", Ch)) for h in (False, True)}
        xc = xs[("COOP", False)]
        m, v, _ = E10._bisect(lambda m: E10.gshare(E10.remove(E11.observe(xc, s_obs, False, 10 ** m)[0], None, None)), *E11.RANGE_C, t["raw_gs"], True, E11.STEPS)
        c = 10 ** m; y, xi = E11.observe(xc, s_obs, False, c)
        ms, vs, _ = E10._bisect(lambda mm: E10.gshare(E10.remove(y, 10 ** mm, xi)), -3.0, 3.0, t["d136"], True, 25); sig = 10 ** ms
        info.update(c=c, sigma=sig, raw_gs=v, d136_gs=vs)
        ok = abs(v - t["raw_gs"]) <= 0.01 and abs(vs - t["d136"]) <= 0.01
        if any(a.startswith("A3cal") for a in arms):
            for mode in ("mean4", "svd5"):
                f3 = lambda mm: E10.gshare(regress(y, pool_regressors(pools(xc, s_obs, False, c, 10 ** mm), mode)))
                m3, v3, _ = E10._bisect(f3, -3.0, 3.0, t["d136"], True, 25); info[f"a3_{mode}"] = dict(noise_rel=10 ** m3, gs=v3)
                ok = ok and abs(v3 - t["d136"]) <= 0.01
        info["cal_ok"] = bool(ok)
        if not ok:
            return dict(info, valid=False, files={})
        for held, nm in ((False, "target"), (True, "heldout")):
            xcoop = xs[("COOP", held)]; y0, xi0 = E11.observe(xcoop, s_obs, held, 0.0); y49, xi49 = E11.observe(xcoop, s_obs, held, c)
            series = {}
            for a in arms:
                if a == "A0":
                    series[a] = E10.remove(y0, None, xi0)
                elif a == "A0x":
                    series[a] = E10.remove(xcoop, None, None)
                elif a == "A1":
                    series[a] = E10.remove(y49, None, xi49)
                elif a == "A2":
                    series[a] = E10.remove(y49, sig, xi49)
                elif a == "A4":
                    series[a] = E10.remove(y0, 0.0, xi0)
                elif a == "Pplus":
                    yh, xih = E11.observe(xs[("CMPH", held)], s_obs, held, c, ref=xcoop); series[a] = E10.remove(yh, sig, xih)
                elif a in ("A3cal", "A3cal_svd", "A3ind"):
                    mode = "svd5" if a == "A3cal_svd" else "mean4"; nr = 1.0 if a == "A3ind" else info[f"a3_{mode}"]["noise_rel"]
                    series[a] = regress(y49, pool_regressors(pools(xcoop, s_obs, held, c, nr), mode))
            for a, r in series.items():
                if not (np.all(np.isfinite(r)) and np.all(r.std(1) > 0)):
                    return dict(info, valid=False, files={}, reason=f"non-finite {a} {nm}")
                p = out + f"inp/k{k}_{sg}_g{g}_{a}_{nm}.npy"; np.save(p, r); files[p] = sha(p)
                if nm == "target":
                    info.setdefault("input", {})[a] = E10.desc(r)
        for tt, Ct in (("COOP", C), ("CMPH", Ch)):
            p = out + f"truth/k{k}_{tt}_C.npy"
            if not os.path.exists(p):
                np.save(p, Ct)
            files[p] = sha(p)
        return dict(info, valid=True, files=files)
    except Exception:
        return dict(info, valid=False, files={}, reason=traceback.format_exc()[-600:])


def truths(fixture=False):
    """(tags, {k: (tag, C, idx)}) from admitted PPC acc cooperative fits; fixture: 2 participants NOT in the formal panel."""
    import r5_groupfc as R5
    X, pairs, rej, fsha = R5.admitted("acc"); by = {t: (idx, d) for t, idx, d in pairs}
    picked = json.load(open(SMOKE))["picked"]
    tags = [t for t in sorted(by) if t not in picked][:FX["n"]] if fixture else picked
    if not fixture and len(tags) != N_TRUTH:
        raise SystemExit("panel size differs from 30")
    return {k + 1: (t, by[t][1]["coop"][2], by[t][0]) for k, t in enumerate(tags)}, fsha


def units():
    stab = sorted((np.random.default_rng(20261021).choice(N_TRUTH, STAB_N, replace=False) + 1).tolist()); U = []
    for k in range(1, N_TRUTH + 1):
        U.append((k, "s01", 0, ARMS_MAIN)); U.append((k, "s001", 0, CORE))
    for k in stab:
        for g in (1, 2):
            U.append((k, "s01", g, CORE))
    return U, stab


def stage_prepare(procs, fixture=False):
    out = FX["dir"] if fixture else OUT
    if not fixture:
        check_rules()
        if os.path.exists(PANEL):
            raise SystemExit("panel already prepared")
    os.makedirs(out + "inp", exist_ok=True); os.makedirs(out + "truth", exist_ok=True)
    T, fsha = truths(fixture); f = np.load(ACC + "e5/freq_REST1.npy")
    U, stab = units()
    if fixture:
        U = [(k, "s01", 0, ARMS_MAIN) for k in T]
    with mp.Pool(procs) as p:
        res = p.map(gen_unit, [(k, sg, g, arms, out, T[k][1], f) for k, sg, g, arms in U], chunksize=1)
    files = {}
    for r in res:
        files.update(r.get("files", {}))
    valid = [[r["k"], r["sg"], r["g"]] for r in res if r.get("valid")]
    panel = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), truths={str(k): dict(tag=v[0], subject_index=v[2], C_sha=sha_arr(v[1])) for k, v in T.items()},
                 acc_freeze=fsha, stability_truths=stab, units=[[k, sg, g, list(a)] for k, sg, g, a in U], valid_units=valid,
                 invalid=[dict(k=r["k"], sg=r["sg"], g=r["g"], reason=r.get("reason"), cal_ok=r.get("cal_ok")) for r in res if not r.get("valid")])
    json.dump(panel, open((out + "r1_panel.json") if fixture else PANEL, "w" if fixture else "x"), indent=1)
    json.dump(dict(created=panel["created"], files=files, calibration=res, code={c: sha(c) for c in CODE if os.path.exists(c)}, raw={p: sha(p) for p in RAW}),
              open(out + "r1_manifest.json", "w"), indent=1, default=float)
    print("units", len(U), "valid", len(valid), "| invalid", panel["invalid"][:3], "| files", len(files), flush=True)
    return panel


def job_list(panel_path=PANEL, out=OUT):
    P = json.load(open(panel_path)); valid = {tuple(v) for v in P["valid_units"]}; J = []
    for k, sg, g, arms in P["units"]:
        if (k, sg, g) not in valid:
            continue
        seeds = [11] if g == 0 else []
        if sg == "s01" and k in P["stability_truths"]:
            seeds = seeds + [s for gg, s in STAB_COMBOS if gg == g and s not in seeds]
        for seed in seeds:
            for a in (arms if (g == 0 and seed == 11) else CORE):
                for m in ("signed", "coop"):
                    J.append(dict(exp="R1", k=k, sg=sg, g=g, seed=seed, arm=a, model=m))
    for j in J:
        j["key"] = "|".join(f"{q}={j[q]}" for q in sorted(j)); j["file"] = out + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
        j["tier"] = 0 if (j["sg"] == "s01" and j["g"] == 0 and j["seed"] == 11 and j["arm"] in CORE) else 1 if j["sg"] == "s001" else 2 if j["arm"] in CORE else 3
    return J


def job_files(j, out=OUT):
    b = out + f"inp/k{j['k']}_{j['sg']}_g{j['g']}_{j['arm']}_"
    return [b + "target.npy", b + "heldout.npy"]


_SIM = {}


def run_job(j, stop_at, out=OUT, iters=None):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if "fit" not in _SIM:
        _SIM["fit"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["fit"]; t0 = time.time(); cfg = dict(L.RELEASED)
    if iters:
        cfg["max_iterations"] = iters
    fs = job_files(j, out); SC = group_sc()
    inp = Lib.bandpass(np.load(fs[0])); hop = Lib.branch(np.load(fs[1]), "F"); f = L.regional_frequencies([inp])
    base = 42_000_000 + 1000 * j["k"] + 10 * j["g"]
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm)
    if not (C.shape == (100, 100) and np.all(np.isfinite(C)) and np.isfinite(mse)):
        raise ValueError("non-finite or mis-shaped fitted matrix or held-out MSE")
    np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"], mse_ho=mse,
                neg=L.neg_stats(C, SC), C_sha=sha_arr(C))


def stage_freeze():
    check_rules()
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    mf = json.load(open(MAN)); J = job_list(); files = {}
    for p in [q for j in J for q in job_files(j)] + [OUT + f"truth/k{k}_{tt}_C.npy" for k in range(1, N_TRUTH + 1) for tt in ("COOP", "CMPH")]:
        if p not in mf["files"] or sha(p) != mf["files"][p]:
            raise SystemExit(f"{p} missing from the manifest or changed")
        files[p] = mf["files"][p]
    fr = dict(group="r1", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE}, raw={p: sha(p) for p in RAW},
              rules=sha(RULES), panel=sha(PANEL), manifest=sha(MAN), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen r1", sha(FREEZE), "jobs", len(J), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(RULES, fr["rules"]), (PANEL, fr["panel"]), (MAN, fr["manifest"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="r1", inputs={p: fr["files"][p] for p in job_files(j)})


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
            return w(dict(j, state="cancelled", prov=prov, reason="budget exhausted before start" if attempt == 1 else "budget exhausted before retry", attempts=attempt - 1, errors=errors))
        try:
            r = run_job(j, deadline)
            r.update(state="done", prov=prov, attempts=attempt, errors=errors); return w(r)
        except L.Cancelled as e:
            return w(dict(j, state="cancelled", prov=prov, reason=str(e), attempts=attempt, errors=errors))
        except Exception:
            errors.append(traceback.format_exc()[-1500:])
    return w(dict(j, state="failed_twice", prov=prov, attempts=2, errors=errors))


def stage_run(procs):
    check_rules(); fsha, fr = verify_freeze(); os.makedirs(OUT + "jobs", exist_ok=True); dl_file = OUT + "r1_budget.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + BUDGET_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + BUDGET_H * 3600 + 1:
        raise SystemExit("budget record does not belong to this freeze")
    J = job_list()
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    J.sort(key=lambda j: (j["tier"], j["k"], j["arm"], j["model"])); print("jobs", len(J), {t: sum(j["tier"] == t for j in J) for t in range(4)}, flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


def _fx_job(j):
    try:
        r = run_job(j, time.time() + 3 * 3600, out=FX["dir"], iters=FX["iters"]); return (j["k"], j["arm"], j["model"], "ok", r["iterations"], round(r["secs"], 1))
    except Exception:
        return (j["k"], j["arm"], j["model"], "error", traceback.format_exc()[-400:], None)


def stage_fixture(procs):
    """2 non-panel participants, all main arms; prints calibration validity, input descriptors and that every fit path runs (short fits)."""
    P = stage_prepare(min(procs, 4), fixture=True)
    man = json.load(open(FX["dir"] + "r1_manifest.json"))
    for r in man["calibration"]:
        print("unit", r["k"], r["sg"], r["g"], "valid", r.get("valid"), "| cal", {q: (round(r[q], 4) if isinstance(r.get(q), float) else r.get(q)) for q in ("c", "sigma", "raw_gs", "d136_gs")},
              "| A3", {q: r.get(q) for q in ("a3_mean4", "a3_svd5")}, "| input nf", {a: round(v["neg_fc"], 3) for a, v in r.get("input", {}).items()}, "| reason", (r.get("reason") or "")[-300:], flush=True)
    os.makedirs(FX["dir"] + "jobs", exist_ok=True); J = job_list(FX["dir"] + "r1_panel.json", FX["dir"])
    J = [j for j in J if j["k"] == 1]
    if not J:
        raise SystemExit("fixture: no valid unit, no fit tested")
    with mp.Pool(min(procs, len(J))) as p:
        for r in p.map(_fx_job, J, chunksize=1):
            print("fit", r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.stage == "rules":
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), hashes=dict(r1_run=sha("r1_run.py"), r1_analyze=sha("r1_analyze.py"), spec=sha(SPEC))), open(RULES, "x"), indent=1)
        print("rules", json.load(open(RULES))["hashes"])
    else:
        {"fixture": lambda: stage_fixture(a.procs), "prepare": lambda: stage_prepare(min(a.procs, 30)), "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
