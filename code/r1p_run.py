"""Simulation 6b runner (Section 2.8.4): post hoc follow-up testing frequency mismatch, finite-sample noise (targets four times
longer) and early stopping, on the Simulation 6 truths with new random streams."""
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT
import r1_run as R

P33 = R.P33; OUT = P33 + "r1p/"; ACC = R.ACC; SPEC = DATA_ROOT + "docs_r1/168_r1p_spec_v0.2.md"
RULES = OUT + "r1p_rules.json"; PANEL = OUT + "r1p_panel.json"; MAN = OUT + "r1p_manifest.json"; FREEZE = OUT + "r1p_freeze.json"
T_SHORT = 1189; T_LONG = 4 * 1189; G = 3; SEED = 13; ARMS = ("A0", "A1", "A2", "Pplus"); BUDGET_H = 24.0
VARIANTS = dict(ftrue=("short", "gen"), fest=("short", "est"), ftrue_long=("long", "gen"), ftrue_full=("short", "gen"))
FULL_ARMS = ("A0", "A2", "Pplus")   # V-ftrue-full: modified-estimator sensitivity (no early stop), signed only
CODE = ["r1p_run.py", "r1p_analyze.py", "r1_run.py", "luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e10c.py",
        "luppi_r4e_e11.py", "luppi_r4e_ppc.py", "r5_groupfc.py", "e14_analyze.py", "e14b_analyze.py", "luppi_r4e_e11_analyze.py"]
RAW = R.RAW + [R.PANEL]
FX = dict(dir=P33 + "r1p_fixture/", n=2, iters=120)
sha = R.sha; sha_arr = R.sha_arr


def check_rules():
    r = json.load(open(RULES)); now = dict(r1p_run=sha("r1p_run.py"), r1p_analyze=sha("r1p_analyze.py"), spec=sha(SPEC))
    if r["hashes"] != now:
        raise SystemExit(f"rules changed since registration: {[k for k in now if now[k] != r['hashes'][k]]}")


_GEN = {}


def gensim(T):
    import luppi_r4e_ext as X
    if T not in _GEN:
        _GEN[T] = X.make_multi_simulator(100, T, 3000.0, 1)
    return _GEN[T]


def neural(C, f, k, held, T):
    import luppi_hopf as L, luppi_r4e_ext as X
    key = (41_500_000 if held else 41_000_000) + 1000 * k + G; kd = (43_500_000 if held else 43_000_000) + 1000 * k + G
    return X.simulate(gensim(T), C, f, key, np.zeros((100, 1)), 0.0, kd, dict(L.HEADER, sigma=0.01))


def gen_unit(args):
    k, length, out, C, f = args
    import luppi_r4e_e10c as E10, luppi_r4e_e11 as E11
    SC = R.group_sc(); t = E10.targets(); s_obs = 6300 + k; T = T_SHORT if length == "short" else T_LONG; info = dict(k=k, length=length); files = {}
    try:
        Ch, meta = R.cmph(C, SC, k); info["cmph"] = meta
        xs = {(tt, h): neural(Ct, f, k, h, T) for tt, Ct in (("COOP", C), ("CMPH", Ch)) for h in (False, True)}
        xc = xs[("COOP", False)]
        m, v, _ = E10._bisect(lambda m: E10.gshare(E10.remove(E11.observe(xc, s_obs, False, 10 ** m)[0], None, None)), *E11.RANGE_C, t["raw_gs"], True, E11.STEPS)
        c = 10 ** m; y, xi = E11.observe(xc, s_obs, False, c)
        ms, vs, _ = E10._bisect(lambda mm: E10.gshare(E10.remove(y, 10 ** mm, xi)), -3.0, 3.0, t["d136"], True, 25); sig = 10 ** ms
        ok = abs(v - t["raw_gs"]) <= 0.01 and abs(vs - t["d136"]) <= 0.01; info.update(c=c, sigma=sig, raw_gs=v, d136_gs=vs, cal_ok=bool(ok))
        if not ok:
            return dict(info, valid=False, files={})
        for held, nm in ((False, "target"), (True, "heldout")):
            xcoop = xs[("COOP", held)]; y0, xi0 = E11.observe(xcoop, s_obs, held, 0.0); y49, xi49 = E11.observe(xcoop, s_obs, held, c)
            yh, xih = E11.observe(xs[("CMPH", held)], s_obs, held, c, ref=xcoop)
            series = dict(A0=E10.remove(y0, None, xi0), A1=E10.remove(y49, None, xi49), A2=E10.remove(y49, sig, xi49), Pplus=E10.remove(yh, sig, xih))
            for a, r in series.items():
                if not (np.all(np.isfinite(r)) and np.all(r.std(1) > 0)):
                    return dict(info, valid=False, files={}, reason=f"non-finite {a} {nm}")
                p = out + f"inp/k{k}_{length}_{a}_{nm}.npy"; np.save(p, r); files[p] = sha(p)
                if nm == "target":
                    info.setdefault("input", {})[a] = E10.desc(r)
        return dict(info, valid=True, files=files)
    except Exception:
        return dict(info, valid=False, files={}, reason=traceback.format_exc()[-600:])


def stage_prepare(procs, fixture=False):
    out = FX["dir"] if fixture else OUT
    if not fixture:
        check_rules()
        if os.path.exists(PANEL):
            raise SystemExit("panel already prepared")
    os.makedirs(out + "inp", exist_ok=True)
    T, fsha = R.truths(fixture); f = np.load(ACC + "e5/freq_REST1.npy"); U = [(k, ln) for k in T for ln in ("short", "long")]
    with mp.Pool(procs) as p:
        res = p.map(gen_unit, [(k, ln, out, T[k][1], f) for k, ln in U], chunksize=1)
    files = {}
    for r in res:
        files.update(r.get("files", {}))
    for k in T:
        for tt, Ct in (("COOP", T[k][1]), ("CMPH", R.cmph(T[k][1], R.group_sc(), k)[0])):
            p = out + f"truth/k{k}_{tt}_C.npy"; os.makedirs(out + "truth", exist_ok=True); np.save(p, Ct); files[p] = sha(p)
    panel = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), truths={str(k): dict(tag=v[0], subject_index=v[2], C_sha=sha_arr(v[1])) for k, v in T.items()},
                 acc_freeze=fsha, units=[[k, ln] for k, ln in U], valid_units=[[r["k"], r["length"]] for r in res if r.get("valid")],
                 invalid=[dict(k=r["k"], length=r["length"], reason=r.get("reason"), cal_ok=r.get("cal_ok")) for r in res if not r.get("valid")])
    json.dump(panel, open(out + "r1p_panel.json", "w" if fixture else "x"), indent=1)
    json.dump(dict(created=panel["created"], files=files, calibration=res, code={c: sha(c) for c in CODE if os.path.exists(c)}, raw={p: sha(p) for p in RAW}),
              open(out + "r1p_manifest.json", "w"), indent=1, default=float)
    print("units", len(U), "valid", len(panel["valid_units"]), "| invalid", panel["invalid"][:3], "| files", len(files), flush=True)
    return panel


def job_list(panel_path=PANEL, out=OUT):
    P = json.load(open(panel_path)); valid = {tuple(v) for v in P["valid_units"]}; J = []
    for kk in sorted(P["truths"], key=int):
        k = int(kk)
        for var, (ln, fq) in VARIANTS.items():
            if (k, ln) not in valid:
                continue
            for a in (FULL_ARMS if var == "ftrue_full" else ARMS):
                for m in (("signed",) if var == "ftrue_full" else ("signed", "coop")):
                    J.append(dict(exp="R1p", k=k, variant=var, arm=a, model=m, seed=SEED))
    for j in J:
        j["key"] = "|".join(f"{q}={j[q]}" for q in sorted(j)); j["file"] = out + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
        j["tier"] = {"ftrue": 0, "fest": 1, "ftrue_long": 2, "ftrue_full": 3}[j["variant"]]
    return J


def job_files(j, out=OUT):
    ln = VARIANTS[j["variant"]][0]; b = out + f"inp/k{j['k']}_{ln}_{j['arm']}_"
    return [b + "target.npy", b + "heldout.npy"]


_SIM = {}


def run_job(j, stop_at, out=OUT, iters=None):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if "fit" not in _SIM:
        _SIM["fit"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["fit"]; t0 = time.time(); cfg = dict(L.RELEASED)
    if j["variant"] == "ftrue_full":
        cfg["conv"] = -1e9                              # never stop early: run max_iterations, return the last iterate (modified estimator)
    if iters:
        cfg["max_iterations"] = iters
    fs = job_files(j, out); SC = R.group_sc(); inp = Lib.bandpass(np.load(fs[0])); hop = Lib.branch(np.load(fs[1]), "F")
    f = np.load(ACC + "e5/freq_REST1.npy") if VARIANTS[j["variant"]][1] == "gen" else L.regional_frequencies([inp])
    base = 42_000_000 + 1000 * j["k"] + 10 * G
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
                neg=L.neg_stats(C, SC), C_sha=sha_arr(C), f_dev=float(np.median(np.abs(f - np.load(ACC + "e5/freq_REST1.npy")) / np.load(ACC + "e5/freq_REST1.npy"))))


def stage_freeze():
    check_rules()
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    mf = json.load(open(MAN)); J = job_list(); files = {}
    P = json.load(open(PANEL))
    for p in [q for j in J for q in job_files(j)] + [OUT + f"truth/k{k}_{tt}_C.npy" for k in P["truths"] for tt in ("COOP", "CMPH")]:
        if p not in mf["files"] or sha(p) != mf["files"][p]:
            raise SystemExit(f"{p} missing from the manifest or changed")
        files[p] = mf["files"][p]
    fr = dict(group="r1p", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE}, raw={p: sha(p) for p in RAW},
              rules=sha(RULES), panel=sha(PANEL), manifest=sha(MAN), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen r1p", sha(FREEZE), "jobs", len(J), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(RULES, fr["rules"]), (PANEL, fr["panel"]), (MAN, fr["manifest"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="r1p", inputs={p: fr["files"][p] for p in job_files(j)})


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
            return w(dict(j, state="cancelled", prov=prov, reason="budget exhausted", attempts=attempt - 1, errors=errors))
        try:
            r = run_job(j, deadline); r.update(state="done", prov=prov, attempts=attempt, errors=errors); return w(r)
        except L.Cancelled as e:
            return w(dict(j, state="cancelled", prov=prov, reason=str(e), attempts=attempt, errors=errors))
        except Exception:
            errors.append(traceback.format_exc()[-1500:])
    return w(dict(j, state="failed_twice", prov=prov, attempts=2, errors=errors))


def stage_run(procs):
    check_rules(); fsha, fr = verify_freeze(); os.makedirs(OUT + "jobs", exist_ok=True); dl_file = OUT + "r1p_budget.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + BUDGET_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha:
        raise SystemExit("budget record does not belong to this freeze")
    J = job_list()
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    J.sort(key=lambda j: (j["tier"], j["k"], j["arm"], j["model"])); print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


def _fx_job(j):
    try:
        r = run_job(j, time.time() + 3 * 3600, out=FX["dir"], iters=FX["iters"]); return (j["k"], j["variant"], j["arm"], j["model"], "ok", r["iterations"], r["secs"], round(r["f_dev"], 4))
    except Exception:
        return (j["k"], j["variant"], j["arm"], j["model"], "error", traceback.format_exc()[-400:])


def stage_fixture(procs):
    stage_prepare(min(procs, 4), fixture=True); man = json.load(open(FX["dir"] + "r1p_manifest.json"))
    for r in man["calibration"]:
        print("unit", r["k"], r["length"], "valid", r.get("valid"), {q: (round(r[q], 4) if isinstance(r.get(q), float) else r.get(q)) for q in ("c", "sigma", "raw_gs", "d136_gs")},
              "| input nf", {a: round(v["neg_fc"], 3) for a, v in r.get("input", {}).items()}, "|", (r.get("reason") or "")[-300:], flush=True)
    os.makedirs(FX["dir"] + "jobs", exist_ok=True); J = [j for j in job_list(FX["dir"] + "r1p_panel.json", FX["dir"]) if j["k"] == 1]
    if not J:
        raise SystemExit("fixture: no valid unit")
    with mp.Pool(min(procs, len(J))) as p:
        for r in p.map(_fx_job, J, chunksize=1):
            print("fit", r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.stage == "rules":
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), hashes=dict(r1p_run=sha("r1p_run.py"), r1p_analyze=sha("r1p_analyze.py"), spec=sha(SPEC))), open(RULES, "x"), indent=1)
        print("rules", json.load(open(RULES))["hashes"])
    else:
        {"fixture": lambda: stage_fixture(a.procs), "prepare": lambda: stage_prepare(min(a.procs, 30)), "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
