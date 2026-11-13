"""Simulation 2 runner (Sections 2.6, 3.4, 3.5): replication of the cooperative counterexample on independent truths, positive
controls, the recovery boundary (CMPN and CMPH) and the RAWX compatibility check."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e_e11/"; P33 = DATA_ROOT + "results/p33/"; E7 = P33 + "r4e_e7/"; E8D = P33 + "r4e_e8/"
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e8_analyze.py", "luppi_r4e_run.py", "luppi_r4e_e7.py",
        "luppi_r4e_e10c.py", "e10c_sources.py", "e10c_targets.py", "luppi_r4e_e11.py", "luppi_r4e_e11_analyze.py", "e11_prereg.py"]
GEN_CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e7.py", "luppi_r4e_e10c.py", "luppi_r4e_e11.py"]
DEMO = DATA_ROOT + "raw/luppi_demo/gec_3/"; TARGETS = P33 + "e10c_targets.json"; SOURCES = P33 + "e10c_sources.npz"; NETS = E8D + "networks.json"
RAW = [E8D + "r4e_e8_freeze.json", E8D + "r4e_e8_manifest.json", NETS, E7 + "r4e_e7_manifest.json", E7 + "r4e_e7_freeze.json", TARGETS, SOURCES, SOURCES + ".json",
       P33 + "ppc_analysis.json", DEMO + "FMRI.mat", DEMO + "SC.mat"]
TTYPES = ("COOP", "CMPN", "CMPH"); SEEDS = tuple(range(5031, 5061)); NEG_SHARE = 0.25; CMPH_POOL_SHARE = 0.8
CONDS = ("N0-raw", "N0-gsr", "N30-raw", "N30-d136", "N49-raw", "N49-d136", "N49-gsr")
RAW_GS = dict(N30=0.30); TOL = dict(raw_gs=0.01, dose_gs=0.01); RANGE_C = (-2.0, 2.5); STEPS = 30
DEADLINE_H = 72.0
FREEZE = OUT + "r4e_e11_freeze.json"; PREREG = OUT + "r4e_e11_analysis_prereg.json"; MAN = OUT + "r4e_e11_manifest.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gammas():
    cal = json.load(open(E7 + "r4e_e7_manifest.json"))["calibration"]
    return {int(c["stratum"]): float(c["gamma"]) for c in cal}


def base_truth(s, root=None):
    """COOP SCG truth (C, f, stratum) from the files written by stage_prepare (root resolved at call time)."""
    root = OUT if root is None else root
    return np.load(root + f"truth/SCG_{s}_C.npy"), np.load(root + f"truth/SCG_{s}_f.npy"), (s - 1) % 3


def truth_c(s, ttype, root=None):
    """CMPN: n = round(0.25 x #positive support entries) drawn from the between-Yeo-network pool (all if short). CMPH: round(0.8 x pool) entries drawn from the Fiedler-opposite pool, so that the pattern varies across truths (directed entries,
 as CMPN)."""
    import luppi_r4e_lib as Lib, luppi_r4e_e10c as E10
    C, f, k = base_truth(s, root)
    if ttype == "COOP":
        return C, f, k, dict(n_neg=0)
    _, SC, _ = Lib.demo_subject(k); pos = Lib.support(SC) & (C > 0)
    if ttype == "CMPN":
        nets = E10.net_info()[0]; pool = np.flatnonzero((pos & (nets[:, None] != nets[None, :])).ravel()); rng = np.random.default_rng([s, 7])
        n = int(round(NEG_SHARE * pos.sum()))
    else:
        g = Lib.fiedler(SC); pool = np.flatnonzero((pos & (np.sign(g)[:, None] * np.sign(g)[None, :] < 0)).ravel()); rng = np.random.default_rng([s, 8])
        n = int(round(CMPH_POOL_SHARE * pool.size))
    short = pool.size < n; pick = pool if short else rng.choice(pool, n, replace=False)
    Cf = C.ravel().copy(); Cf[pick] = -Cf[pick]
    return Cf.reshape(C.shape), f, k, dict(n_neg=int(pick.size), n_target=n, pool=int(pool.size), short=bool(short), share_of_positive=float(pick.size / pos.sum()))


def neural(s, ttype, held, root=None):
    import luppi_hopf as L, luppi_r4e_ext as X, luppi_r4e_e8 as E8
    C, f, _, _ = truth_c(s, ttype, root)
    return X.simulate(E8.msim(1), C, f, 14_000_000 + s + (500_000 if held else 0), np.zeros((100, 1)), 0.0, 26_000_000 + 100 * s + (500_000 if held else 0), L.HEADER)


def observe(x, s, held, c, parts=False, ref=None):
    """y = x + sources + noise. The source amplitude (c * median SD) and the per-region noise scale (in-band SNR 15)
 are computed from the REFERENCE series ref (the paired COOP neural series), so that the nuisance matrices are identical across truth
 types; ref = x reproduces luppi_r4e_e8.add_noise exactly (same draws, same formula)."""
    import luppi_r4e_ext as X, luppi_r4e_lib as Lib, luppi_r4e_e10c as E10
    ref = x if ref is None else ref; h = 1 if held else 0; T = x.shape[1]; B = E10.sources()
    common = (c * float(np.median(ref.std(1)))) * (B @ E10.ou(np.random.default_rng([s, 0, h, 4]), T, X.TAU, B.shape[1])) if c > 0 else np.zeros_like(x)
    e = np.random.default_rng([s, 15, 2 if held else 1]).standard_normal(x.shape)
    scale = np.sqrt(Lib.bandpass(ref + common).var(1) / (E10.SNR * Lib.bandpass(e).var(1))); noise = scale[:, None] * e
    y = x + common + noise; xi = E10.surrogate(y.mean(0), np.random.default_rng([s, 0, h, 3]))
    return (y, xi, (x, common, noise)) if parts else (y, xi)


def cond_split(cond):
    nu, dose = cond.split("-"); return nu, dose


# ---------------------------------------------------------------- targets
def target_job(args):
    """Per-truth calibration on the COOP target realisation (c per nuisance level, sigma per dose), then targets / held-out for all
 three truth types under the SAME c, sigma and random streams; input descriptors, variance fractions and the d042 arm recorded."""
    s, out = args
    import luppi_r4e_e10c as E10
    t = E10.targets(); g042 = E10.gsr_gs_target(); info = dict(tag=f"SCG_{s}", stratum=(s - 1) % 3); files = []
    xs = {(tt, h): neural(s, tt, h, out) for tt in TTYPES for h in (False, True)}
    xc = xs[("COOP", False)]; c = dict(N0=0.0); sig = {}; got = {}
    for nu, tgt in (("N30", RAW_GS["N30"]), ("N49", t["raw_gs"])):
        m, v, _ = E10._bisect(lambda m: E10.gshare(E10.remove(observe(xc, s, False, 10 ** m)[0], None, None)), *RANGE_C, tgt, True, STEPS)
        c[nu] = 10 ** m; got[f"{nu}-raw"] = v
    for nu in ("N30", "N49"):
        y, xi = observe(xc, s, False, c[nu])
        for d, tgt in (("d136", t["d136"]),) + ((("d042", g042),) if nu == "N49" else ()):
            mm, vv, _ = E10._bisect(lambda mm: E10.gshare(E10.remove(y, 10 ** mm, xi)), -3.0, 3.0, tgt, True, 25); sig[f"{nu}-{d}"] = 10 ** mm; got[f"{nu}-{d}"] = vv
    ok = (abs(got["N30-raw"] - RAW_GS["N30"]) <= TOL["raw_gs"] and abs(got["N49-raw"] - t["raw_gs"]) <= TOL["raw_gs"]
          and all(abs(got[k] - (t["d136"] if k.endswith("d136") else g042)) <= TOL["dose_gs"] for k in got if not k.endswith("raw")))
    info.update(c=c, sigma=sig, calibrated_gs=got, truth_cal_ok=bool(ok), meta={tt: truth_c(s, tt, out)[3] for tt in TTYPES})
    if not ok:
        return dict(info, valid=False, files=[])
    info["measures"], info["var_fractions"], info["heldout_measures"], info["input_only"] = {}, {}, {}, {}
    for tt in TTYPES:
        for held, nm in ((False, "target"), (True, "heldout")):
            for nu in ("N0", "N30", "N49"):
                y, xi, comps = observe(xs[(tt, held)], s, held, c[nu], parts=True, ref=xs[("COOP", held)])
                for cond in [q for q in CONDS if q.startswith(nu + "-")]:
                    d = cond_split(cond)[1]; sg = None if d == "raw" else (0.0 if d == "gsr" else sig[cond])
                    r = E10.remove(y, sg, xi)
                    if not (np.all(np.isfinite(r)) and np.all(r.std(1) > 0)):
                        return dict(info, valid=False, files=[], reason=f"non-finite {tt} {cond} {nm}")
                    (info["heldout_measures"] if held else info["measures"]).setdefault(tt, {})[cond] = E10.desc(r)
                    if not held:
                        info["var_fractions"].setdefault(tt, {})[cond] = E10.var_fractions(y, sg, xi, comps)
                    p = f"truth/SCG_{s}_{tt}_{cond}_{nm}.npy"; np.save(out + p, r); files.append(dict(file=p, sha=sha(out + p)))
                if nu == "N49" and not held:
                    info["input_only"][tt] = dict(d042=E10.desc(E10.remove(y, sig["N49-d042"], xi)))
    return dict(info, valid=True, files=files)


def stage_prepare(procs, seeds=SEEDS, out=OUT):
    import luppi_r4e_e8 as E8, luppi_r4e_e7 as E7M, luppi_r4e_e10c as E10
    E8.verify_freeze(); os.makedirs(out + "truth", exist_ok=True); gk = gammas(); base = []
    for s in seeds:                                                                   # estimator-independent COOP truths
        C, SC, f, k = E7M.scg_truth(s, gk)
        for n, a in (("C", C), ("f", f)):
            p = f"truth/SCG_{s}_{n}.npy"; np.save(out + p, a); base.append(dict(file=p, sha=sha(out + p)))
    with mp.Pool(procs) as p:
        tg = p.map(target_job, [(s, out) for s in seeds], chunksize=1)
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seeds=list(seeds), gammas=gk, targets=tg, files=base + [f for t in tg for f in t["files"]],
               code={c: sha(c) for c in CODE if os.path.exists(c)}, generation_code={c: sha(c) for c in GEN_CODE},
               constants=dict(NEG_SHARE=NEG_SHARE, RAW_GS=RAW_GS, TOL=TOL, RANGE_C=RANGE_C, STEPS=STEPS, CONDS=CONDS, targets=E10.targets(),
                              gsr_gs_target=E10.gsr_gs_target(), targets_sha=sha(TARGETS), sources_sha=sha(SOURCES), e7_manifest_sha=sha(E7 + "r4e_e7_manifest.json")))
    json.dump(man, open(out + "r4e_e11_manifest.json", "w"), indent=1, default=float)
    V = [t for t in tg if t["valid"]]
    print("valid truths", len(V), "of", len(tg), "| invalid", [(t["tag"], t.get("reason"), t.get("calibrated_gs")) for t in tg if not t["valid"]], flush=True)
    print("pools", {tt: sorted({(t["meta"][tt]["n_neg"], t["meta"][tt]["pool"], t["meta"][tt]["short"]) for t in V})[:4] for tt in ("CMPN", "CMPH")}, flush=True)
    for tt in TTYPES:
        print(tt, {q: "gs %.3f negFC %.3f neural %.2f" % (np.median([t["measures"][tt][q]["gs"] for t in V]), np.median([t["measures"][tt][q]["neg_fc"] for t in V]),
                                                          np.median([t["var_fractions"][tt][q]["neural"] for t in V])) for q in CONDS},
              "| d042 negFC %.3f" % np.median([t["input_only"][tt]["d042"]["neg_fc"] for t in V]) if V else "", flush=True)


# ---------------------------------------------------------------- jobs / freeze / run
def job_list():
    J = [dict(exp="E11", family="SCG", tag=f"SCG_{s}", stratum=(s - 1) % 3, ttype=tt, cond=q, branch="F", seed=11, model=m)
         for s in SEEDS for tt in TTYPES for q in CONDS for m in ("signed", "coop")]
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    b = OUT + f"truth/{j['tag']}_{j['ttype']}_{j['cond']}"
    return [OUT + f"truth/{j['tag']}_C.npy", OUT + f"truth/{j['tag']}_f.npy", b + "_target.npy", b + "_heldout.npy"]


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
    base = 25_000_000 + 1000 * s; sims = []
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
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("luppi_r4e_e11_analyze.py"):
        raise SystemExit("analysis pre-registration missing or not matching")
    import luppi_r4e_e8 as E8
    E8.verify_freeze(); mf = json.load(open(MAN))
    if tuple(mf["seeds"]) != SEEDS:
        raise SystemExit("manifest is not the production preparation")
    if mf.get("generation_code") != {c: sha(c) for c in GEN_CODE}:
        raise SystemExit("generation code changed since the targets were prepared")
    k = mf["constants"]
    if k.get("targets_sha") != sha(TARGETS) or k.get("sources_sha") != sha(SOURCES) or k.get("e7_manifest_sha") != sha(E7 + "r4e_e7_manifest.json"):
        raise SystemExit("target / source / E7 artefacts changed since preparation")
    if len(mf["targets"]) != len(SEEDS) or not all(t["valid"] for t in mf["targets"]):
        raise SystemExit("targets incomplete or invalid")
    man = {OUT + r["file"]: r["sha"] for r in mf["files"]}; J = job_list(); files = {}
    for j in J:
        for p in job_files(j):
            if p not in man or sha(p) != man[p]:
                raise SystemExit(f"{p} missing from the manifest or changed")
            files[p] = man[p]
    fr = dict(group="e11", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE}, raw={p: sha(p) for p in RAW},
              manifest=sha(MAN), prereg=sha(PREREG), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen e11", sha(FREEZE), "jobs", len(J), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(MAN, fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="e11", inputs={p: fr["files"][p] for p in job_files(j)})


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
    dl_file = OUT + "r4e_e11_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    order = {"N49-d136": 0, "N0-raw": 1}
    J = job_list(); J.sort(key=lambda j: (order.get(j["cond"], 2), j["cond"], j["ttype"], j["tag"]))
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
