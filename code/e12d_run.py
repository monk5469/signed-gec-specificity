"""Simulation 4 runner (Section 2.5.2): new truths (seeds 5091-5120) for the in-family residual diagnostic; calls the Simulation 1
and 2 functions without modifying them."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/e12d/"; P33 = DATA_ROOT + "results/p33/"; DEMO = DATA_ROOT + "raw/luppi_demo/"
GSIM = P33 + "e12d_gsim.json"; PANEL = OUT + "e12d_panel.json"; MAN = OUT + "e12d_manifest.json"; FREEZE = OUT + "e12d_freeze.json"
PREREG = P33 + "e12d_analyze_prereg.json"; SEEDS = tuple(range(5091, 5121)); MIN_N = 15
CELLS = (("D1", "COOP", "N49-raw"), ("D1", "COOP", "N49-d136"), ("D1", "COOP", "N49-d042"), ("D2", "CMPH", "N0-raw"), ("D2opt", "CMPH", "N49-raw"),
         ("D2opt", "CMPH", "N49-d136"))
COOP_MODELS = {("CMPH", "N0-raw"), ("COOP", "N49-d042")}
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e7.py", "luppi_r4e_e10c.py", "luppi_r4e_e11.py",
        "e12a_validate.py", "e12d_gsim_freeze.py", "e12d_run.py", "e12d_analyze.py"]
RAW = [GSIM, P33 + "e10c_targets.json", P33 + "e10c_sources.npz", P33 + "e10c_sources.npz.json", P33 + "r4e_e7/r4e_e7_manifest.json", DEMO + "single/SC.mat",
       DEMO + "gec_3/SC.mat", DEMO + "gec_3/FMRI.mat"]
DEADLINE_H = 72.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cell_name(tt, cond):
    return f"{tt}_{cond}"


def resid_path(s, tt, cond, nm):
    return OUT + f"truth/SCG_{s}_{tt}_{cond}_{nm}.npy"


def _target(s):
    import luppi_r4e_e11 as E11, luppi_r4e_e10c as E10
    info = E11.target_job((s, OUT))
    if info.get("valid"):
        xs = {h: E11.neural(s, "COOP", h, OUT) for h in (False, True)}; c = info["c"]["N49"]; sg = info["sigma"]["N49-d042"]
        for h, nm in ((False, "target"), (True, "heldout")):
            y, xi = E11.observe(xs[h], s, h, c, ref=xs[h]); r = E10.remove(y, sg, xi); p = resid_path(s, "COOP", "N49-d042", nm); np.save(p, r)
            info["files"].append(dict(file=os.path.relpath(p, OUT).replace("\\", "/"), sha=sha(p)))
    return info


def stage_prepare(procs):
    import scipy.io as sio
    import luppi_r4e_e7 as E7M, luppi_r4e_e11 as E11, luppi_r4e_lib as Lib, luppi_hopf as L, e12a_validate as V
    if os.path.exists(PANEL):
        raise SystemExit("panel already frozen")
    Gs = json.load(open(GSIM))
    if Gs["script_sha"] != sha("e12d_gsim_freeze.py"):
        raise SystemExit("frozen g_sim does not verify")
    lo, hi = Gs["train_range_nf"]
    os.makedirs(OUT + "truth", exist_ok=True); os.makedirs(OUT + "inp", exist_ok=True); gk = E11.gammas(); base = []
    for s in SEEDS:
        C, SC, f, k = E7M.scg_truth(s, gk)
        for n, a in (("C", C), ("f", f)):
            p = OUT + f"truth/SCG_{s}_{n}.npy"; np.save(p, a); base.append(dict(file=f"truth/SCG_{s}_{n}.npy", sha=sha(p)))
    with mp.Pool(procs) as p:
        tg = p.map(_target, SEEDS, chunksize=1)
    valid = {t["tag"]: t for t in tg if t.get("valid")}
    panel = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), gsim_sha=sha(GSIM), min_n=MIN_N, cells={}, invalid_seeds=[t["tag"] for t in tg if not t.get("valid")])
    for role, tt, cond in CELLS:
        mem = {}
        for tag in valid:
            s = int(tag.split("_")[1]); nf, gs = V.inp_measures(Lib.bandpass(np.load(resid_path(s, tt, cond, "target"))))
            mem[tag] = dict(nf=nf, gs=gs, supported=bool(lo <= nf <= hi))
        sup = sorted(t for t, v in mem.items() if v["supported"])
        panel["cells"][cell_name(tt, cond)] = dict(role=role, ttype=tt, cond=cond, eligible=len(sup) >= MIN_N, supported=sup, n_valid=len(mem),
                                                   membership=mem, models=["signed", "coop"] if (tt, cond) in COOP_MODELS else ["signed"])
    json.dump(panel, open(PANEL, "x"), indent=1)
    SCg = sio.loadmat(DEMO + "single/SC.mat")["SC"].astype(float); np.save(OUT + "inp/SC_group.npy", SCg)
    files = base + [dict(file="inp/SC_group.npy", sha=sha(OUT + "inp/SC_group.npy"))] + [f for t in tg for f in t.get("files", [])]
    for name, c in panel["cells"].items():
        if c["eligible"]:
            f = L.regional_frequencies([Lib.bandpass(np.load(resid_path(int(t.split("_")[1]), c["ttype"], c["cond"], "target"))) for t in valid])
            p = OUT + f"inp/freq_{name}.npy"; np.save(p, f); files.append(dict(file=f"inp/freq_{name}.npy", sha=sha(p)))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seeds=list(SEEDS), panel_sha=sha(PANEL),
                   calibration={t["tag"]: {k: t.get(k) for k in ("c", "sigma", "calibrated_gs", "truth_cal_ok")} for t in tg}, files=files,
                   code={c: sha(c) for c in CODE if os.path.exists(c)}), open(MAN, "w"), indent=1, default=float)
    print("valid truths", len(valid), "| invalid", panel["invalid_seeds"], flush=True)
    for name, c in panel["cells"].items():
        print(name, c["role"], "eligible", c["eligible"], "in range", len(c["supported"]), "/", c["n_valid"],
              "| median nf %.3f gs %.3f" % (np.median([v["nf"] for v in c["membership"].values()]), np.median([v["gs"] for v in c["membership"].values()])), flush=True)


def job_list():
    P = json.load(open(PANEL)); J = []
    for ci, (name, c) in enumerate(P["cells"].items()):
        if not c["eligible"]:
            continue
        for tag in c["supported"]:
            for m in c["models"]:
                J.append(dict(exp="E12d", cell=name, cell_index=ci, ttype=c["ttype"], cond=c["cond"], tag=tag, stratum=(int(tag.split("_")[1]) - 1) % 3,
                              setting="S1", seed=11, model=m))
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    s = int(j["tag"].split("_")[1])
    return [resid_path(s, j["ttype"], j["cond"], "target"), resid_path(s, j["ttype"], j["cond"], "heldout"), OUT + "inp/SC_group.npy", OUT + f"inp/freq_{j['cell']}.npy"]


_SIM = {}


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib, luppi_r4e_e8 as E8
    if "fit" not in _SIM:
        _SIM["fit"] = L.make_simulator(100, E8.T, 3000.0)
    sim = _SIM["fit"]; t0 = time.time(); cfg = dict(L.HEADER); fs = job_files(j); s = int(j["tag"].split("_")[1])
    inp = Lib.bandpass(np.load(fs[0])); hop = Lib.branch(np.load(fs[1]), "F"); SC = np.load(fs[2]); f = np.load(fs[3])
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    base = 61_000_000 + 1000 * s + 10 * j["cell_index"]; sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm)
    if not (C.shape == (100, 100) and np.all(np.isfinite(C)) and np.isfinite(mse)):
        raise ValueError("non-finite or mis-shaped fitted matrix or held-out MSE")
    np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"], mse_ho=mse, C_sha=sha_arr(C))


def stage_freeze():
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("e12d_analyze.py"):
        raise SystemExit("analysis pre-registration missing or not matching")
    mf = json.load(open(MAN))
    if mf["panel_sha"] != sha(PANEL) or json.load(open(PANEL))["gsim_sha"] != sha(GSIM):
        raise SystemExit("panel or g_sim changed since preparation")
    man = {OUT + r["file"]: r["sha"] for r in mf["files"]}; J = job_list(); files = {}
    for j in J:
        for p in job_files(j):
            if p not in man or sha(p) != man[p]:
                raise SystemExit(f"{p} missing from the manifest or changed")
            files[p] = man[p]
    fr = dict(group="e12d", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.environ.get("COMPUTERNAME"), code={c: sha(c) for c in CODE},
              raw={p: sha(p) for p in RAW}, panel=sha(PANEL), manifest=sha(MAN), prereg=sha(PREREG), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen e12d", sha(FREEZE), "jobs", len(J), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(PANEL, fr["panel"]), (MAN, fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="e12d", inputs={p: fr["files"][p] for p in job_files(j)})


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
    dl_file = OUT + "e12d_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    J = job_list(); J.sort(key=lambda j: (j["cell"], j["tag"]))
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"prepare": lambda: stage_prepare(min(a.procs, 30)), "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
