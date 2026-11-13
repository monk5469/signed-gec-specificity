"""Validation set V2 (Section 2.5.1): preprocessing of the 60 additional HCP participants under the eight PPC conditions with the
frozen PPC code, followed by fits with the PPC estimator protocol."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/e12a_v2/"; P33 = DATA_ROOT + "results/p33/"; NEWRAW = DATA_ROOT + "raw/p33_hcp_new/vol/"
NEWWORK = DATA_ROOT + "work/e12a_v2/"; SUBJ = DATA_ROOT + "raw/p33_hcp_new/e12a_v2_subjects.json"; DEMO = DATA_ROOT + "raw/luppi_demo/"; DROP = 11
CONDS = ("vol0", "mot", "null10", "acc1", "acc3", "acc", "accero", "gsr")
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_ppc_prep.py", "luppi_pipeline_vol.py", "e12a_new.py", "e12a_g_freeze.py", "e12a_validate_v2.py"]
LINEAGE = OUT + "e12a_v2_lineage.json"; MAN = OUT + "e12a_v2_manifest.json"; FREEZE = OUT + "e12a_v2_freeze.json"
PREREG = P33 + "e12a_validate_v2_prereg.json"; XCHECK = P33 + "e12a_xcheck.json"; ETAGV = DATA_ROOT + "raw/p33_hcp_new/v2_etag_verify.json"
RAW = [SUBJ, DEMO + "single/SC.mat", P33 + "e12a_g.json", XCHECK, ETAGV]; DEADLINE_H = 72.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def subjects():
    return json.load(open(SUBJ))["selected"]


def _repoint():
    import luppi_ppc_prep as PP
    PP.RAW = NEWRAW; PP.WORK = NEWWORK


def _prep_job(args):
    import luppi_ppc_prep as PP
    return PP.job(args)


def stage_prep(procs):
    if not os.path.exists(ETAGV) or not json.load(open(ETAGV)).get("all_ok"):
        raise SystemExit("download ETag verification missing or failed")
    os.makedirs(OUT, exist_ok=True)
    with mp.Pool(procs, initializer=_repoint) as p:
        res = p.map(_prep_job, [(s, d) for s in subjects() for d in ("LR", "RL")], chunksize=1)
    import luppi_ppc_prep as PP, luppi_pipeline_vol as PV
    man = dict(n=len(res), n_ok=sum(r["ok"] for r in res), raw=NEWRAW, work=NEWWORK, mask_source=dict(file=PP.MASK_SRC, sha=sha(PP.MASK_SRC)),
               atlas=dict(file=PV.ATLAS, sha=sha(PV.ATLAS)), code={c: sha(c) for c in ("luppi_ppc_prep.py", "luppi_pipeline_vol.py", "e12a_new.py")}, records=res)
    json.dump(man, open(LINEAGE, "x"), indent=1)
    print("runs", len(res), "ok", man["n_ok"], "bad", [(r["subject"], r["run"], r.get("error")) for r in res if not r["ok"]][:5], flush=True)
    for c in CONDS:
        g = [r["cond"][c]["gs_share"] for r in res if r["ok"] and r["run"] == "LR"]; n = [r["cond"][c]["neg_fc"] for r in res if r["ok"] and r["run"] == "LR"]
        print(c, "LR gs median %.3f | negFC median %.3f" % (np.median(g), np.median(n)), flush=True)


def stage_prepare():
    import scipy.io as sio
    import luppi_hopf as L, luppi_r4e_lib as Lib
    lin = json.load(open(LINEAGE)); okr = {(r["subject"], r["run"]) for r in lin["records"] if r["ok"]}
    subs = [s for s in subjects() if (s, "LR") in okr and (s, "RL") in okr]
    os.makedirs(OUT + "inp", exist_ok=True); files = []; src = {}
    np.save(OUT + "inp/SC.npy", sio.loadmat(DEMO + "single/SC.mat")["SC"].astype(float)); files.append(dict(file="inp/SC.npy", sha=sha(OUT + "inp/SC.npy")))
    for c in CONDS:
        lr = []
        for sub in subs:
            for d, kind in (("LR", "input"), ("RL", "heldout")):
                p0 = NEWWORK + f"p33_ts_{c}/{sub}_REST1_{d}_s100.npy"; src[p0] = sha(p0)
                y = Lib.bandpass(np.load(p0).T.astype(float)[:, DROP:]); p = f"inp/{c}_{sub}_{kind}.npy"; np.save(OUT + p, y); files.append(dict(file=p, sha=sha(OUT + p)))
                if kind == "input":
                    lr.append(y)
        f = L.regional_frequencies(lr); p = f"inp/{c}_freq.npy"; np.save(OUT + p, f); files.append(dict(file=p, sha=sha(OUT + p)))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), subjects=subs, conds=CONDS, files=files, sources=src, lineage_sha=sha(LINEAGE),
                   code={c: sha(c) for c in CODE if os.path.exists(c)}), open(MAN, "w"), indent=1)
    print("prepared", len(subs), "subjects x", len(CONDS), "conditions", flush=True)


def job_list():
    subs = json.load(open(MAN))["subjects"]
    J = [dict(exp="E12a_V2", cond=c, cond_index=ci, tag=str(sub), subject_index=i, session="REST1", input_run="LR", seed=11, model=m)
         for ci, c in enumerate(CONDS) for i, sub in enumerate(subs) for m in ("signed", "coop")]
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    return [OUT + "inp/SC.npy", OUT + f"inp/{j['cond']}_freq.npy", OUT + f"inp/{j['cond']}_{j['tag']}_input.npy", OUT + f"inp/{j['cond']}_{j['tag']}_heldout.npy"]


_SIM = {}


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if "sim" not in _SIM:
        _SIM["sim"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["sim"]; t0 = time.time(); cfg = dict(L.HEADER); fs = job_files(j)
    SC = np.load(fs[0]); f = np.load(fs[1]); inp = np.load(fs[2]); hop = L.prep_emp(np.load(fs[3]))
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    base = 41_000_000 + 100_000 * j["cond_index"] + 1000 * j["subject_index"]; sims = []
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
    if not json.load(open(XCHECK)).get("identical"):
        raise SystemExit("cross-machine pre-check not bit-identical")
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("e12a_validate_v2.py"):
        raise SystemExit("V2 validation pre-registration missing or not matching")
    mf = json.load(open(MAN)); man = {OUT + r["file"]: r["sha"] for r in mf["files"]}
    if mf["lineage_sha"] != sha(LINEAGE) or any(sha(p) != h for p, h in mf["sources"].items()):
        raise SystemExit("lineage or source series changed")
    J = job_list(); files = {}
    for j in J:
        for p in job_files(j):
            if p not in man or sha(p) != man[p]:
                raise SystemExit(f"{p} missing from the manifest or changed")
            files[p] = man[p]
    fr = dict(group="e12a_v2", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.environ.get("COMPUTERNAME"), code={c: sha(c) for c in CODE},
              raw={p: sha(p) for p in RAW}, lineage=sha(LINEAGE), manifest=sha(MAN), prereg=sha(PREREG), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen e12a_v2", sha(FREEZE), "jobs", len(J), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(LINEAGE, fr["lineage"]), (MAN, fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="e12a_v2", inputs={p: fr["files"][p] for p in job_files(j)})


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
    dl_file = OUT + "e12a_v2_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    J = job_list(); J.sort(key=lambda j: (j["cond"] != "acc", j["cond"], j["subject_index"]))
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=60); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"prep": lambda: stage_prep(min(a.procs, 30)), "prepare": stage_prepare, "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
