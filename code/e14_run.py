"""Initial exploratory L1 run under the released configuration K3 (Supplementary Note 10): cooperative truths with acc-matched
removal, the CMPH positive control and real acc fits. Its constants and job definitions are also used by the later L1 analyses."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/e14/"; P33 = DATA_ROOT + "results/p33/"; E11D = P33 + "r4e_e11/"; ACC = P33 + "r4e_ppc_acc/"
PANEL = OUT + "e14_panel.json"; MAN = OUT + "e14_manifest.json"; FREEZE = OUT + "e14_freeze.json"; PREREG = P33 + "e14_analyze_prereg.json"
DEMO = {"100307", "100408", "101107"}; N_REAL = 30; SEEDS = tuple(range(5031, 5061))
SIM_CELLS = (("L-COOP-d136", "COOP", "N49-d136"), ("L-COOP-raw", "COOP", "N49-raw"), ("L-CMPH-d136", "CMPH", "N49-d136"))
RAW = [DATA_ROOT + "raw/luppi_demo/gec_3/SC.mat", DATA_ROOT + "raw/luppi_demo/gec_3/FMRI.mat"]           # example SCs (Lib.demo_subject)
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_e8.py", "luppi_r4e_e11.py", "luppi_r4e_ext.py", "luppi_r4e_ppc.py", "e14_run.py", "e14_analyze.py"]
DEADLINE_H = 72.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sim_files(tag, tt, cond):
    b = E11D + f"truth/{tag}_{tt}_{cond}"
    return [b + "_target.npy", b + "_heldout.npy"]


def real_files(tag):
    return [ACC + "e5/SC.npy", ACC + "e5/freq_REST1.npy", ACC + f"e5/{tag}_REST1_input.npy", ACC + f"e5/{tag}_REST1_heldout.npy"]


def stage_prepare():
    import luppi_r4e_e11 as E11, luppi_r4e_ppc as W
    if os.path.exists(PANEL):
        raise SystemExit("panel already frozen")
    e11_fsha, e11_fr = E11.verify_freeze()
    X = W.X; W.point("acc"); acc_fsha, acc_fr = X.verify_freeze("e5")
    jobs = [j for j in X.job_list("e5") if j["model"] == "signed" and j["tag"] not in DEMO]
    tags = sorted({j["tag"] for j in jobs}); idx = {j["tag"]: j["subject_index"] for j in jobs}
    if len(tags) != 95:
        raise SystemExit(f"expected 95 non-demo acc subjects, found {len(tags)}")
    pick = sorted(np.random.default_rng(14).choice(len(tags), N_REAL, replace=False).tolist()); real = [tags[i] for i in pick]
    files = {}
    for tag in [f"SCG_{s}" for s in SEEDS]:
        for _, tt, cond in SIM_CELLS:
            for p in sim_files(tag, tt, cond):
                if p not in e11_fr["files"]:
                    raise SystemExit(f"{p} is not an E11 frozen input")
                files[p] = e11_fr["files"][p]
    for tag in real:
        for p, rel in zip(real_files(tag), ("e5/SC.npy", "e5/freq_REST1.npy", f"e5/{tag}_REST1_input.npy", f"e5/{tag}_REST1_heldout.npy")):
            k = f"ext:{rel}"
            if k not in acc_fr["files"]:
                raise SystemExit(f"{k} is not a PPC acc frozen input")
            files[p] = acc_fr["files"][k]
    bad = [p for p, h in files.items() if sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} upstream inputs changed, e.g. {bad[:3]}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), rng="default_rng(14).choice(95, 30, replace=False) over sorted non-demo tags",
                   real=real, subject_index={t: idx[t] for t in real}, sim_tags=[f"SCG_{s}" for s in SEEDS], sim_cells=[list(c) for c in SIM_CELLS],
                   upstream=dict(e11_freeze=e11_fsha, acc_freeze=acc_fsha)), open(PANEL, "x"), indent=1)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), panel_sha=sha(PANEL), files=files, code={c: sha(c) for c in CODE if os.path.exists(c)}),
              open(MAN, "w"), indent=1)
    print("panel: real", real, "| sim tags", len(SEEDS), "| input files verified", len(files), flush=True)


def job_list():
    P = json.load(open(PANEL)); J = []
    for cell, tt, cond in SIM_CELLS:
        for tag in P["sim_tags"]:
            for m in ("signed", "coop"):
                J.append(dict(exp="E14", cell=cell, kind="sim", ttype=tt, cond=cond, tag=tag, stratum=(int(tag.split("_")[1]) - 1) % 3, cfg="released", seed=11, model=m))
    for tag in P["real"]:
        for m in ("signed", "coop"):
            J.append(dict(exp="E14", cell="L-REAL-acc", kind="real", ttype="REAL", cond="acc", tag=tag, subject_index=P["subject_index"][tag], cfg="released", seed=11, model=m))
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    return sim_files(j["tag"], j["ttype"], j["cond"]) if j["kind"] == "sim" else real_files(j["tag"])


_SIM = {}


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib, luppi_r4e_e8 as E8
    if "fit" not in _SIM:
        _SIM["fit"] = L.make_simulator(100, E8.T, 3000.0)          # E8.T = Lib.T_LEN = 1189
    sim = _SIM["fit"]; t0 = time.time(); cfg = dict(L.RELEASED); fs = job_files(j)
    if j["kind"] == "sim":
        s = int(j["tag"].split("_")[1]); SC = Lib.demo_subject(j["stratum"])[1]
        inp = Lib.bandpass(np.load(fs[0])); hop = Lib.branch(np.load(fs[1]), "F"); f = L.regional_frequencies([inp]); base = 25_000_000 + 1000 * s
    else:
        SC = np.load(fs[0]); f = np.load(fs[1]); inp = np.load(fs[2]); hop = L.prep_emp(np.load(fs[3])); base = 9_000_000 + 1000 * j["subject_index"]
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
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("e14_analyze.py"):
        raise SystemExit("analysis pre-registration missing or not matching")
    mf = json.load(open(MAN))
    if mf["panel_sha"] != sha(PANEL):
        raise SystemExit("panel changed since preparation")
    upstream_check()
    J = job_list(); files = {}
    for j in J:
        for p in job_files(j):
            if p not in mf["files"] or sha(p) != mf["files"][p]:
                raise SystemExit(f"{p} missing from the manifest or changed")
            files[p] = mf["files"][p]
    fr = dict(group="e14", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.environ.get("COMPUTERNAME"), code={c: sha(c) for c in CODE},
              raw={p: sha(p) for p in RAW}, upstream=json.load(open(PANEL))["upstream"], panel=sha(PANEL), manifest=sha(MAN), prereg=sha(PREREG), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen e14", sha(FREEZE), "jobs", len(J), flush=True)


def upstream_check():
    """The upstream Simulation 2 and PPC-acc freezes must still verify and equal the hashes recorded in the panel at preparation."""
    import luppi_r4e_e11 as E11, luppi_r4e_ppc as W
    up = json.load(open(PANEL))["upstream"]; e11_fsha, _ = E11.verify_freeze(); X = W.X; W.point("acc"); acc_fsha, _ = X.verify_freeze("e5")
    if e11_fsha != up["e11_freeze"] or acc_fsha != up["acc_freeze"]:
        raise SystemExit("an upstream freeze differs from the one recorded at preparation")


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(PANEL, fr["panel"]), (MAN, fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    if fr["upstream"] != json.load(open(PANEL))["upstream"]:
        raise SystemExit("upstream record differs from the panel")
    upstream_check()
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="e14", inputs={p: fr["files"][p] for p in job_files(j)})


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
    dl_file = OUT + "e14_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    J = job_list(); J.sort(key=lambda j: (j["cell"], j["tag"], j["model"]))
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"prepare": stage_prepare, "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
