"""Preprocessing comparison, step 2 (Section 2.3): the human fitting protocol (luppi_r4e_ext, unchanged) applied to one
preprocessing condition of the REST1 data.
Usage: python luppi_r4e_ppc.py COND e5_prepare | contract | freeze e5 | run e5 [--procs 56]"""
from paths import DATA_ROOT
import sys, os, json, time
import numpy as np
import luppi_r4e_ext as X

CONDS = ("vol0", "mot", "acc", "acc1", "acc3", "accero", "gsr", "null10")
COND = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in CONDS else os.environ.get("PPC_COND", "acc")
X.OUT = f"{DATA_ROOT}results/p33/r4e_ppc_{COND}/"
X.HCP = f"{DATA_ROOT}work/p33_ts_{COND}/"
X.CODE = X.CODE + ["luppi_r4e_ppc.py", "luppi_ppc_prep.py", "luppi_pipeline_vol.py", "luppi_ppc_analyze.py", "luppi_r4e_e8_analyze.py", "luppi_r4e_run.py",
                   "luppi_r4e_e5fix.py", "luppi_r4e_e5fix_analyze.py"]
X.RAW = X.RAW + [DATA_ROOT + "results/p33/ppc_lineage_manifest.json"]


def point(cond):
    """Re-point the frozen PPC module to a condition (used by preparation and by the analysis, which reads every condition)."""
    global COND
    COND = cond; X.OUT = f"{DATA_ROOT}results/p33/r4e_ppc_{cond}/"; X.HCP = f"{DATA_ROOT}work/p33_ts_{cond}/"


_job_list = X.job_list


def job_list_rest1(group):
    J = _job_list(group)
    return [j for j in J if group != "e5" or j["session"] == "REST1"]


X.job_list = job_list_rest1


def stage_prepare_rest1():
    """REST1-only copy of luppi_r4e_ext.stage_e5_prepare (identical operations and manifest format)."""
    import scipy.io as sio
    import luppi_hopf as L, luppi_r4e_lib as Lib
    os.makedirs(X.OUT + "e5", exist_ok=True); files = []
    np.save(X.OUT + "e5/SC.npy", sio.loadmat(X.DEMO + "single/SC.mat")["SC"].astype(float)); files.append(dict(file="e5/SC.npy", sha=X.sha(X.OUT + "e5/SC.npy")))
    lr = []
    for sub in X.subjects():
        for d, kind in (("LR", "input"), ("RL", "heldout")):
            x = np.load(X.HCP + f"{sub}_REST1_{d}_s100.npy").T.astype(float)[:, X.DROP:]
            y = Lib.bandpass(x); p = f"e5/{sub}_REST1_{kind}.npy"; np.save(X.OUT + p, y); files.append(dict(file=p, sha=X.sha(X.OUT + p)))
            if kind == "input":
                lr.append(y)
    f = L.regional_frequencies(lr); p = "e5/freq_REST1.npy"; np.save(X.OUT + p, f); files.append(dict(file=p, sha=X.sha(X.OUT + p)))
    print(COND, "REST1 subjects", len(lr), "frames", lr[0].shape, "freq range %.4f-%.4f Hz" % (f.min(), f.max()), flush=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), condition=COND, subjects=X.subjects(), sessions=["REST1"], drop=X.DROP, band=[0.008, 0.09], files=files,
                   code={c: X.sha(c) for c in X.CODE if os.path.exists(c)}, regeneration=X.compare_previous("r4e_ext_e5_manifest.json", files),
                   hcp_inputs={f"{s}_REST1_{d}": X.sha(X.HCP + f"{s}_REST1_{d}_s100.npy") for s in X.subjects() for d in ("LR", "RL")}),
              open(X.OUT + "r4e_ext_e5_manifest.json", "w"), indent=1)



def _worker_paths(_):
    return X.OUT, X.HCP, COND, len(X.job_list("e5"))


def _seed_bases(jobs):
    """Execute the FROZEN luppi_r4e_ext.run_job with a stub fitter and a recording simulator (no real fit, temp files only) and
 return the PRNG keys it uses for the 8 held-out evaluation simulations of each job."""
    import tempfile, jax, luppi_hopf as L
    rec, orig_fit, orig_sim = [], L.fit, X._SIM.get("sim")
    def stub_fit(inp, SC, f, cfg, **kw):
        return np.full((100, 100), 0.01) - np.eye(100) * 0.01, dict(stop="stub", iterations=0, fc_corr=0.0)
    def rec_sim(C, f, a, sigma, key):
        rec.append(int(np.asarray(jax.random.key_data(key) if hasattr(jax.random, "key_data") else key).ravel()[-1]))
        return np.random.default_rng(len(rec)).standard_normal((100, 1189))
    out = {}
    try:
        L.fit = stub_fit; X._SIM["sim"] = rec_sim
        with tempfile.TemporaryDirectory() as td:
            for j in jobs:
                rec.clear(); X.run_job(dict(j, file=td + "/x.json"), time.time() + 600); out[j["key"]] = list(rec)
    finally:
        L.fit = orig_fit
        if orig_sim is None:
            X._SIM.pop("sim", None)
        else:
            X._SIM["sim"] = orig_sim
    return out


def contract():
    import multiprocessing as mp
    J = X.job_list("e5"); full = _job_list("e5"); rest1 = [j for j in full if j["session"] == "REST1"]
    ok_keys = [j["key"] for j in J] == [j["key"] for j in rest1] and len(J) == 2 * len(X.subjects())
    with mp.Pool(2) as p:
        w = p.map(_worker_paths, range(2))
    ok_spawn = all(x == (X.OUT, X.HCP, COND, len(J)) for x in w)
    probe = [j for j in J if j["subject_index"] in (0, 1, 97)]                       # both models, first / second / last subject
    got = _seed_bases(probe); want = {j["key"]: [9_000_000 + 1000 * j["subject_index"] + r for r in range(8)] for j in probe}
    ok_seed = got == want
    res = dict(condition=COND, n_jobs=len(J), keys_equal_e5_rest1=ok_keys, spawned_paths=w, spawn_ok=ok_spawn, seed_probe_jobs=len(probe), seed_ok=ok_seed,
               seed_rule="evaluation keys 9e6 + 1000 * subject_index + r (REST1), executed through the frozen run_job")
    json.dump(res, open(X.OUT + "ppc_contract.json", "w"), indent=1); print({k: v for k, v in res.items() if k != "spawned_paths"}, flush=True)
    if not (ok_keys and ok_spawn and ok_seed):
        raise SystemExit("contract check failed")


def preflight():
    """Production runs only under a verified experiment freeze that pins this condition's group freeze."""
    import hashlib
    sh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    P = DATA_ROOT + "results/p33/ppc_experiment_freeze.json"
    if not os.path.exists(P):
        raise SystemExit("no experiment freeze: production run refused")
    ef = json.load(open(P)); bad = [p for p, h in ef["pinned"].items() if not os.path.exists(p) or sh(p) != h]
    fsha, _ = X.verify_freeze("e5")
    if bad or ef["group_freezes"].get(COND) != fsha:
        raise SystemExit(f"experiment freeze does not verify for {COND}: {bad[:3]}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cond", choices=CONDS); ap.add_argument("stage"); ap.add_argument("group", nargs="?"); ap.add_argument("--procs", type=int, default=56)
    a = ap.parse_args(); os.makedirs(X.OUT, exist_ok=True)
    if a.stage == "e5_prepare":
        stage_prepare_rest1()
    elif a.stage == "contract":
        contract()
    elif a.stage == "freeze":
        assert a.group == "e5"; X.stage_freeze("e5")
    elif a.stage == "run":
        assert a.group == "e5"; preflight(); X.stage_run("e5", a.procs)
