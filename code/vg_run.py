"""Implementation check (Section 2.2.3; Supplementary Note 13): the authors' unmodified MATLAB fitting function (via
vg_matlab_run.m) and the reimplementation on the three example participants, one run per configuration.
Usage: python vg_run.py probe | rules | run"""
import sys, os, json, time, hashlib, argparse, subprocess, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT, MATLAB_EXE

OUT = DATA_ROOT + "results/p33/vg/"; VGDIR = DATA_ROOT + "scripts/vg/"; MATLAB = MATLAB_EXE
REPO = DATA_ROOT + "raw/luppi_repo_da592aa/competitive-cooperative-hopf-da592aa/"; SPEC = DATA_ROOT + "docs_r1/176_vg_spec_v0.1.md"
RULES = OUT + "vg_rules.json"; CODE = (VGDIR + "vg_matlab_run.m", "vg_run.py", "vg_analyze.py")
K = {"K1": dict(sigma=0.01, eta_fc=2e-4, eta_tau=4e-5, max_C=0.10, alpha=0.0),
     "K2": dict(sigma=0.01, eta_fc=2e-4, eta_tau=4e-5, max_C=0.15, alpha=0.0),
     "K3": dict(sigma=0.001, eta_fc=1e-3, eta_tau=1e-4, max_C=0.15, alpha=1e-4),
     "K4": dict(sigma=0.001, eta_fc=1e-3, eta_tau=1e-4, max_C=0.15, alpha=0.0)}
JOBS = [(s, k, coop) for s in (1, 2, 3) for k in K for coop in (0, 1)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def data():
    import scipy.io as sio
    a = sio.loadmat(REPO + "data/matlab/gec_3/SC.mat")["DTI_fibers_VolNorm_HCP"].ravel()
    b = sio.loadmat(REPO + "data/matlab/gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel()
    return [np.asarray(x, float) for x in a], [np.asarray(x, float) for x in b]


def cfg(k):
    return dict(K[k], max_iterations=5000, conv=1e-3, tau=1, a=-0.02, burnin=3000.0)


def mfile(s, k, coop):
    return OUT + f"matlab/s{s}_{k}_c{coop}.mat"


def pfile(s, k, coop):
    return OUT + f"python/s{s}_{k}_c{coop}.npz"


def matlab_job(args):
    s, k, coop = args
    if os.path.exists(mfile(s, k, coop)):
        return "skip"
    cmd = [MATLAB, "-batch", f"vg_matlab_run({s},'{k}',{coop},'{mfile(s, k, coop)}',0)"]
    r = subprocess.run(cmd, cwd=VGDIR, capture_output=True, text=True, errors="replace")
    return (r.stdout.strip().splitlines() or ["?"])[-1][:200]


def python_job(args, iters=None):
    s, k, coop = args
    if iters is None and os.path.exists(pfile(s, k, coop)):
        return "skip"
    import luppi_hopf as L
    SCs, TSs = data(); SC, TS = SCs[s - 1], TSs[s - 1]; f = L.regional_frequencies([TS]); c = cfg(k)
    if iters:
        c["max_iterations"] = iters
    t0 = time.time(); C, info = L.fit(TS, SC, f, c, seed=11, cooperative_only=bool(coop), lag="matlab")
    if iters:
        return dict(secs=round(time.time() - t0, 1), iterations=info["iterations"])
    np.savez(pfile(s, k, coop), C=C, fc_corr_last=info["fc_corr_last"], fc_corr_post=info["fc_corr"], iterations=info["iterations"],
             stop=info["stop"], secs=time.time() - t0)
    return f"{k} s{s} c{coop} r {info['fc_corr_last']:.4f} {info['stop']} {info['iterations']}"


def check_rules():
    r = json.load(open(RULES))["hashes"]; now = {os.path.basename(c): sha(c) for c in CODE}; now["spec"] = sha(SPEC)
    if r != now:
        raise SystemExit(f"rules changed: {[q for q in now if now[q] != r.get(q)]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=48); a = ap.parse_args()
    os.makedirs(OUT + "matlab", exist_ok=True); os.makedirs(OUT + "python", exist_ok=True)
    if a.stage == "probe":
        r = subprocess.run([MATLAB, "-batch", f"vg_matlab_run(1,'K1',0,'{OUT}probe.mat',1)"], cwd=VGDIR, capture_output=True, text=True, errors="replace")
        print("matlab:", r.stdout.strip()[-300:], r.stderr.strip()[-300:], flush=True)
        print("python 120 iterations:", python_job((1, "K1", 0), iters=120), flush=True)
    elif a.stage == "rules":
        if os.path.exists(RULES):
            raise SystemExit("rules already registered")
        h = {os.path.basename(c): sha(c) for c in CODE}; h["spec"] = sha(SPEC)
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), hashes=h), open(RULES, "x"), indent=1); print("rules", h, flush=True)
    elif a.stage == "run":
        check_rules()
        with mp.Pool(a.procs) as p:
            rm = p.map_async(matlab_job, JOBS, chunksize=1); rp = p.map_async(python_job, JOBS, chunksize=1)
            for line in rm.get() + rp.get():
                print(line, flush=True)
