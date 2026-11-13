"""Distributional implementation comparison (Section 2.2.3; Supplementary Note 13): eight random streams per participant, model
and implementation under K1 and K3, evaluated with shared keys against the +/-0.03 equivalence criterion."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import sys, json, time, hashlib, argparse, subprocess, tempfile, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT, MATLAB_EXE

ROOT = DATA_ROOT; OUT = ROOT + "results/p33/vhvi/vh/"; VHDIR = ROOT + "scripts/vh/"
MATLAB = MATLAB_EXE
REPO = ROOT + "raw/luppi_repo_da592aa/competitive-cooperative-hopf-da592aa/"
K = {"K1": dict(sigma=0.01, eta_fc=2e-4, eta_tau=4e-5, max_C=0.10, alpha=0.0),
     "K3": dict(sigma=0.001, eta_fc=1e-3, eta_tau=1e-4, max_C=0.15, alpha=1e-4)}
SUBJ = (1, 2, 3); MODELS = (0, 1); STREAMS = range(8); IMPL = ("M", "P"); BOUND = 0.03; MIN_FITS = 6
JOBS = [dict(impl=i, subj=s, K=k, coop=c, r=r) for i in IMPL for s in SUBJ for k in K for c in MODELS for r in STREAMS]


def rules():
    import vhvi_rules
    vhvi_rules.check("vh", (VHDIR + "vh_matlab_run.m", os.path.abspath(__file__)))


def data():
    import scipy.io as sio
    a = sio.loadmat(REPO + "data/matlab/gec_3/SC.mat")["DTI_fibers_VolNorm_HCP"].ravel()
    b = sio.loadmat(REPO + "data/matlab/gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel()
    return [np.asarray(x, float) for x in a], [np.asarray(x, float) for x in b]


def cfg(k):
    return dict(K[k], max_iterations=5000, conv=1e-3, tau=1, a=-0.02, burnin=3000.0)


def seed_of(j):
    return 300000 + 1000 * j["subj"] + 100 * int(j["K"][1]) + 10 * j["coop"] + j["r"]


def name(j):
    return f"{j['impl']}_s{j['subj']}_{j['K']}_c{j['coop']}_r{j['r']}"


def fp_stats(C, SC, alpha):
    m = (SC != 0) & ~np.eye(len(SC), dtype=bool)
    return dict(fp0=float((C[m] < 0).mean()), fp2a=float((C[m] < -2 * alpha).mean()) if alpha > 0 else None)


_SIM = {}


def resim(C, f, TS, k, j):
    """r_sim: mean Pearson correlation (upper triangle) between 8 simulated FCs and the empirical FC (prep_emp)."""
    import jax, luppi_hopf as L
    T = TS.shape[1]
    if T not in _SIM:
        _SIM[T] = L.make_simulator(len(TS), T, 3000.0)
    c = cfg(k); FCe = L.fc(L.prep_emp(TS)); iu = np.triu_indices(len(TS), 1); rs = []
    for q in range(8):
        key = jax.random.PRNGKey(7_000_000 + 1000 * j["subj"] + 100 * int(k[1]) + 10 * j["coop"] + q)
        s = np.asarray(_SIM[T](C, f, c["a"], c["sigma"], key)); rs.append(float(np.corrcoef(FCe[iu], L.fc(s)[iu])[0, 1]))
    if not np.all(np.isfinite(rs)):
        raise ValueError("non-finite re-simulation")
    return float(np.mean(rs)), rs


def run_matlab(j, outdir):
    import scipy.io as sio
    mf = outdir + name(j) + ".mat"
    if not os.path.exists(mf):
        cmd = [MATLAB, "-batch", f"vh_matlab_run({j['subj']},'{j['K']}',{j['coop']},{j['r']},'{mf}')"]
        p = subprocess.run(cmd, cwd=VHDIR, capture_output=True, text=True, errors="replace")
        if not os.path.exists(mf):
            raise RuntimeError("MATLAB produced no output: " + (p.stdout + p.stderr)[-600:])
    m = sio.loadmat(mf); line = str(np.ravel(m["stopline"])[0])
    it = int(line.split()[-1]) if line.split()[-1].isdigit() else 5000
    return dict(C=np.asarray(m["GEC"], float), f=np.ravel(m["freqs"]).astype(float), r_last=float(np.ravel(m["FC_corr"])[0]),
                stop=line, iterations=it, secs=float(np.ravel(m["elapsed"])[0]))


def run_python(j, iters=None):
    import luppi_hopf as L
    SCs, TSs = data(); SC, TS = SCs[j["subj"] - 1], TSs[j["subj"] - 1]; f = L.regional_frequencies([TS]); c = cfg(j["K"])
    if iters:
        c["max_iterations"] = iters
    t0 = time.time(); C, info = L.fit(TS, SC, f, c, seed=seed_of(j), cooperative_only=bool(j["coop"]), lag="matlab")
    return dict(C=C, f=np.asarray(f, float), r_last=float(info["fc_corr_last"]), stop=info["stop"], iterations=int(info["iterations"]),
                secs=round(time.time() - t0, 1))


def run_job(j, outdir, iters=None):
    SCs, TSs = data(); SC, TS = SCs[j["subj"] - 1], TSs[j["subj"] - 1]
    fit = run_matlab(j, outdir) if j["impl"] == "M" else run_python(j, iters)
    if not np.all(np.isfinite(fit["C"])):
        raise ValueError("non-finite fit")
    r_sim, rs = resim(fit["C"], fit["f"], TS, j["K"], j)
    np.save(outdir + name(j) + "_C.npy", fit["C"]); np.save(outdir + name(j) + "_f.npy", fit["f"])
    return dict(j, state="done", seed=seed_of(j), r_sim=r_sim, r_sims=rs, r_last=fit["r_last"], stop=fit["stop"], iterations=fit["iterations"],
                secs=fit["secs"], **fp_stats(fit["C"], SC, K[j["K"]]["alpha"]),
                C_sha=hashlib.sha256(np.ascontiguousarray(fit["C"]).tobytes()).hexdigest())


def safe(j):
    p = OUT + "fits/" + name(j) + ".json"
    if os.path.exists(p) and json.load(open(p)).get("state") == "done":
        return "done"
    err = ""
    for attempt in (1, 2):
        try:
            r = run_job(j, OUT + "fits/"); json.dump(r, open(p, "w"), indent=1); return "done"
        except Exception:
            err = traceback.format_exc()[-800:]
    json.dump(dict(j, state="failed_twice", error=err), open(p, "w"), indent=1); return "failed"


def stage_run(procs):
    rules(); os.makedirs(OUT + "fits", exist_ok=True)
    J = sorted(JOBS, key=lambda j: j["impl"] != "M")                    # MATLAB jobs (slowest) first
    with mp.Pool(procs) as p:
        st = p.map(safe, J, chunksize=1)
    print({s: st.count(s) for s in set(st)}, flush=True)


# ---------------------------------------------------------------- analysis (pure functions; tested by stage_fixture)
def primary(k):
    return [("r_sim", 0), ("r_sim", 1), ("fp0" if k == "K1" else "fp2a", 0)]


def compare(recs, k, ep, coop, rng, B=2000):
    """Delta (ours - MATLAB) with stream bootstrap; reading against BOUND."""
    cells = {(i, s): np.array([r[ep] for r in recs if r["impl"] == i and r["subj"] == s and r["K"] == k and r["coop"] == coop
                               and r.get(ep) is not None], float) for i in IMPL for s in SUBJ}
    n = {f"{i}{s}": int(v.size) for (i, s), v in cells.items()}
    if min(v.size for v in cells.values()) < MIN_FITS:
        return dict(n=n, reading="not analysable")
    d = float(np.mean([cells[("P", s)].mean() - cells[("M", s)].mean() for s in SUBJ])); bs = np.empty(B)
    for b in range(B):
        bs[b] = np.mean([rng.choice(cells[("P", s)], cells[("P", s)].size).mean() - rng.choice(cells[("M", s)], cells[("M", s)].size).mean() for s in SUBJ])
    lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
    reading = "equivalent" if (lo > -BOUND and hi < BOUND) else "different" if (lo > BOUND or hi < -BOUND) else "undetermined"
    sd = {i: float(np.mean([cells[(i, s)].std(ddof=1) for s in SUBJ])) for i in IMPL}
    per = {s: dict(M=float(cells[("M", s)].mean()), P=float(cells[("P", s)].mean())) for s in SUBJ}
    return dict(n=n, delta=d, ci95=[lo, hi], reading=reading, mean_within_sd=sd, per_participant=per)


def analyze_records(recs, seed=20261026):
    rng = np.random.default_rng(seed); out = dict(primary={}, secondary={})
    for k in K:
        for ep, coop in primary(k):
            out["primary"][f"{k}|{ep}|{'coop' if coop else 'signed'}"] = compare(recs, k, ep, coop, rng)
        sec = [("r_last", 0), ("r_last", 1), ("iterations", 0), ("iterations", 1), ("fp0", 0)] + ([("fp2a", 0)] if k == "K3" else [])
        for ep, coop in sec:
            key = f"{k}|{ep}|{'coop' if coop else 'signed'}"
            if key not in out["primary"]:
                out["secondary"][key] = compare(recs, k, ep, coop, rng)
    rd = [v["reading"] for v in out["primary"].values()]
    out["VH-A"] = bool(all(x == "equivalent" for x in rd)); out["VH-S"] = bool(any(x == "different" for x in rd))
    out["overall"] = "VH-A" if out["VH-A"] else "VH-S" if out["VH-S"] else ("not analysable" if all(x == "not analysable" for x in rd) else "partly determined")
    return out


def stage_analyze(path):
    rules(); recs = []; fdiff = []
    for j in JOBS:
        p = OUT + "fits/" + name(j) + ".json"
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("state") == "done":
                C = np.load(OUT + "fits/" + name(j) + "_C.npy")
                if hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest() == r["C_sha"]:
                    recs.append(r)
    o = analyze_records(recs); o["n_fits"] = len(recs)
    for s in SUBJ:                                         # frequency diagnostic: the authors' vs our frequency vector
        fm = [np.load(OUT + "fits/" + name(dict(impl="M", subj=s, K=k, coop=c, r=r)) + "_f.npy") for k in K for c in MODELS for r in STREAMS
              if os.path.exists(OUT + "fits/" + name(dict(impl="M", subj=s, K=k, coop=c, r=r)) + "_f.npy")]
        fp_ = [np.load(OUT + "fits/" + name(dict(impl="P", subj=s, K=k, coop=c, r=r)) + "_f.npy") for k in K for c in MODELS for r in STREAMS
               if os.path.exists(OUT + "fits/" + name(dict(impl="P", subj=s, K=k, coop=c, r=r)) + "_f.npy")]
        if fm and fp_:
            fdiff.append(dict(subj=s, max_abs_diff=float(np.abs(fm[0] - fp_[0]).max()), n_regions_differing=int((np.abs(fm[0] - fp_[0]) > 1e-12).sum()),
                              matlab_constant=bool(all(np.array_equal(fm[0], x) for x in fm)), python_constant=bool(all(np.array_equal(fp_[0], x) for x in fp_))))
    o["frequency_diagnostic"] = fdiff
    json.dump(o, open(path, "x"), indent=1)
    print("VH | fits", len(recs), "| overall", o["overall"], flush=True)
    for key, v in o["primary"].items():
        print(" ", key, v.get("reading"), "delta %.4f" % v["delta"] if "delta" in v else "", v.get("ci95"), v.get("mean_within_sd"), flush=True)
    print("  frequency", fdiff, flush=True)


def stage_fixture():
    """(1) one full MATLAB job (subject 1, K3, cooperative, r = 0) into a temporary directory: wrapper, saved frequencies, resim;
    (2) one Python job with 120 iterations; (3) the analysis on synthetic records covering every reading. No endpoint is kept."""
    import luppi_hopf as L
    td = tempfile.mkdtemp() + "/"; SCs, TSs = data()
    j = dict(impl="P", subj=1, K="K1", coop=0, r=0); t0 = time.time(); r = run_job(j, td, iters=120)
    print("python fixture | iterations", r["iterations"], "| r_sim finite", np.isfinite(r["r_sim"]), "| secs %.0f" % (time.time() - t0), flush=True)
    j = dict(impl="M", subj=1, K="K3", coop=1, r=0); t0 = time.time(); r = run_job(j, td)
    fm = np.load(td + name(j) + "_f.npy"); fo = L.regional_frequencies([TSs[0]])
    print("matlab fixture | stop", r["stop"], "| r_sim finite", np.isfinite(r["r_sim"]), "| freq max|M-P| %.2e" % np.abs(fm - fo).max(),
          "| secs %.0f" % (time.time() - t0), flush=True)
    g = np.random.default_rng(1); recs = []
    for i in IMPL:
        for s in SUBJ:
            for k in K:
                for c in MODELS:
                    for rr in STREAMS:
                        shift = {"K1": 0.0, "K3": 0.2}[k] if i == "P" and c == 0 else 0.0     # K3 signed: large difference
                        noisy = 0.5 if (k == "K1" and c == 1) else 0.005                      # K1 coop: undetermined
                        if k == "K3" and c == 1 and s == 1 and i == "M" and rr > 3:
                            continue                                                           # K3 coop: < 6 fits -> not analysable
                        recs.append(dict(impl=i, subj=s, K=k, coop=c, r=rr, r_sim=0.6 + shift + g.normal(0, noisy), r_last=0.6 + g.normal(0, 0.05),
                                         fp0=0.25 + g.normal(0, 0.005), fp2a=0.15 + g.normal(0, 0.005), iterations=300))
    o = analyze_records(recs)
    print("analysis fixture |", {k: v["reading"] for k, v in o["primary"].items()}, "| overall", o["overall"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("arg", nargs="?"); ap.add_argument("--procs", type=int, default=60); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(fixture=stage_fixture, run=lambda: stage_run(a.procs), analyze=lambda: stage_analyze(a.arg))[a.stage]()
