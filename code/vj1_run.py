"""Bifurcation-parameter check on the example data (Sections 2.2.3, 3.7.5; Supplementary Note 13): the authors' unmodified MATLAB
function under the configurations with a = -0.025, with repeated random streams and the aggregate criterion."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import sys, json, time, hashlib, argparse, subprocess, tempfile, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT, MATLAB_EXE

ROOT = DATA_ROOT; OUT = ROOT + "results/p33/vj/vj1/"; VJDIR = ROOT + "scripts/vj/"; VH = ROOT + "results/p33/vhvi/vh/fits/"
MATLAB = MATLAB_EXE
REPO = ROOT + "raw/luppi_repo_da592aa/competitive-cooperative-hopf-da592aa/"
K = {"K1a": dict(kidx=1, a=-0.025, sigma=0.01, eta_fc=2e-4, eta_tau=4e-5, max_C=0.10, alpha=0.0),
     "K3a": dict(kidx=3, a=-0.025, sigma=0.001, eta_fc=1e-3, eta_tau=1e-4, max_C=0.15, alpha=1e-4),
     "K4a": dict(kidx=4, a=-0.025, sigma=0.001, eta_fc=1e-3, eta_tau=1e-4, max_C=0.15, alpha=0.0),
     "KR": dict(kidx=5, a=-0.025, sigma=0.005, eta_fc=1e-3, eta_tau=1e-4, max_C=0.15, alpha=1e-4),
     "K4r": dict(kidx=6, a=-0.02, sigma=0.001, eta_fc=1e-3, eta_tau=1e-4, max_C=0.15, alpha=0.0)}
VH_ALPHA = {"K1": 0.0, "K3": 1e-4}
A_PAIRS = (("K1a", "K1"), ("K3a", "K3"), ("K4a", "K4r")); A025 = ("K1a", "K3a", "K4a", "KR")
TARGET = dict(r_signed=0.74, r_coop=0.51, fp0=0.25); D_MAX = 0.15
SUBJ = (1, 2, 3); MODELS = (0, 1); STREAMS = range(8); MIN_FITS = 6
JOBS = [dict(subj=s, K=k, coop=c, r=r) for s in SUBJ for k in K for c in MODELS for r in STREAMS]


def rules():
    import vj_rules
    vj_rules.check("vj1", (VJDIR + "vj_matlab_run.m", os.path.abspath(__file__)))


def data():
    import scipy.io as sio
    a = sio.loadmat(REPO + "data/matlab/gec_3/SC.mat")["DTI_fibers_VolNorm_HCP"].ravel()
    b = sio.loadmat(REPO + "data/matlab/gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel()
    return [np.asarray(x, float) for x in a], [np.asarray(x, float) for x in b]


def seed_of(j):
    return 400000 + 1000 * j["subj"] + 100 * K[j["K"]]["kidx"] + 10 * j["coop"] + j["r"]


def name(j):
    return f"M_s{j['subj']}_{j['K']}_c{j['coop']}_r{j['r']}"


def fp_stats(C, SC, alpha):
    m = (SC != 0) & ~np.eye(len(SC), dtype=bool)
    return dict(fp0=float((C[m] < 0).mean()), fp2a=float((C[m] < -2 * alpha).mean()) if alpha > 0 else None)


_SIM = {}


def resim(C, f, TS, j):
    import jax, luppi_hopf as L
    T = TS.shape[1]
    if T not in _SIM:
        _SIM[T] = L.make_simulator(len(TS), T, 3000.0)
    k = K[j["K"]]; FCe = L.fc(L.prep_emp(TS)); iu = np.triu_indices(len(TS), 1); rs = []
    for q in range(8):
        key = jax.random.PRNGKey(7_500_000 + 1000 * j["subj"] + 100 * k["kidx"] + 10 * j["coop"] + q)
        s = np.asarray(_SIM[T](C, f, k["a"], k["sigma"], key)); rs.append(float(np.corrcoef(FCe[iu], L.fc(s)[iu])[0, 1]))
    if not np.all(np.isfinite(rs)):
        raise ValueError("non-finite re-simulation")
    return float(np.mean(rs)), rs


def run_matlab(j, outdir):
    import scipy.io as sio
    mf = outdir + name(j) + ".mat"
    if not os.path.exists(mf):
        cmd = [MATLAB, "-batch", f"vj_matlab_run({j['subj']},'{j['K']}',{j['coop']},{j['r']},'{mf}')"]
        p = subprocess.run(cmd, cwd=VJDIR, capture_output=True, text=True, errors="replace")
        if not os.path.exists(mf):
            raise RuntimeError("MATLAB produced no output: " + (p.stdout + p.stderr)[-600:])
    m = sio.loadmat(mf); line = str(np.ravel(m["stopline"])[0])
    it = int(line.split()[-1]) if line.split()[-1].isdigit() else 5000
    return dict(C=np.asarray(m["GEC"], float), f=np.ravel(m["freqs"]).astype(float), r_last=float(np.ravel(m["FC_corr"])[0]),
                stop=line, iterations=it, secs=float(np.ravel(m["elapsed"])[0]))


def run_job(j, outdir):
    SCs, TSs = data(); SC, TS = SCs[j["subj"] - 1], TSs[j["subj"] - 1]; fit = run_matlab(j, outdir)
    if not np.all(np.isfinite(fit["C"])):
        raise ValueError("non-finite fit")
    r_sim, rs = resim(fit["C"], fit["f"], TS, j)
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
    J = sorted(JOBS, key=lambda j: (j["K"] not in ("K1a", "K4a"), j["subj"], j["coop"], j["r"]))
    with mp.Pool(procs) as p:
        st = p.map(safe, J, chunksize=1)
    print({s: st.count(s) for s in set(st)}, flush=True)


# ---------------------------------------------------------------- analysis (pure functions; tested by stage_fixture)
def cell(recs, k, coop, ep, s):
    return np.array([r[ep] for r in recs if r["K"] == k and r["coop"] == coop and r["subj"] == s and r.get(ep) is not None], float)


def d_of(recs, k):
    """D on the medians over participants of the per-participant stream means; None if any cell has fewer than MIN_FITS fits."""
    parts = dict(r_signed=(0, "r_sim"), r_coop=(1, "r_sim"), fp0=(0, "fp0"))
    med = {}
    for nm, (coop, ep) in parts.items():
        cs = [cell(recs, k, coop, ep, s) for s in SUBJ]
        if min(c.size for c in cs) < MIN_FITS:
            return dict(D=None, reading="not analysable", n=[int(c.size) for c in cs])
        med[nm] = float(np.median([c.mean() for c in cs]))
    D = sum(abs(med[x] - TARGET[x]) for x in TARGET)
    return dict(D=float(D), medians=med, reproduces=bool(D <= D_MAX))


def a_effect(recs, ka, kr, ep, coop, rng, B=2000):
    ca = {s: cell(recs, ka, coop, ep, s) for s in SUBJ}; cr = {s: cell(recs, kr, coop, ep, s) for s in SUBJ}
    if min(v.size for v in list(ca.values()) + list(cr.values())) < MIN_FITS:
        return dict(reading="not analysable")
    d = float(np.mean([ca[s].mean() - cr[s].mean() for s in SUBJ])); bs = np.empty(B)
    for b in range(B):
        bs[b] = np.mean([rng.choice(ca[s], ca[s].size).mean() - rng.choice(cr[s], cr[s].size).mean() for s in SUBJ])
    return dict(delta=d, ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                per_participant={s: dict(a=float(ca[s].mean()), ref=float(cr[s].mean())) for s in SUBJ})


def analyze_records(recs, seed=20261028):
    rng = np.random.default_rng(seed); o = dict(D={k: d_of(recs, k) for k in list(K) + list(VH_ALPHA)}, a_effect={})
    for ka, kr in A_PAIRS:
        eps = [("r_sim", 0), ("r_sim", 1), ("fp0", 0), ("r_last", 0), ("r_last", 1), ("iterations", 0)] + ([("fp2a", 0)] if K[ka]["alpha"] > 0 else [])
        for ep, coop in eps:
            o["a_effect"][f"{ka}-{kr}|{ep}|{'coop' if coop else 'signed'}"] = a_effect(recs, ka, kr, ep, coop, rng)
    ok = {k: v for k, v in o["D"].items() if k in A025 and v.get("D") is not None}
    best = min(ok, key=lambda k: ok[k]["D"]) if ok else None
    o["best_a025"] = best; o["reproduced_a025"] = sorted(k for k, v in ok.items() if v["reproduces"])
    o["VJ2b_trigger"] = bool(best is not None and ok[best]["reproduces"] and best not in ("K1a", "K4a"))
    o["reading"] = ("a = -0.025 configuration reproduces: " + ", ".join(o["reproduced_a025"])) if o["reproduced_a025"] else \
        ("no a = -0.025 configuration reproduces" if len(ok) == len(A025) else "partly analysable")
    return o


def load_vh():
    """distributional implementation comparison MATLAB fits of K1 and K3 (a = -0.02), hash-checked, relabelled to K1 / K3 with the same record fields."""
    recs = []
    for s in SUBJ:
        for k in VH_ALPHA:
            for c in MODELS:
                for r in STREAMS:
                    nm = f"M_s{s}_{k}_c{c}_r{r}"
                    if os.path.exists(VH + nm + ".json"):
                        x = json.load(open(VH + nm + ".json"))
                        if x.get("state") == "done" and hashlib.sha256(np.ascontiguousarray(np.load(VH + nm + "_C.npy")).tobytes()).hexdigest() == x["C_sha"]:
                            recs.append(dict(subj=s, K=k, coop=c, r=r, r_sim=x["r_sim"], r_last=x["r_last"], fp0=x["fp0"], fp2a=x.get("fp2a"), iterations=x["iterations"]))
    return recs


def stage_analyze(path):
    rules(); recs = []
    for j in JOBS:
        p = OUT + "fits/" + name(j) + ".json"
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("state") == "done" and hashlib.sha256(np.ascontiguousarray(np.load(OUT + "fits/" + name(j) + "_C.npy")).tobytes()).hexdigest() == r["C_sha"]:
                recs.append(r)
    vh = load_vh(); o = analyze_records(recs + vh); o["n_fits"] = len(recs); o["n_vh_reference"] = len(vh)
    json.dump(o, open(path, "x"), indent=1)
    print("VJ1 | fits", len(recs), "| VH reference", len(vh), "|", o["reading"], "| best", o["best_a025"], "| VJ2b", o["VJ2b_trigger"], flush=True)
    for k, v in o["D"].items():
        print(" ", k, v, flush=True)
    for k, v in o["a_effect"].items():
        print(" ", k, "delta %.4f" % v["delta"] if "delta" in v else v.get("reading"), v.get("ci95"), flush=True)


def stage_fixture():
    """(1) one full MATLAB job (subject 1, KR, signed, r = 0) into a temporary directory: wrapper, a and sigma, saved frequencies,
    re-simulation; (2) the distributional implementation comparison reference loader (count only); (3) the analysis on synthetic records covering every reading. No
    endpoint is kept."""
    import luppi_hopf as L
    td = tempfile.mkdtemp() + "/"; SCs, TSs = data(); j = dict(subj=1, K="KR", coop=0, r=0); t0 = time.time(); r = run_job(j, td)
    import scipy.io as sio
    c = sio.loadmat(td + name(j) + ".mat")["c"][0, 0]
    print("matlab fixture | a", float(c["a"][0, 0]), "sigma", float(c["sigma"][0, 0]), "alpha", float(c["alphaL1"][0, 0]), "| stop", r["stop"],
          "| r_sim finite", np.isfinite(r["r_sim"]), "| fp2a present", r["fp2a"] is not None, "| secs %.0f" % (time.time() - t0), flush=True)
    print("vh reference loader | records", len(load_vh()), flush=True)
    g = np.random.default_rng(2); recs = []
    for k in list(K) + list(VH_ALPHA):
        for s in SUBJ:
            for c in MODELS:
                for rr in STREAMS:
                    if k == "KR" and s == 1 and c == 1 and rr > 3:
                        continue                                                   # KR: < 6 fits -> not analysable
                    good = k == "K3a"                                              # K3a reproduces (D ~ 0.03); others D ~ 0.35
                    recs.append(dict(subj=s, K=k, coop=c, r=rr, r_sim=(0.74 if c == 0 else 0.51) + (0 if good else 0.15) + g.normal(0, 0.01),
                                     r_last=0.6 + g.normal(0, 0.05), fp0=0.25 + (0 if good else 0.05) + g.normal(0, 0.005), fp2a=0.2, iterations=300))
    o = analyze_records(recs)
    print("analysis fixture |", {k: (round(v["D"], 3) if v.get("D") is not None else v["reading"]) for k, v in o["D"].items()}, "| best", o["best_a025"],
          "| VJ2b", o["VJ2b_trigger"], "|", o["reading"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("arg", nargs="?"); ap.add_argument("--procs", type=int, default=24); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(fixture=stage_fixture, run=lambda: stage_run(a.procs), analyze=lambda: stage_analyze(a.arg))[a.stage]()
