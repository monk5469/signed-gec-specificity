"""Bifurcation-parameter check on the PPC data (Sections 2.4.3, 3.7.5; Supplementary Note 14): the operator contrast at a = -0.025
under K1a and K4a."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import sys, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; OUT = P33 + "vj/vj2/"; CONDS = ("mot", "acc", "gsr"); MODELS = ("signed", "coop")
KC = {"K1a": dict(sigma=0.01, max_iterations=5000, eta_fc=2e-4, eta_tau=4e-5, alpha=0.0, max_C=0.10, conv=1e-3, tau=1, a=-0.025, burnin=3000.0),
      "K4a": dict(sigma=0.001, max_iterations=5000, eta_fc=1e-3, eta_tau=1e-4, alpha=0.0, max_C=0.15, conv=1e-3, tau=1, a=-0.025, burnin=3000.0)}
DEMO = {"100307", "100408", "101107"}; BUDGET_H = 48.0; MIN_N = 86


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


AMEND = DATA_ROOT + "docs_r1/183a_vj_amendment_A1.md"


def rules():
    import vj_rules
    vj_rules.check("vj2a", (os.path.abspath(__file__), AMEND))


def ppc(cond):
    import r23_run as R
    return R.ppc_inputs(cond)                       # (out_dir, freeze files, {tag: subject_index}, freeze sha)


def stage_prepare():
    rules()
    if os.path.exists(OUT + "vj_manifest.json"):
        raise SystemExit("prepared once")
    os.makedirs(OUT + "jobs", exist_ok=True); files = {}; J = []
    for c in CONDS:
        od, fr, idx, fsha = ppc(c)
        for rel in ["e5/freq_REST1.npy", "e5/SC.npy"] + [f"e5/{s}_REST1_{k}.npy" for s in idx if s not in DEMO for k in ("input", "heldout")]:
            key = f"ext:{rel}" if f"ext:{rel}" in fr else rel
            if key in fr and sha(od + rel) != fr[key]:
                raise SystemExit(f"{c} {rel} differs from the PPC freeze")
            files[od + rel] = sha(od + rel)
        for k in KC:
            for s in sorted(idx):
                if s in DEMO:
                    continue
                for m in MODELS:
                    J.append(dict(K=k, cond=c, tag=s, subject_index=idx[s], model=m, file=OUT + f"jobs/{k}_{c}_{s}_{m}.json"))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), configs=KC, jobs=J, files=files), open(OUT + "vj_manifest.json", "x"), indent=1)
    print("prepared | jobs", len(J), "| files", len(files), flush=True)


def job_list():
    return json.load(open(OUT + "vj_manifest.json"))["jobs"]


def paths(j):
    od = P33 + f"r4e_ppc_{j['cond']}/e5/"; return od + f"{j['tag']}_REST1_input.npy", od + f"{j['tag']}_REST1_heldout.npy", od + "freq_REST1.npy", od + "SC.npy"


_SIM = {}


def run_job(j, stop_at, iters=None):
    import jax, luppi_hopf as L, luppi_r4e_lib as Lib
    if "s" not in _SIM:
        _SIM["s"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["s"]; fi, fh, ff, fs = paths(j); x = np.load(fi); hop = L.prep_emp(np.load(fh)); f = np.load(ff); SC = np.load(fs); cfg = dict(KC[j["K"]])
    if iters:
        cfg["max_iterations"] = iters
    t0 = time.time(); C, info = L.fit(x, SC, f, cfg, seed=11, cooperative_only=j["model"] == "coop", lag="matlab", sim=sim, stop_at=stop_at)
    base = 9_000_000 + 1000 * j["subject_index"]; sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
    FCm = np.mean([L.fc(s) for s in sims], 0); mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, np.mean([L.lag_fc(s, 1, "matlab") for s in sims], 0))
    if not (np.all(np.isfinite(C)) and np.isfinite(mse)):
        raise ValueError("non-finite fit")
    if not iters:
        np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, state="done", secs=round(time.time() - t0, 1), iterations=info["iterations"], stop=info["stop"], fc_corr_last=info["fc_corr_last"],
                fc_corr=info["fc_corr"], mse_ho=float(mse), C_sha=sha_arr(C))


def stage_freeze():
    rules(); m = json.load(open(OUT + "vj_manifest.json")); bad = [p for p, h in m["files"].items() if sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} inputs changed")
    if os.path.exists(OUT + "vj_freeze.json"):
        raise SystemExit("frozen once")
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), manifest=sha(OUT + "vj_manifest.json"), code=sha(os.path.abspath(__file__)), n_jobs=len(m["jobs"])),
              open(OUT + "vj_freeze.json", "x"), indent=1); print("frozen | jobs", len(m["jobs"]), flush=True)


def safe(args):
    j, deadline = args
    if os.path.exists(j["file"]) and json.load(open(j["file"])).get("state") == "done":
        return "done"
    err = ""
    for attempt in (1, 2):
        if time.time() > deadline:
            json.dump(dict(j, state="cancelled"), open(j["file"], "w")); return "cancelled"
        try:
            r = run_job(j, deadline); json.dump(r, open(j["file"], "w"), indent=1, default=float); return "done"
        except Exception:
            err = traceback.format_exc()[-800:]
    json.dump(dict(j, state="failed_twice", error=err), open(j["file"], "w"), indent=1); return "failed"


def stage_run(procs):
    rules(); fr = json.load(open(OUT + "vj_freeze.json"))
    if fr["manifest"] != sha(OUT + "vj_manifest.json") or fr["code"] != sha(os.path.abspath(__file__)):
        raise SystemExit("freeze mismatch")
    dl = OUT + "vj_deadline.json"
    if not os.path.exists(dl):
        json.dump(dict(deadline=time.time() + BUDGET_H * 3600), open(dl, "w"))
    J = sorted(job_list(), key=lambda j: (j["K"] != "K1a", j["cond"] == "gsr", j["tag"], j["model"]))
    with mp.Pool(procs) as p:
        st = p.map(safe, [(j, json.load(open(dl))["deadline"]) for j in J], chunksize=1)
    print({s: st.count(s) for s in set(st)}, flush=True)


# ---------------------------------------------------------------- analysis
def boot(v, rng, B=2000):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    bs = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(B)]
    return dict(n=int(v.size), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def endpoints(A, SC):
    """A[(config, cond, tag, model)] = (record, C) -> E[config][cond] = dict(fp, p2, r) keyed by tag (both models required)."""
    m = (SC != 0) & ~np.eye(len(SC), dtype=bool); E = {}
    for (k, c, t, mo) in A:
        if mo != "signed" or (k, c, t, "coop") not in A:
            continue
        (rs, Cs), (rc, _) = A[(k, c, t, "signed")], A[(k, c, t, "coop")]
        d = E.setdefault(k, {}).setdefault(c, dict(fp={}, p2={}, r={}))
        d["fp"][t] = float((Cs[m] < 0).mean()); d["p2"][t] = rs["mse_ho"] - rc["mse_ho"]; d["r"][t] = rs.get("fc_corr_last", float("nan"))     # A1: absent in the frozen PPC records
    return E


def analyze_endpoints(E, ref=None, seed=20261029):
    rng = np.random.default_rng(seed); o = {}
    def pdiff(A_, B_):
        ks = [t for t in A_ if t in B_]; return dict(boot([A_[t] - B_[t] for t in ks], rng), n_pairs=len(ks))
    for k in KC:
        e = E.get(k, {}); ok = {c: c in e for c in CONDS}; r = dict(cells={c: {ep: boot(list(e[c][ep].values()), rng) for ep in ("fp", "p2", "r")} for c in CONDS if ok[c]})
        r["share_signed_better"] = {c: float(np.mean([v < 0 for v in e[c]["p2"].values()])) for c in CONDS if ok[c]}
        for nm, (a, b, ep, rule) in {"VJ-T": ("acc", "mot", "fp", "lower"), "VJ-G": ("gsr", "mot", "fp", "lower"), "VJ-P": ("acc", "mot", "p2", "upper")}.items():
            if not (ok[a] and ok[b]):
                r[nm] = dict(reading="not analysable", n_pairs=0); continue
            x = pdiff(e[a][ep], e[b][ep])
            x["reading"] = "not analysable" if x["n_pairs"] < MIN_N else bool(x["ci95"][0] > 0.10) if rule == "lower" else bool(x["ci95"][1] < 0)
            r[nm] = x
        if ref and k in ref:
            r["vs_a002"] = {f"{c}|{ep}": pdiff(e[c][ep], ref[k][c][ep]) for c in CONDS if ok[c] and c in ref[k] for ep in ("fp", "p2", "r")}
        o[k] = r
    return o


def load_ref():
    """a = -0.02 comparators: K1 = frozen PPC REST1 fits; K4 = the K4 public-configuration check fits (vavf/va4). Hash-checked."""
    import luppi_r4e_ppc as W
    A = {}
    for c in CONDS:
        W.point(c)
        for j in W.X.job_list("e5"):
            if j["session"] != "REST1" or j["tag"] in DEMO or not os.path.exists(j["file"]):
                continue
            r = json.load(open(j["file"]))
            if "C_sha" in r and "mse_ho" in r:
                C = np.load(j["file"].replace(".json", "_C.npy"))
                if sha_arr(C) == r["C_sha"] and np.isfinite(r["mse_ho"]):
                    A[("K1a", c, j["tag"], j["model"])] = (r, C)
        for t in os.listdir(P33 + "vavf/va4/jobs/"):
            if t.startswith(c + "_") and t.endswith(".json"):
                r = json.load(open(P33 + "vavf/va4/jobs/" + t))
                if r.get("state") == "done":
                    C = np.load(P33 + "vavf/va4/jobs/" + t.replace(".json", "_C.npy"))
                    if sha_arr(C) == r["C_sha"]:
                        A[("K4a", c, r["tag"], r["model"])] = (r, C)
    return A


def stage_analyze(path):
    rules(); SC = np.load(P33 + "r4e_ppc_acc/e5/SC.npy"); A = {}
    for j in job_list():
        try:
            r = json.load(open(j["file"]))
        except Exception:
            continue
        if r.get("state") == "done":
            C = np.load(j["file"].replace(".json", "_C.npy"))
            if sha_arr(C) == r["C_sha"]:
                A[(j["K"], j["cond"], j["tag"], j["model"])] = (r, C)
    ref = endpoints(load_ref(), SC); o = analyze_endpoints(endpoints(A, SC), ref); o["n_records"] = len(A)
    json.dump(o, open(path, "x"), indent=1, default=float)
    for k in KC:
        print("VJ2", k, {n: (o[k][n].get("reading"), o[k][n].get("median"), o[k][n].get("ci95"), o[k][n].get("n_pairs")) for n in ("VJ-T", "VJ-G", "VJ-P")}, flush=True)
        print("   cells fp", {c: o[k]["cells"][c]["fp"].get("median") for c in o[k]["cells"]}, "| vs a-0.02 fp", {kk: (v.get("median"), v.get("ci95")) for kk, v in o[k].get("vs_a002", {}).items() if kk.endswith("|fp")}, flush=True)


def stage_fixture():
    """Before code registration: input paths for every condition, one 120-iteration fit per configuration, the reference loader (counts
    only) and the analysis on synthetic endpoints covering every reading. No endpoint is kept."""
    import tempfile
    idx = ppc("acc")[2]; tag = sorted(t for t in idx if t not in DEMO)[0]
    for c in CONDS:
        print(c, [os.path.exists(p) for p in paths(dict(cond=c, tag=tag))], flush=True)
    for k in KC:
        j = dict(K=k, cond="acc", tag=tag, subject_index=idx[tag], model="signed", file=tempfile.mkdtemp() + "/fx.json")
        r = run_job(j, time.time() + 3600, iters=120); print("fixture", k, "| iterations", r["iterations"], "| secs", r["secs"], "| mse finite", np.isfinite(r["mse_ho"]), flush=True)
    A = load_ref(); print("reference loader |", {k: sum(1 for x in A if x[0] == k) for k in KC}, flush=True)
    SC = np.load(P33 + "r4e_ppc_acc/e5/SC.npy"); R = endpoints(A, SC)                 # A1: reference endpoints, counts only
    print("reference endpoints |", {k: {c: (len(R[k][c]["fp"]), sum(np.isfinite(v) for v in R[k][c]["r"].values())) for c in R[k]} for k in R}, flush=True)
    g = np.random.default_rng(4); tags = [f"s{i}" for i in range(95)]
    mk = lambda mu, sd, n=95: {t: mu + g.normal(0, sd) for t in tags[:n]}
    E = {"K1a": {"mot": dict(fp=mk(0.02, 0.01), p2=mk(0.0, 0.001), r=mk(0.3, 0.01)), "acc": dict(fp=mk(0.27, 0.01), p2=mk(-0.01, 0.001), r=mk(0.6, 0.01)),
                 "gsr": dict(fp=mk(0.08, 0.05), p2=mk(-0.02, 0.001), r=mk(0.7, 0.01))},                      # T, P true; G false
         "K4a": {"mot": dict(fp=mk(0.02, 0.01, 80), p2=mk(0.0, 0.001, 80), r=mk(0.3, 0.01, 80)), "acc": dict(fp=mk(0.27, 0.01), p2=mk(0.01, 0.001), r=mk(0.6, 0.01))}}  # < 86 -> not analysable; gsr missing
    o = analyze_endpoints(E, {"K1a": {"mot": dict(fp=mk(0.02, 0.01), p2=mk(0.0, 0.001), r=mk(0.3, 0.01))}})
    print("analysis fixture |", {k: {n: o[k][n]["reading"] for n in ("VJ-T", "VJ-G", "VJ-P")} for k in KC}, "| vs_a002 keys", list(o["K1a"].get("vs_a002", {})), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("arg", nargs="?"); ap.add_argument("--procs", type=int, default=36); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(fixture=stage_fixture, prepare=stage_prepare, freeze=stage_freeze, run=lambda: stage_run(a.procs), analyze=lambda: stage_analyze(a.arg))[a.stage]()
