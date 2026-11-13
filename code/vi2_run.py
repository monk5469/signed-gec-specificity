"""Frequency control (Sections 2.3.2, 3.2; Supplementary Note 15): refits of acc and gsr with mot frequencies and of mot with acc
frequencies, alongside own-frequency refits, with the PPC inputs, seeds and evaluation keys."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import sys, json, time, hashlib, argparse, tempfile, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

ROOT = DATA_ROOT; P33 = ROOT + "results/p33/"; OUT = P33 + "vhvi/vi2/"; AMEND = (ROOT + "docs_r1/179a_vhvi_amendment_A1.md", ROOT + "docs_r1/179b_vhvi_amendment_A2.md")
CELLS = {"acc|f_mot": ("acc", "mot"), "gsr|f_mot": ("gsr", "mot"), "mot|f_acc": ("mot", "acc"),
         "acc|f_acc": ("acc", "acc"), "gsr|f_gsr": ("gsr", "gsr"), "mot|f_mot": ("mot", "mot")}
OWN = {"acc": "acc|f_acc", "gsr": "gsr|f_gsr", "mot": "mot|f_mot"}
MODELS = ("signed", "coop"); DEMO = {"100307", "100408", "101107"}; BUDGET_H = 48.0; MIN_N = 86; EQ = 0.023


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rules():
    import vhvi_rules
    vhvi_rules.check("vi2", (os.path.abspath(__file__),) + AMEND)


def ppc(cond):
    import luppi_r4e_ppc as W
    X = W.X; W.point(cond); fsha, fr = X.verify_freeze("e5")
    return X.OUT, fr["files"], {j["tag"]: j["subject_index"] for j in X.job_list("e5")}, [j for j in X.job_list("e5")]


def e5(cond):
    return P33 + f"r4e_ppc_{cond}/e5/"


def stage_prepare():
    rules()
    if os.path.exists(OUT + "vi_manifest.json"):
        raise SystemExit("prepared once")
    os.makedirs(OUT + "jobs", exist_ok=True); files = {}; J = []
    for cell, (data_c, freq_c) in CELLS.items():
        for c in (data_c, freq_c):
            od, fr, idx, _ = ppc(c)
            rels = ["e5/freq_REST1.npy", "e5/SC.npy"] + [f"e5/{s}_REST1_{k}.npy" for s in idx if s not in DEMO for k in ("input", "heldout")]
            for rel in rels:
                key = f"ext:{rel}" if f"ext:{rel}" in fr else rel
                if key in fr and sha(od + rel) != fr[key]:
                    raise SystemExit(f"{c} {rel} differs from the PPC freeze")
                files[od + rel] = sha(od + rel)
        _, _, idx, _ = ppc(data_c)
        for s in sorted(idx):
            if s not in DEMO:
                for m in MODELS:
                    J.append(dict(cell=cell, data=data_c, freq=freq_c, tag=s, subject_index=idx[s], model=m, file=OUT + f"jobs/{cell.replace('|', '_')}_{s}_{m}.json"))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), jobs=J, files=files), open(OUT + "vi_manifest.json", "x"), indent=1)
    print("prepared | jobs", len(J), "| files", len(files), flush=True)


def job_list():
    return json.load(open(OUT + "vi_manifest.json"))["jobs"]


_SIM = {}


def run_job(j, stop_at, iters=None):
    """As luppi_r4e_ext.run_job for PPC REST1, except that f is the frequency vector of j['freq']."""
    import jax, luppi_hopf as L, luppi_r4e_lib as Lib
    if "s" not in _SIM:
        _SIM["s"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["s"]; cfg = dict(L.HEADER)
    if iters:
        cfg["max_iterations"] = iters
    SC = np.load(e5(j["data"]) + "SC.npy"); inp = np.load(e5(j["data"]) + f"{j['tag']}_REST1_input.npy"); f = np.load(e5(j["freq"]) + "freq_REST1.npy")
    hop = L.prep_emp(np.load(e5(j["data"]) + f"{j['tag']}_REST1_heldout.npy")); base = 9_000_000 + 1000 * j["subject_index"]
    t0 = time.time(); C, info = L.fit(inp, SC, f, cfg, seed=11, cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm)
    if not (np.all(np.isfinite(C)) and np.isfinite(mse)):
        raise ValueError("non-finite fit")
    return C, dict(j, state="done", secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"],
                   fc_corr_last=info["fc_corr_last"], mse_ho=float(mse), C_sha=sha_arr(C))


def stage_freeze():
    rules(); m = json.load(open(OUT + "vi_manifest.json")); bad = [p for p, h in m["files"].items() if sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} inputs changed")
    if os.path.exists(OUT + "vi_freeze.json"):
        raise SystemExit("frozen once")
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), manifest=sha(OUT + "vi_manifest.json"), code=sha(os.path.abspath(__file__)), n_jobs=len(m["jobs"])),
              open(OUT + "vi_freeze.json", "x"), indent=1); print("frozen | jobs", len(m["jobs"]), flush=True)


def safe(args):
    j, deadline = args
    if os.path.exists(j["file"]) and json.load(open(j["file"])).get("state") == "done":
        return "done"
    err = ""
    for attempt in (1, 2):
        if time.time() > deadline:
            json.dump(dict(j, state="cancelled"), open(j["file"], "w")); return "cancelled"
        try:
            C, r = run_job(j, deadline); np.save(j["file"].replace(".json", "_C.npy"), C); json.dump(r, open(j["file"], "w"), indent=1, default=float); return "done"
        except Exception:
            err = traceback.format_exc()[-800:]
    json.dump(dict(j, state="failed_twice", error=err), open(j["file"], "w"), indent=1); return "failed"


def stage_run(procs):
    rules(); fr = json.load(open(OUT + "vi_freeze.json"))
    if fr["manifest"] != sha(OUT + "vi_manifest.json") or fr["code"] != sha(os.path.abspath(__file__)):
        raise SystemExit("freeze mismatch")
    dl = OUT + "vi_deadline.json"
    if not os.path.exists(dl):
        json.dump(dict(deadline=time.time() + BUDGET_H * 3600), open(dl, "w"))
    J = sorted(job_list(), key=lambda j: (j["cell"] not in ("acc|f_mot", "mot|f_mot"), j["tag"], j["model"]))
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


def endpoints(E):
    """E[(unit, tag, model)] = (record, C, SC) -> per-unit dicts of FP, P2, dr keyed by participant (both models required)."""
    out = {}
    for u in {u for u, _, _ in E}:
        tags = sorted({t for uu, t, _ in E if uu == u and (u, t, "signed") in E and (u, t, "coop") in E}); d = dict(fp={}, p2={}, dr={}, dr_last={})
        for t in tags:
            (rs, Cs, SC), (rc, _, _) = E[(u, t, "signed")], E[(u, t, "coop")]; m = (SC != 0) & ~np.eye(len(SC), dtype=bool)
            d["fp"][t] = float((Cs[m] < 0).mean()); d["p2"][t] = rs["mse_ho"] - rc["mse_ho"]; d["dr"][t] = rs["fc_corr"] - rc["fc_corr"]
            if "fc_corr_last" in rs and "fc_corr_last" in rc:
                d["dr_last"][t] = rs["fc_corr_last"] - rc["fc_corr_last"]
        out[u] = d
    return out


def paired(A, B, rng):
    ks = [t for t in A if t in B]
    return dict(boot([A[t] - B[t] for t in ks], rng), n_pairs=len(ks))


def analyze_endpoints(D, seed=20261027):
    rng = np.random.default_rng(seed); o = dict(cells={u: {k: boot(list(v.values()), rng) for k, v in d.items()} for u, d in D.items()})
    def crit(name, a, b, ep, rule):
        if a not in D or b not in D:
            o[name] = dict(reading="not analysable", n_pairs=0); return
        r = paired(D[a][ep], D[b][ep], rng)
        if r["n_pairs"] < MIN_N:
            r["reading"] = "not analysable"
        elif rule == "lower>0.10":
            r["reading"] = bool(r["ci95"][0] > 0.10)
        elif rule == "upper<0":
            r["reading"] = bool(r["ci95"][1] < 0)
        else:
            lo, hi = r["ci95"]; r["reading"] = "equivalent" if (lo > -EQ and hi < EQ) else "outside" if (lo > EQ or hi < -EQ) else "undetermined"
        o[name] = r
    crit("VI-T", "acc|f_mot", "mot|f_mot", "fp", "lower>0.10"); crit("VI-G", "gsr|f_mot", "mot|f_mot", "fp", "lower>0.10")
    crit("VI-P", "acc|f_mot", "mot|f_mot", "p2", "upper<0"); crit("VI-E", "acc|f_mot", "acc|f_acc", "fp", "equiv")
    crit("VI-E_gsr", "gsr|f_mot", "gsr|f_gsr", "fp", "equiv"); crit("VI-E_mot", "mot|f_acc", "mot|f_mot", "fp", "equiv")
    crit("VI-T_reverse", "acc|f_acc", "mot|f_acc", "fp", "lower>0.10")
    for a, b in (("acc|f_mot", "acc|f_acc"), ("gsr|f_mot", "gsr|f_gsr"), ("mot|f_acc", "mot|f_mot"), ("acc|f_acc", "acc"), ("gsr|f_gsr", "gsr"), ("mot|f_mot", "mot")):
        o[f"desc|{a}-{b}|p2"] = paired(D[a]["p2"], D[b]["p2"], rng) if a in D and b in D else None
        o[f"desc|{a}-{b}|dr"] = paired(D[a]["dr"], D[b]["dr"], rng) if a in D and b in D else None
    return o


def load_all():
    import luppi_r4e_ppc as W
    E = {}; SCref = None
    for j in job_list():                                                   # frequency control cells
        try:
            r = json.load(open(j["file"]))
        except Exception:
            continue
        if r.get("state") == "done":
            C = np.load(j["file"].replace(".json", "_C.npy"))
            if sha_arr(C) == r["C_sha"]:
                E[(j["cell"], j["tag"], j["model"])] = (r, C, np.load(e5(j["data"]) + "SC.npy"))
    for c in ("acc", "gsr", "mot"):                                        # frozen PPC comparators
        _, _, _, J = ppc(c)
        for j in J:
            if j["session"] != "REST1" or j["tag"] in DEMO or not os.path.exists(j["file"]):
                continue
            r = json.load(open(j["file"]))
            if "C_sha" not in r or "mse_ho" not in r:
                continue
            C = np.load(j["file"].replace(".json", "_C.npy"))
            if sha_arr(C) == r["C_sha"] and np.isfinite(r["mse_ho"]):
                E[(c, j["tag"], j["model"])] = (r, C, np.load(e5(c) + "SC.npy"))
    return E


def reproduction(E):
    """A2, descriptive: refitted own-frequency cells against the frozen PPC fits (same participant, model and seed)."""
    out = {}
    for own, cell in OWN.items():
        d = [float(np.abs(E[(u, t, m)][1] - E[(own, t, m)][1]).max()) for (u, t, m) in E if u == cell and (own, t, m) in E]
        out[cell] = dict(n=len(d), n_identical=int(sum(x == 0 for x in d)), median_max_abs_dC=float(np.median(d)) if d else None,
                         max_max_abs_dC=float(max(d)) if d else None, n_above_1e_12=int(sum(x > 1e-12 for x in d)))
    return out


def stage_analyze(path):
    rules(); E = load_all(); o = analyze_endpoints(endpoints(E)); o["n_records"] = len(E); o["reproduction_vs_frozen_ppc"] = reproduction(E)
    json.dump(o, open(path, "x"), indent=1, default=float)
    for k in ("VI-T", "VI-G", "VI-P", "VI-E", "VI-E_gsr", "VI-E_mot", "VI-T_reverse"):
        print(k, o[k].get("reading"), o[k].get("median"), o[k].get("ci95"), "n", o[k].get("n_pairs"), flush=True)


def stage_fixture():
    """Before code registration: a 120-iteration f-swapped fit and the analysis on synthetic endpoints covering every reading."""
    _, _, idx, _ = ppc("acc"); tag = sorted(t for t in idx if t not in DEMO)[0]
    j = dict(cell="acc|f_mot", data="acc", freq="mot", tag=tag, subject_index=idx[tag], model="signed")
    C, r = run_job(j, time.time() + 3600, iters=120); print("f-swap fixture | iterations", r["iterations"], "| mse finite", np.isfinite(r["mse_ho"]), flush=True)
    g = np.random.default_rng(3); tags = [f"s{i}" for i in range(95)]
    mk = lambda mu, sd, n=95: {t: mu + g.normal(0, sd) for t in tags[:n]}
    D = {"mot|f_mot": dict(fp=mk(0.02, 0.01), p2=mk(0.0, 0.001), dr=mk(0.01, 0.01), dr_last={}),
         "acc|f_acc": dict(fp=mk(0.27, 0.01), p2=mk(-0.01, 0.001), dr=mk(0.25, 0.01), dr_last={}),
         "acc|f_mot": dict(fp=mk(0.27, 0.01), p2=mk(-0.01, 0.001), dr=mk(0.25, 0.01), dr_last={}),      # T, P true; E equivalent
         "gsr|f_gsr": dict(fp=mk(0.37, 0.01), p2=mk(-0.02, 0.001), dr=mk(0.3, 0.01), dr_last={}),
         "gsr|f_mot": dict(fp=mk(0.42, 0.01), p2=mk(-0.02, 0.001), dr=mk(0.3, 0.01), dr_last={}),       # E_gsr outside
         "mot|f_acc": dict(fp=mk(0.03, 0.08, 80), p2=mk(0.0, 0.001, 80), dr=mk(0.01, 0.01, 80), dr_last={})}   # < 86 -> not analysable
    o = analyze_endpoints(D)
    I = np.eye(3); E = {("acc|f_acc", "a", "signed"): (None, I, None), ("acc", "a", "signed"): (None, I, None), ("acc|f_acc", "b", "signed"): (None, I + 1e-16, None),
         ("acc", "b", "signed"): (None, I, None)}
    print("reproduction fixture |", reproduction(E)["acc|f_acc"], flush=True)
    print("analysis fixture |", {k: o[k]["reading"] for k in ("VI-T", "VI-G", "VI-P", "VI-E", "VI-E_gsr", "VI-E_mot", "VI-T_reverse")}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("arg", nargs="?"); ap.add_argument("--procs", type=int, default=60); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(fixture=stage_fixture, prepare=stage_prepare, freeze=stage_freeze, run=lambda: stage_run(a.procs), analyze=lambda: stage_analyze(a.arg))[a.stage]()
