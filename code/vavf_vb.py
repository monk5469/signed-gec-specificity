"""Tissue-pool check (Sections 2.6, 3.4.1, 3.5; Supplementary Note 7): COOP and CMPN truths of Simulation 6 under tissue-pool and
surrogate global removal, and CMPN negative-edge recovery."""
import sys, os, json, time, hashlib, argparse, shutil, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; OUT = P33 + "vavf/vb/"; R1 = P33 + "r1/"; ACC = P33 + "r4e_ppc_acc/"
ARMS = {"N0": "A0", "S": "A1", "GS": "A2", "TP": "A3cal"}; TYPES = ("COOP", "CMPN"); MODELS = ("signed", "coop"); BUDGET_H = 24.0
FX = dict(dir=P33 + "vavf/vb_fixture/", n=2, iters=120)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rules():
    import vavf_rules
    vavf_rules.check("vb", ("vavf_vb.py",))


def cmpn(C, SC, k):
    import luppi_r4e_lib as Lib, luppi_r4e_e10c as E10
    nets = E10.net_info()[0]; pos = Lib.support(SC) & (C > 0); pool = np.flatnonzero((pos & (nets[:, None] != nets[None, :])).ravel())
    n = int(round(0.25 * pos.sum())); pick = np.random.default_rng([7000 + k, 9]).choice(pool, min(n, pool.size), replace=False)
    Cf = C.ravel().copy(); Cf[pick] = -Cf[pick]; return Cf.reshape(C.shape), dict(n_neg=int(pick.size), n_target=n, pool=int(pool.size))


def gen(args):
    """CMPN series for truth k with the frozen COOP calibration of Simulation 6 (s01, g 0)."""
    k, C, out = args
    import r1_run as R, luppi_r4e_e10c as E10, luppi_r4e_e11 as E11
    try:
        SC = R.group_sc(); f = np.load(ACC + "e5/freq_REST1.npy"); cal = json.load(open(R.MAN))
        info = next(u for u in cal["calibration"] if u["k"] == k and u["sg"] == "s01" and u["g"] == 0)
        if not info.get("valid"):
            return dict(k=k, valid=False, reason="COOP calibration invalid in Simulation 6")
        c, sig, nr = info["c"], info["sigma"], info["a3_mean4"]["noise_rel"]; s_obs = 6000 + k
        Cn, meta = cmpn(C, SC, k); files = {}; desc = {}
        for held, nm in ((False, "target"), (True, "heldout")):
            xco = R.neural(C, f, k, 0, held, "s01"); xcm = R.neural(Cn, f, k, 0, held, "s01")
            y0, xi0 = E11.observe(xcm, s_obs, held, 0.0, ref=xco); y49, xi49 = E11.observe(xcm, s_obs, held, c, ref=xco)
            ser = dict(N0=E10.remove(y0, None, xi0), S=E10.remove(y49, None, xi49), GS=E10.remove(y49, sig, xi49),
                       TP=R.regress(y49, R.pool_regressors(R.pools(xcm, s_obs, held, c, nr), "mean4")))
            for a, r in ser.items():
                if not (np.all(np.isfinite(r)) and np.all(r.std(1) > 0)):
                    return dict(k=k, valid=False, reason=f"non-finite CMPN {a} {nm}")
                p = out + f"inp/k{k}_CMPN_{a}_{nm}.npy"; np.save(p, r); files[p] = sha(p)
                if nm == "target":
                    desc[a] = E10.desc(r)
        p = out + f"truth/k{k}_CMPN_C.npy"; np.save(p, Cn); files[p] = sha(p)
        return dict(k=k, valid=True, cmpn=meta, files=files, input=desc)
    except Exception:
        return dict(k=k, valid=False, reason=traceback.format_exc()[-600:])


def coop_files(k):
    return {a: {nm: R1 + f"inp/k{k}_s01_g0_{A}_{nm}.npy" for nm in ("target", "heldout")} for a, A in ARMS.items()}


def stage_prepare(procs, fixture=False):
    import r1_run as R
    out = FX["dir"] if fixture else OUT
    if not fixture:
        rules()
        if os.path.exists(out + "vb_manifest.json"):
            raise SystemExit("prepared once")
    else:
        shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out + "inp", exist_ok=True); os.makedirs(out + "truth", exist_ok=True)
    T, fsha = R.truths(False); ks = sorted(T)[:FX["n"]] if fixture else sorted(T); fr1 = json.load(open(R1 + "r1_freeze.json"))["files"]
    with mp.Pool(procs) as p:
        res = p.map(gen, [(k, T[k][1], out) for k in ks], chunksize=1)
    files = {}; units = []
    for r in res:
        units.append({q: v for q, v in r.items() if q != "files"}); files.update(r.get("files", {}))
        if r.get("valid"):
            for a, d in coop_files(r["k"]).items():
                for nm, p in d.items():
                    if fr1.get(p) != sha(p):
                        raise SystemExit(f"Simulation 6 input changed or missing from its freeze: {p}")
                    files[p] = fr1[p]
            files[R1 + f"truth/k{r['k']}_COOP_C.npy"] = sha(R1 + f"truth/k{r['k']}_COOP_C.npy")
    for p in (ACC + "e5/SC.npy", ACC + "e5/freq_REST1.npy"):
        files[p] = sha(p)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), truths={k: T[k][0] for k in ks}, units=units, files=files, r1_freeze=sha(R1 + "r1_freeze.json")),
              open(out + "vb_manifest.json", "w" if fixture else "x"), indent=1, default=float)
    print("prepared | valid", sum(u.get("valid", False) for u in units), "of", len(units), "|", [u.get("reason", "")[:80] for u in units if not u.get("valid")], flush=True)


def job_list(out=OUT):
    man = json.load(open(out + "vb_manifest.json")); J = []
    for u in man["units"]:
        if not u.get("valid"):
            continue
        for t in TYPES:
            for a in ARMS:
                for m in MODELS:
                    j = dict(k=u["k"], type=t, arm=a, model=m); j["file"] = out + f"jobs/k{u['k']}_{t}_{a}_{m}.json"; J.append(j)
    return J


def job_files(j, out=OUT):
    if j["type"] == "COOP":
        d = coop_files(j["k"])[j["arm"]]; return d["target"], d["heldout"], R1 + f"truth/k{j['k']}_COOP_C.npy"
    return tuple(out + f"inp/k{j['k']}_CMPN_{j['arm']}_{nm}.npy" for nm in ("target", "heldout")) + (out + f"truth/k{j['k']}_CMPN_C.npy",)


_SIM = {}


def run_job(j, stop_at, out=OUT, iters=None):
    import jax, luppi_hopf as L, luppi_r4e_lib as Lib
    if "s" not in _SIM:
        _SIM["s"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["s"]; fi, fh, _ = job_files(j, out); x = np.load(fi); hop = L.prep_emp(np.load(fh)); f = np.load(ACC + "e5/freq_REST1.npy")
    SC = np.load(ACC + "e5/SC.npy"); cfg = dict(L.HEADER)
    if iters:
        cfg["max_iterations"] = iters
    t0 = time.time(); C, info = L.fit(x, SC, f, cfg, seed=11, cooperative_only=j["model"] == "coop", lag="matlab", sim=sim, stop_at=stop_at)
    base = 44_000_000 + 1000 * j["k"] + 100 * TYPES.index(j["type"]) + 10 * list(ARMS).index(j["arm"]); sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), np.mean([L.fc(s) for s in sims], 0), np.mean([L.lag_fc(s, 1, "matlab") for s in sims], 0))
    np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, state="done", secs=round(time.time() - t0, 1), iterations=info["iterations"], stop=info["stop"], fc_corr=info["fc_corr"], mse_ho=float(mse), C_sha=sha_arr(C))


def stage_freeze():
    rules(); man = OUT + "vb_manifest.json"; m = json.load(open(man))
    bad = [p for p, h in m["files"].items() if sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} inputs changed, e.g. {bad[:2]}")
    if os.path.exists(OUT + "vb_freeze.json"):
        raise SystemExit("frozen once")
    J = job_list(); json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), manifest=sha(man), n_jobs=len(J), code=sha(os.path.abspath(__file__))),
                              open(OUT + "vb_freeze.json", "x"), indent=1); print("frozen | jobs", len(J), flush=True)


def safe(args):
    j, deadline = args
    if os.path.exists(j["file"]) and json.load(open(j["file"])).get("state") == "done":
        return "done"
    for attempt in (1, 2):
        try:
            r = run_job(j, deadline); json.dump(r, open(j["file"], "w"), indent=1, default=float); return "done"
        except Exception:
            err = traceback.format_exc()[-800:]
    json.dump(dict(j, state="failed_twice", error=err), open(j["file"], "w"), indent=1); return "failed"


def stage_run(procs):
    rules(); fr = json.load(open(OUT + "vb_freeze.json"))
    if fr["manifest"] != sha(OUT + "vb_manifest.json") or fr["code"] != sha(os.path.abspath(__file__)):
        raise SystemExit("freeze mismatch")
    os.makedirs(OUT + "jobs", exist_ok=True); dl = OUT + "vb_deadline.json"
    if not os.path.exists(dl):
        json.dump(dict(deadline=time.time() + BUDGET_H * 3600), open(dl, "w"))
    J = job_list(); J.sort(key=lambda j: (j["arm"] != "TP", j["type"], j["k"], j["model"]))
    with mp.Pool(procs) as p:
        st = p.map(safe, [(j, json.load(open(dl))["deadline"]) for j in J], chunksize=1)
    print({s: st.count(s) for s in set(st)}, flush=True)


def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = y.sum(); n0 = y.size - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def boot(v, rng, B=2000):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    bs = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(B)]
    return dict(n=int(v.size), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def stage_analyze(path):
    import luppi_hopf as L, luppi_r4e_e10c as E10
    rules(); SC = np.load(ACC + "e5/SC.npy"); m = (SC != 0) & ~np.eye(100, dtype=bool); rng = np.random.default_rng(20261025); R = {}
    for j in job_list():
        try:
            r = json.load(open(j["file"]))
        except Exception:
            continue
        if r.get("state") != "done":
            continue
        C = np.load(j["file"].replace(".json", "_C.npy"))
        if sha_arr(C) != r["C_sha"]:
            continue
        R[(j["k"], j["type"], j["arm"], j["model"])] = (r, C)
    ks = sorted({k for k, *_ in R}); FP, P2, AUC, AUCfc, INP = {}, {}, {}, {}, {}
    for k in ks:
        for t in TYPES:
            for a in ARMS:
                s, c = R.get((k, t, a, "signed")), R.get((k, t, a, "coop"))
                if s:
                    FP[(k, t, a)] = float((s[1][m] < 0).mean())
                    fi, _, ft = job_files(dict(k=k, type=t, arm=a)); x = np.load(fi); INP[(k, t, a)] = E10.desc(x)
                    if t == "CMPN":
                        Ct = np.load(ft); y = (Ct[m] < 0); AUC[(k, a)] = auc(-s[1][m], y); AUCfc[(k, a)] = auc(-np.corrcoef(L.prep_emp(x))[m], y)
                if s and c:
                    P2[(k, t, a)] = s[0]["mse_ho"] - c[0]["mse_ho"]
    cell = lambda D, t, a: boot([D[(k, t, a)] for k in ks if (k, t, a) in D], rng)
    pdiff = lambda D, t1, a1, t2, a2: boot([D[(k, t1, a1)] - D[(k, t2, a2)] for k in ks if (k, t1, a1) in D and (k, t2, a2) in D], rng)
    o = dict(n_truths=len(ks), FP={f"{t}|{a}": cell(FP, t, a) for t in TYPES for a in ARMS}, P2={f"{t}|{a}": cell(P2, t, a) for t in TYPES for a in ARMS})
    vb1 = dict(fp=o["FP"]["COOP|TP"], p2=o["P2"]["COOP|TP"], tp_minus_s=pdiff(FP, "COOP", "TP", "COOP", "S"))
    vb1["met"] = bool(vb1["fp"].get("n", 0) >= 20 and vb1["fp"]["median"] >= 0.20 and vb1["p2"]["ci95"][1] < 0 and vb1["tp_minus_s"]["ci95"][0] > 0.10)
    d2 = pdiff(FP, "CMPN", "TP", "COOP", "TP")
    vb2 = dict(cmpn_minus_coop_tp=d2, reading=("not analysable" if d2.get("n", 0) < 20 else "indistinguishable" if -0.023 < d2["ci95"][0] and d2["ci95"][1] < 0.023
                                               else "distinguishable" if d2["ci95"][0] > 0.023 else "undetermined"))
    o.update(VB1=vb1, VB2=vb2, tp_minus_gs=dict(COOP=pdiff(FP, "COOP", "TP", "COOP", "GS"), CMPN=pdiff(FP, "CMPN", "TP", "CMPN", "GS")),
             auc=dict(est={a: boot([AUC[(k, a)] for k in ks if (k, a) in AUC], rng) for a in ARMS}, fc={a: boot([AUCfc[(k, a)] for k in ks if (k, a) in AUCfc], rng) for a in ARMS},
                      est_minus_fc={a: boot([AUC[(k, a)] - AUCfc[(k, a)] for k in ks if (k, a) in AUC and AUC[(k, a)] is not None and AUCfc[(k, a)] is not None], rng) for a in ARMS}),
             inputs={f"{t}|{a}": {q: boot([INP[(k, t, a)][q] for k in ks if (k, t, a) in INP], rng) for q in ("neg_fc", "gs")} for t in TYPES for a in ARMS})
    json.dump(o, open(path, "x"), indent=1, default=float)
    print("VB1", vb1["met"], "| COOP-TP FP", vb1["fp"].get("median"), "| P2", vb1["p2"].get("ci95"), "| TP-S", vb1["tp_minus_s"].get("ci95"), flush=True)
    print("VB2", vb2["reading"], d2, "| TP-GS COOP", o["tp_minus_gs"]["COOP"].get("median"), flush=True)


def stage_fixture():
    stage_prepare(2, fixture=True); J = job_list(FX["dir"]); os.makedirs(FX["dir"] + "jobs", exist_ok=True)
    for j in [x for x in J if x["arm"] == "TP"][:4]:
        r = run_job(j, time.time() + 3600, out=FX["dir"], iters=FX["iters"]); print("fixture fit", j["k"], j["type"], j["arm"], j["model"], "| iterations", r["iterations"], "| secs", r["secs"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("arg", nargs="?"); ap.add_argument("--procs", type=int, default=30); a = ap.parse_args()
    dict(fixture=lambda: stage_fixture(), prepare=lambda: stage_prepare(a.procs), freeze=stage_freeze, run=lambda: stage_run(a.procs), analyze=lambda: stage_analyze(a.arg))[a.stage]()
