"""Released-configuration check (K3; Sections 2.4.3, 3.7.3; Supplementary Note 14): the PPC core contrasts under the authors'
public configuration as executed by their MATLAB function."""
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; OUT = P33 + "vavf/va/"; CONDS = ("mot", "null10", "acc", "gsr"); MODELS = ("signed", "coop")
K3 = dict(sigma=0.001, max_iterations=5000, eta_fc=1e-3, eta_tau=1e-4, alpha=1e-4, max_C=0.15, conv=1e-3, tau=1, a=-0.02, burnin=3000.0)
DEMO = {"100307", "100408", "101107"}; BUDGET_H = 48.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rules():
    import vavf_rules
    vavf_rules.check("va", ("vavf_va.py",))


def ppc(cond):
    import r23_run as R
    return R.ppc_inputs(cond)                       # (out_dir, freeze files, {tag: subject_index}, freeze sha)


def stage_prepare():
    rules()
    if os.path.exists(OUT + "va_manifest.json"):
        raise SystemExit("prepared once")
    os.makedirs(OUT + "jobs", exist_ok=True); files = {}; J = []; idx_all = None
    for c in CONDS:
        od, fr, idx, fsha = ppc(c); idx_all = idx_all or idx
        for rel in ["e5/freq_REST1.npy", "e5/SC.npy"] + [f"e5/{s}_REST1_{k}.npy" for s in idx if s not in DEMO for k in ("input", "heldout")]:
            key = f"ext:{rel}" if f"ext:{rel}" in fr else rel
            if key in fr and sha(od + rel) != fr[key]:
                raise SystemExit(f"{c} {rel} differs from the PPC freeze")
            files[od + rel] = sha(od + rel)
        for s in sorted(idx):
            if s in DEMO:
                continue
            for m in MODELS:
                J.append(dict(cond=c, tag=s, subject_index=idx[s], model=m, file=OUT + f"jobs/{c}_{s}_{m}.json"))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), config=K3, jobs=J, files=files, vg=sha(P33 + "vg/vg_analysis.json")), open(OUT + "va_manifest.json", "x"), indent=1)
    print("prepared | jobs", len(J), "| files", len(files), flush=True)


def job_list():
    return json.load(open(OUT + "va_manifest.json"))["jobs"]


def paths(j):
    od = P33 + f"r4e_ppc_{j['cond']}/e5/"; return od + f"{j['tag']}_REST1_input.npy", od + f"{j['tag']}_REST1_heldout.npy", od + "freq_REST1.npy", od + "SC.npy"


_SIM = {}


def run_job(j, stop_at, iters=None):
    import jax, luppi_hopf as L, luppi_r4e_lib as Lib
    if "s" not in _SIM:
        _SIM["s"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["s"]; fi, fh, ff, fs = paths(j); x = np.load(fi); hop = L.prep_emp(np.load(fh)); f = np.load(ff); SC = np.load(fs); cfg = dict(K3)
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
        np.save(j["file"].replace(".json", "_C.npy"), C); np.save(j["file"].replace(".json", "_FCsim.npy"), FCm.astype(np.float32))
    return dict(j, state="done", secs=round(time.time() - t0, 1), iterations=info["iterations"], stop=info["stop"], fc_corr_last=info["fc_corr_last"],
                fc_corr=info["fc_corr"], mse_ho=float(mse), C_sha=sha_arr(C))


def stage_freeze():
    rules(); m = json.load(open(OUT + "va_manifest.json")); bad = [p for p, h in m["files"].items() if sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} inputs changed")
    if os.path.exists(OUT + "va_freeze.json"):
        raise SystemExit("frozen once")
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), manifest=sha(OUT + "va_manifest.json"), code=sha(os.path.abspath(__file__)), n_jobs=len(m["jobs"])),
              open(OUT + "va_freeze.json", "x"), indent=1); print("frozen | jobs", len(m["jobs"]), flush=True)


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
    rules(); fr = json.load(open(OUT + "va_freeze.json"))
    if fr["manifest"] != sha(OUT + "va_manifest.json") or fr["code"] != sha(os.path.abspath(__file__)):
        raise SystemExit("freeze mismatch")
    dl = OUT + "va_deadline.json"
    if not os.path.exists(dl):
        json.dump(dict(deadline=time.time() + BUDGET_H * 3600), open(dl, "w"))
    J = sorted(job_list(), key=lambda j: (j["cond"] not in ("mot", "acc"), j["tag"], j["model"]))
    with mp.Pool(procs) as p:
        st = p.map(safe, [(j, json.load(open(dl))["deadline"]) for j in J], chunksize=1)
    print({s: st.count(s) for s in set(st)}, flush=True)


def boot(v, rng, B=2000):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    bs = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(B)]
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else None,
                ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def stage_analyze(path):
    from scipy.stats import spearmanr
    import luppi_hopf as L
    rules(); rng = np.random.default_rng(20261024); SC = np.load(P33 + "r4e_ppc_acc/e5/SC.npy"); m = (SC != 0) & ~np.eye(100, dtype=bool); A = {}
    for j in job_list():
        try:
            r = json.load(open(j["file"]))
        except Exception:
            continue
        if r.get("state") == "done":
            C = np.load(j["file"].replace(".json", "_C.npy"))
            if sha_arr(C) == r["C_sha"]:
                A[(j["cond"], j["tag"], j["model"])] = (r, C)
    tags = sorted({t for _, t, _ in A}); a2 = 2 * K3["alpha"]; E = {}
    for (c, t, mo), (r, C) in A.items():
        E[(c, t, mo)] = dict(fp2a=float((C[m] < -a2).mean()), fp0=float((C[m] < 0).mean()), fp1e3=float((C[m] < -1e-3).mean()), r=r["fc_corr_last"], mse=r["mse_ho"])
    q = lambda c, mo, k: {t: E[(c, t, mo)][k] for t in tags if (c, t, mo) in E}
    P2 = {c: {t: E[(c, t, "signed")]["mse"] - E[(c, t, "coop")]["mse"] for t in tags if (c, t, "signed") in E and (c, t, "coop") in E} for c in CONDS}
    pd = lambda D1, D2: boot([D1[t] - D2[t] for t in D1 if t in D2], rng)
    comp = sum(all((c, t, mo) in E for c in ("mot", "acc") for mo in MODELS) for t in tags); ok = comp >= 86
    o = dict(complete=comp, analysable=ok, cells={c: {k: boot(list(q(c, "signed", k).values()), rng) for k in ("fp2a", "fp0", "fp1e3", "r")} for c in CONDS},
             r_coop={c: boot(list(q(c, "coop", "r").values()), rng) for c in CONDS}, P2={c: boot(list(P2[c].values()), rng) for c in CONDS},
             share_signed_better={c: float(np.mean([v < 0 for v in P2[c].values()])) if P2[c] else None for c in CONDS})
    o["acc_minus_mot"] = pd(q("acc", "signed", "fp2a"), q("mot", "signed", "fp2a")); o["acc_minus_null10"] = pd(q("acc", "signed", "fp2a"), q("null10", "signed", "fp2a"))
    o["P2_acc_minus_mot"] = pd(P2["acc"], P2["mot"]); o["fp0_acc_minus_mot"] = pd(q("acc", "signed", "fp0"), q("mot", "signed", "fp0"))
    o["VA-T"] = bool(o["acc_minus_mot"]["ci95"][0] > 0.10) if ok else "not analysable"
    o["VA-N"] = bool(o["acc_minus_null10"]["ci95"][0] > 0.10) if ok else "not analysable"
    o["VA-P"] = bool(o["P2_acc_minus_mot"]["ci95"][1] < 0) if ok else "not analysable"
    gl = {}; iu = np.triu_indices(100, 1)
    for c in CONDS:
        ks = [t for t in tags if all((c, t, mo) in E for mo in MODELS)]
        if len(ks) >= 86:
            Fe = np.mean([np.corrcoef(L.prep_emp(np.load(paths(dict(cond=c, tag=t))[0]))) for t in ks], 0)
            for mo in MODELS:
                Fs = np.mean([np.load(OUT + f"jobs/{c}_{t}_{mo}_FCsim.npy").astype(float) for t in ks], 0)
                gl[f"{c}|{mo}"] = dict(pearson=float(np.corrcoef(Fe[iu], Fs[iu])[0, 1]), spearman=float(spearmanr(Fe[iu], Fs[iu])[0]), n=len(ks))
    o["group_level"] = gl
    json.dump(o, open(path, "x"), indent=1, default=float)
    print("VA | T", o["VA-T"], "N", o["VA-N"], "P", o["VA-P"], "| acc-mot", o["acc_minus_mot"], "| P2 acc-mot", o["P2_acc_minus_mot"], flush=True)
    print("VA | r signed acc", o["cells"]["acc"]["r"], "| r coop acc", o["r_coop"]["acc"], "| group", gl.get("acc|signed"), gl.get("acc|coop"), gl.get("mot|signed"), gl.get("mot|coop"), flush=True)


def stage_fixture():
    import tempfile
    j = dict(cond="acc", tag="101309", subject_index=0, model="signed", file=tempfile.mkdtemp() + "/fx.json")
    for c in CONDS:
        print(c, [os.path.exists(p) for p in paths(dict(j, cond=c))], flush=True)
    r = run_job(j, time.time() + 3600, iters=120); print("fixture | iterations", r["iterations"], "| secs", r["secs"], "| mse finite", np.isfinite(r["mse_ho"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("arg", nargs="?"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dict(fixture=stage_fixture, prepare=stage_prepare, freeze=stage_freeze, run=lambda: stage_run(a.procs), analyze=lambda: stage_analyze(a.arg))[a.stage]()
