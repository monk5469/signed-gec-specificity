"""Fits with participant-specific SC at 100 and 232 regions (Section 2.4.2): signed / individual-SC, cooperative / individual-SC
and cooperative / group-SC models on the frozen PPC inputs with the PPC protocol."""
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; OUT = P33 + "r23/"; SCDIR = OUT + "sc/"; R3INP = OUT + "r3inp/"; ATL = DATA_ROOT + "raw/p33_atlas232/"
PANEL = OUT + "r23_panel.json"; MAN = OUT + "r23_manifest.json"; FREEZE = OUT + "r23_freeze.json"; RULES = OUT + "r23_rules.json"
SPEC = DATA_ROOT + "docs_r1/172_r23_spec_v0.3.md"; DEMO = {"100307", "100408", "101107"}; BUDGET_H = 96.0
R2_FULL = ("mot", "null10", "acc", "gsr"); R2_SONLY = ("acc1", "acc3"); R3_CONDS = ("mot", "null10", "acc", "gsr"); ARMS = ("S-ind", "C-ind", "C-grp")
CODE = ["r23_run.py", "r23_analyze.py", "r23_sc.py", "r3_prep.py", "luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_ppc.py",
        "luppi_ppc_prep.py", "luppi_pipeline_vol.py", "r5_groupfc.py", "e12a_g_freeze.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_rules():
    r = json.load(open(RULES)); now = {c: sha(c) for c in ("r23_run.py", "r23_analyze.py", "r23_sc.py", "r3_prep.py")}; now["spec"] = sha(SPEC)
    if r["hashes"] != now:
        raise SystemExit(f"rules changed since registration: {[k for k in now if now[k] != r['hashes'].get(k)]}")


def volumes(path):
    import nibabel as nib
    A = np.asarray(nib.load(path).dataobj).astype(int); return np.array([(A == k).sum() * 8.0 for k in range(1, int(A.max()) + 1)])


def selected_sc(sid, n, sel):
    ce, w = sel.split("_"); M = np.load(SCDIR + f"{sid}.npz")[f"sc{n}_{ce}"].astype(float)
    if w == "volnorm":
        v = volumes(ATL + ("mni_schaefer100.nii.gz" if n == 100 else "mni_schaefer232.nii.gz")); M = M / (v[:, None] + v[None, :])
    np.fill_diagonal(M, 0.0); return M


def ppc_inputs(cond):
    """PPC e5 files of a condition with the frozen group-freeze hashes; subject_index map."""
    import luppi_r4e_ppc as W
    X = W.X; W.point(cond); fsha, fr = X.verify_freeze("e5"); idx = {j["tag"]: j["subject_index"] for j in X.job_list("e5")}
    return X.OUT, fr["files"], idx, fsha


def stage_prepare():
    check_rules()
    if os.path.exists(PANEL):
        raise SystemExit("panel already prepared")
    gsc = json.load(open(OUT + "r23_gsc.json")); log = json.load(open(SCDIR + "build_log.json")); sel = gsc["selected"]; ce = int(sel.split("_")[0][2:])
    valid = {r["id"] for r in log["runs"] if r["ce"] == ce and not r["error"] and r["val100"]["valid"] and r["val232"]["valid"]}
    r3m = json.load(open(OUT + "r3_prep_manifest.json")); out_dir, files_acc, idx, _ = ppc_inputs("acc")
    subs = [s for s in r3m["subjects"] if s not in DEMO and s in valid]; files = {}; os.makedirs(OUT + "scsel", exist_ok=True)
    for n in (100, 232):
        mats = {s: selected_sc(s, n, sel) for s in subs}; G = np.mean([mats[s] for s in subs], 0)
        for s, M in mats.items():
            p = OUT + f"scsel/{s}_sc{n}.npy"; np.save(p, M); files[p] = sha(p)
        p = OUT + f"scsel/group_sc{n}.npy"; np.save(p, G); files[p] = sha(p)
    for c in R2_FULL + R2_SONLY:
        od, fr, _, fsha = ppc_inputs(c)
        for rel in [f"e5/freq_REST1.npy"] + [f"e5/{s}_REST1_{k}.npy" for s in subs for k in ("input", "heldout")]:
            if sha(od + rel) != fr[f"ext:{rel}"]:
                raise SystemExit(f"{c} {rel} differs from the PPC freeze")
            files[od + rel] = fr[f"ext:{rel}"]
    for p, h in r3m["files"].items():
        if sha(p) != h:
            raise SystemExit(f"{p} differs from the R3 manifest")
        files[p] = h
    off = lambda M: (M[~np.eye(len(M), dtype=bool)] > 0).mean()
    panel = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), subjects=subs, subject_index={s: idx[s] for s in subs}, gsc_selected=sel, gsc_label=gsc["label"],
                 excluded_invalid_sc=sorted(set(r3m["subjects"]) - DEMO - set(subs)),
                 density=dict(ind100_median=float(np.median([off(np.load(OUT + f"scsel/{s}_sc100.npy")) for s in subs])), grp100=float(off(np.load(OUT + "scsel/group_sc100.npy"))),
                              ind232_median=float(np.median([off(np.load(OUT + f"scsel/{s}_sc232.npy")) for s in subs])), grp232=float(off(np.load(OUT + "scsel/group_sc232.npy")))))
    json.dump(panel, open(PANEL, "x"), indent=1)
    json.dump(dict(created=panel["created"], files=files, code={c: sha(c) for c in CODE}, gsc_sha=sha(OUT + "r23_gsc.json"), build_log_sha=sha(SCDIR + "build_log.json"),
                   r3_manifest_sha=sha(OUT + "r3_prep_manifest.json")), open(MAN, "w"), indent=1)
    print("prepared | participants", len(subs), "| selected", sel, gsc["label"], "| densities", panel["density"], flush=True)


def job_list():
    P = json.load(open(PANEL)); J = []
    for s in P["subjects"]:
        for res, conds in ((100, R2_FULL + R2_SONLY), (232, R3_CONDS)):
            for c in conds:
                for a in (("S-ind",) if c in R2_SONLY else ARMS):
                    J.append(dict(exp="R23", res=res, cond=c, arm=a, tag=s, subject_index=P["subject_index"][s], seed=11))
    for j in J:
        j["key"] = "|".join(f"{q}={j[q]}" for q in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
        core = j["arm"] in ("S-ind", "C-ind") and j["cond"] in ("mot", "acc")
        j["tier"] = (0 if core else 1) if j["res"] == 100 else (2 if core else 3)
    return J


def job_files(j):
    if j["res"] == 100:
        od = P33 + f"r4e_ppc_{j['cond']}/"; inp = [od + f"e5/{j['tag']}_REST1_input.npy", od + f"e5/{j['tag']}_REST1_heldout.npy", od + "e5/freq_REST1.npy"]
    else:
        d = R3INP + f"{j['cond']}/"; inp = [d + f"{j['tag']}_REST1_input.npy", d + f"{j['tag']}_REST1_heldout.npy", d + "freq_REST1.npy"]
    return inp + [OUT + (f"scsel/group_sc{j['res']}.npy" if j["arm"] == "C-grp" else f"scsel/{j['tag']}_sc{j['res']}.npy")]


_SIM = {}


def run_job(j, stop_at, iters=None):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    N = j["res"]
    if N not in _SIM:
        _SIM[N] = L.make_simulator(N, Lib.T_LEN, 3000.0)
    sim = _SIM[N]; t0 = time.time(); cfg = dict(L.HEADER)
    if iters:
        cfg["max_iterations"] = iters
    fi, fh, ff, fs = job_files(j); inp = np.load(fi); hop = L.prep_emp(np.load(fh)); f = np.load(ff); SC = np.load(fs)
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=j["arm"].startswith("C"), lag="matlab", sim=sim, stop_at=stop_at)
    base = 9_000_000 + 1000 * j["subject_index"]; sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm)
    if not (C.shape == (N, N) and np.all(np.isfinite(C)) and np.isfinite(mse)):
        raise ValueError("non-finite or mis-shaped fitted matrix or held-out MSE")
    np.save(j["file"].replace(".json", "_C.npy"), C); np.save(j["file"].replace(".json", "_FCsim.npy"), FCm.astype(np.float32))
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"], mse_ho=mse,
                C_sha=sha_arr(C), FCsim_sha=sha_arr(FCm.astype(np.float32)))


def stage_freeze():
    check_rules()
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    mf = json.load(open(MAN)); J = job_list(); files = {}
    for p in sorted({q for j in J for q in job_files(j)}):
        if p not in mf["files"] or sha(p) != mf["files"][p]:
            raise SystemExit(f"{p} missing from the manifest or changed")
        files[p] = mf["files"][p]
    fr = dict(group="r23", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE}, rules=sha(RULES), panel=sha(PANEL),
              manifest=sha(MAN), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen r23", sha(FREEZE), "jobs", len(J), {t: sum(j["tier"] == t for j in J) for t in range(4)}, flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + [(RULES, fr["rules"]), (PANEL, fr["panel"]), (MAN, fr["manifest"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="r23", inputs={p: fr["files"][p] for p in job_files(j)})


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
            return w(dict(j, state="cancelled", prov=prov, reason="budget exhausted", attempts=attempt - 1, errors=errors))
        try:
            r = run_job(j, deadline); r.update(state="done", prov=prov, attempts=attempt, errors=errors); return w(r)
        except L.Cancelled as e:
            return w(dict(j, state="cancelled", prov=prov, reason=str(e), attempts=attempt, errors=errors))
        except Exception:
            errors.append(traceback.format_exc()[-1500:])
    return w(dict(j, state="failed_twice", prov=prov, attempts=2, errors=errors))


def stage_run(procs):
    check_rules(); fsha, fr = verify_freeze(); os.makedirs(OUT + "jobs", exist_ok=True); dl_file = OUT + "r23_budget.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + BUDGET_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha:
        raise SystemExit("budget record does not belong to this freeze")
    J = job_list()
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    J.sort(key=lambda j: (j["tier"], j["res"], j["cond"], j["tag"], j["arm"])); print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.stage == "rules":
        if os.path.exists(RULES):
            raise SystemExit("rules already registered")
        h = {c: sha(c) for c in ("r23_run.py", "r23_analyze.py", "r23_sc.py", "r3_prep.py")}; h["spec"] = sha(SPEC)
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), hashes=h, stage="code registered before production SC construction"), open(RULES, "x"), indent=1)
        print("rules", {k: v[:12] for k, v in h.items()})
    else:
        {"prepare": stage_prepare, "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
