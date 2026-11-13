"""Shared run framework of the known-truth simulations: truth construction, freeze and provenance checks, deadline-controlled job
execution and the held-out evaluation P2."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e/"
R4V2 = DATA_ROOT + "results/p33/r4v2/"
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_run.py"]
SEEDS = (11, 12, 13)
TAU_D = (20.0, 50.0); Q_TARGET = (0.05, 0.20)
DEADLINE_H = 72.0
FAMILIES = {"QM1": range(1001, 1031), "QM3": range(2001, 2007), "CMP": range(3001, 3016), "CMP-H": range(3101, 3107)}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def code_hashes():
    return {c: sha(c) for c in CODE}


# ---------------------------------------------------------------- truths
def build_truth(family, s):
    import luppi_r4e_lib as Lib
    if family in ("QM1", "QM3"):
        return Lib.truth_qm(s, 1.0 if family == "QM1" else 3.0)
    return Lib.truth_competitive(s, aligned=(family == "CMP-H"))


def sg_truth(k, cfgname):
    import luppi_r4e_lib as Lib
    _, SC, f = Lib.demo_subject(k)
    C = np.load(R4V2 + f"C_s{k}_{cfgname}_coop_matlab_seed11.npy")
    return C, SC, f, dict(family="SG" if cfgname == "header" else "SG-L1", seed=k, subject=k)


FC_RANGE = (0.03, 0.40)          #: target realism gate (mean off-diagonal FC, R branch); example participants 0.10-0.23


def valid_series(x):
    return bool(np.all(np.isfinite(x)) and np.all(x.std(1) > 0) and np.max(np.abs(x)) < 10)


def realistic(x):
    import luppi_hopf as L
    F = np.corrcoef(L.prep_emp(x))[~np.eye(len(x), dtype=bool)]; return bool(FC_RANGE[0] <= F.mean() <= FC_RANGE[1]), float(F.mean())


def truth_job(args):
    family, s = args
    import luppi_r4e_lib as Lib
    sim = Lib.make_driven_simulator(100, Lib.T_LEN, 3000.0)
    tries = [s] if family in ("SG", "SG-L1") else [s, s + 6000, s + 12000]
    log = []
    for s2 in tries:
        C, SC, f, meta = sg_truth(s2, "header" if family == "SG" else "l1p") if family in ("SG", "SG-L1") else build_truth(family, s2)
        tgt = Lib.simulate(C, f, 1_000_000 + s2, sim); ho = Lib.simulate(C, f, 2_000_000 + s2, sim)
        real, mfc = realistic(tgt); ok = valid_series(tgt) and valid_series(ho) and real; log.append(dict(seed=s2, valid=ok, mean_fc=mfc))
        if ok:
            tag = f"{family}_{s}"
            for name, arr in (("C", C), ("f", f), ("target", tgt), ("heldout", ho)):
                np.save(OUT + f"truth/{tag}_{name}.npy", arr)
            m = Lib.support(SC); neg_fc = float(np.mean(np.corrcoef(tgt)[~np.eye(100, dtype=bool)] < 0))
            return dict(tag=tag, family=family, seed=s, used_seed=s2, stratum=int(meta["subject"]), attempts=log,
                        n_neg_true=int(np.sum(C[m] < 0)), max_pos=float(C.max()), target_fc_neg_frac=neg_fc,
                        sha={n: sha(OUT + f"truth/{tag}_{n}.npy") for n in ("C", "f", "target", "heldout")},
                        f_sha=sha_arr(f), meta={k: v for k, v in meta.items() if k not in ("family", "seed")})
    return dict(tag=f"{family}_{s}", family=family, seed=s, attempts=log, missing=True)


def stage_truths(procs):
    os.makedirs(OUT + "truth", exist_ok=True)
    jobs = [("SG", k) for k in range(3)] + [(fam, s) for fam, rr in FAMILIES.items() for s in rr]
    with mp.Pool(procs) as p:
        res = p.map(truth_job, jobs, chunksize=1)
    inputs = {n: sha(DATA_ROOT + "raw/luppi_demo/gec_3/" + n) for n in ("FMRI.mat", "SC.mat")}
    sg_src = {f"C_s{k}_header_coop_matlab_seed11.npy": sha(R4V2 + f"C_s{k}_header_coop_matlab_seed11.npy") for k in range(3)}
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), spec="see the paper, Methods", rng=dict(jax="threefry2x32 PRNGKey", numpy="PCG64 default_rng"),
               namespaces=dict(target="1e6+s", heldout="2e6+s", driver="3e6+100s+10i_tau+i_q (heldout +500000)", calib="5e6+j", eval="4e6+1000s+10seed+r (r = 0..7, shared by signed and coop)"),
               inputs=inputs, sg_sources=sg_src, code=code_hashes(), truths=res)
    json.dump(man, open(OUT + "r4e_manifest.json", "w"), indent=1, default=float)
    print("truths:", len(res), "missing:", sum(1 for r in res if r.get("missing")), "replaced:", sum(1 for r in res if r.get("used_seed", r["seed"]) != r["seed"]), flush=True)


# ---------------------------------------------------------------- calibration and targets
def drive_share(C, f, g, s_drive, tau, sim, j):
    import luppi_r4e_lib as Lib
    x0 = Lib.simulate(C, f, 5_000_000 + j, sim, g=g, s=0.0, tau_d=tau, seed_d=5_100_000 + j)
    xs = Lib.simulate(C, f, 5_000_000 + j, sim, g=g, s=s_drive, tau_d=tau, seed_d=5_100_000 + j)
    return float(np.median((xs.var(1) - x0.var(1)) / xs.var(1)))


def calib_job(args):
    tau, q = args
    import luppi_r4e_lib as Lib
    sim = Lib.make_driven_simulator(100, Lib.T_LEN, 3000.0)
    C = np.load(OUT + "truth/QM1_1001_C.npy"); f = np.load(OUT + "truth/QM1_1001_f.npy")
    k = (1001 - 1) % 3; _, SC, _ = Lib.demo_subject(k); g = Lib.fiedler(SC)
    lo, hi = -4.0, 1.0; trace = []
    for _ in range(30):
        mid = (lo + hi) / 2; val = float(np.mean([drive_share(C, f, g, 10 ** mid, tau, sim, j) for j in range(4)])); trace.append((mid, val))
        if val < q:
            lo = mid
        else:
            hi = mid
    best = min(trace, key=lambda t: abs(t[1] - q))
    return dict(tau_d=tau, q_target=q, log10_s=best[0], s=10 ** best[0], q=best[1], success=bool(abs(best[1] - q) <= 0.1 * q), trace=trace)


def e2_target_job(args):
    s, it, iq, sdrive, tau = args
    import luppi_r4e_lib as Lib
    sim = Lib.make_driven_simulator(100, Lib.T_LEN, 3000.0)
    tag = f"QM1_{s}"; C = np.load(OUT + f"truth/{tag}_C.npy"); f = np.load(OUT + f"truth/{tag}_f.npy")
    g = np.load(OUT + f"truth/{tag}_g.npy")                  # written once, sequentially, by stage_calibrate
    kd = 3_000_000 + 100 * s + 10 * it + iq
    tgt = Lib.simulate(C, f, 1_000_000 + s, sim, g=g, s=sdrive, tau_d=tau, seed_d=kd)
    ho = Lib.simulate(C, f, 2_000_000 + s, sim, g=g, s=sdrive, tau_d=tau, seed_d=kd + 500_000)
    cell = f"tau{int(tau)}_q{iq}"
    np.save(OUT + f"truth/{tag}_{cell}_target.npy", tgt); np.save(OUT + f"truth/{tag}_{cell}_heldout.npy", ho)
    return dict(tag=tag, cell=cell, valid=valid_series(tgt) and valid_series(ho), sha_target=sha(OUT + f"truth/{tag}_{cell}_target.npy"),
                sha_heldout=sha(OUT + f"truth/{tag}_{cell}_heldout.npy"))


E2_TRUTHS = (1001, 1002, 1003, 1004, 1005, 1006)        # 2 per stratum


def stage_calibrate(procs):
    import luppi_r4e_lib as Lib
    gfiles = []
    for s in E2_TRUTHS:
        _, SC, _ = Lib.demo_subject((s - 1) % 3); p = f"truth/QM1_{s}_g.npy"; np.save(OUT + p, Lib.fiedler(SC)); gfiles.append(dict(file=p, sha=sha(OUT + p)))
    with mp.Pool(min(procs, 4)) as p:
        cal = p.map(calib_job, [(tau, q) for tau in TAU_D for q in Q_TARGET], chunksize=1)
    jobs = [(s, it, iq, c["s"], c["tau_d"]) for s in E2_TRUTHS for it, tau in enumerate(TAU_D) for iq, q in enumerate(Q_TARGET)
            for c in cal if c["tau_d"] == tau and c["q_target"] == q]
    with mp.Pool(procs) as p:
        tg = p.map(e2_target_job, jobs, chunksize=1)
    files = gfiles + [dict(file=f"truth/{t['tag']}_{t['cell']}_{n}.npy", sha=t[f"sha_{n}"]) for t in tg for n in ("target", "heldout")]
    json.dump(dict(calibration=cal, targets=tg, files=files, code=code_hashes()), open(OUT + "r4e_e2_calibration.json", "w"), indent=1, default=float)
    for c in cal:
        print("tau", c["tau_d"], "q*", c["q_target"], "-> s %.4g q %.4f success %s" % (c["s"], c["q"], c["success"]), flush=True)


# ---------------------------------------------------------------- fitting jobs
def job_list(exps):
    man = json.load(open(OUT + "r4e_manifest.json")); ok = {t["tag"]: t for t in man["truths"] if not t.get("missing")}
    J = []
    def add(exp, tag, br, cfg, lag, fsrc, cell, seed, model):
        if tag in ok:
            J.append(dict(exp=exp, tag=tag, family=ok[tag]["family"], stratum=ok[tag]["stratum"], branch=br, cfg=cfg, lag=lag,
                          freq=fsrc, cell=cell, seed=seed, model=model))
    for e in exps:
        if e == "E0":
            for k in range(3):
                for br in "RF":
                    for sd in SEEDS:
                        for mdl in ("signed", "coop"):
                            add("E0", f"SG_{k}", br, "released", "matlab", "target", "none", sd, mdl)
        if e == "E0m":                                   #: generator and estimator both at sigma 0.001
            for k in range(3):
                for br in "RF":
                    for sd in SEEDS:
                        for mdl in ("signed", "coop"):
                            add("E0m", f"SG_{k}", br, "released", "matlab", "target", "sig0.001", sd, mdl)
        if e == "E1":
            tags = [f"SG_{k}" for k in range(3)] + [f"QM1_{s}" for s in FAMILIES["QM1"]] + [f"QM3_{s}" for s in FAMILIES["QM3"]]
            for t in tags:
                for br in "RF":
                    for sd in SEEDS:
                        for mdl in ("signed", "coop"):
                            add("E1", t, br, "header", "matlab", "target", "none", sd, mdl)
        if e == "E1w":
            for s in FAMILIES["QM1"]:
                for br in "RF":
                    for mdl in ("signed", "coop"):
                        add("E1w", f"QM1_{s}", br, "header", "matlab", "true", "none", 11, mdl)
        if e == "E1cpp":
            for s in list(FAMILIES["QM1"])[:5]:
                for mdl in ("signed", "coop"):
                    add("E1cpp", f"QM1_{s}", "F", "header", "cpp", "target", "none", 11, mdl)
        if e == "E2":
            for s in E2_TRUTHS:
                for tau in TAU_D:
                    for iq in range(2):
                        for br in "RF":
                            for sd in SEEDS:
                                for mdl in ("signed", "coop"):
                                    add("E2", f"QM1_{s}", br, "header", "matlab", "target", f"tau{int(tau)}_q{iq}", sd, mdl)
        if e == "E3":
            for fam in ("CMP", "CMP-H"):
                for s in FAMILIES[fam]:
                    for br in "RF":
                        for sd in SEEDS:
                            for mdl in ("signed", "coop"):
                                add("E3", f"{fam}_{s}", br, "header", "matlab", "target", "none", sd, mdl)
    for j in J:
        j["key"] = "|".join(str(j[k]) for k in ("exp", "family", "tag", "stratum", "branch", "cfg", "lag", "freq", "cell", "seed", "model"))
        j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J, man


_SIM = {}


def cell_prefix(j):
    return "" if j["cell"] == "none" else f"_{j['cell']}"


def run_job(j, stop_at):
    import luppi_hopf as L, luppi_r4e_lib as Lib
    import jax
    if "sim" not in _SIM:
        _SIM["sim"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["sim"]; tag = j["tag"]; t0 = time.time(); pre = cell_prefix(j)
    C_true = np.load(OUT + f"truth/{tag}_C.npy"); f_true = np.load(OUT + f"truth/{tag}_f.npy")
    k = j["stratum"]; _, SC, _ = Lib.demo_subject(k)
    tgt = np.load(OUT + f"truth/{tag}{pre}_target.npy"); ho = np.load(OUT + f"truth/{tag}{pre}_heldout.npy")
    inp = Lib.bandpass(tgt) if j["branch"] == "F" else tgt                 # what the authors would pass in
    f = f_true if j["freq"] == "true" else L.regional_frequencies([inp])
    cfg = dict(dict(header=L.HEADER, l1p=L.L1P, released=L.RELEASED)[j["cfg"]])
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag=j["lag"], sim=sim, stop_at=stop_at)
    # P2: 8 RAW simulations of C (fc / lag_fc directly, no detrending) with evaluation keys
    # independent of optimisation length and shared by signed / coop; held-out side processed by branch.
    s_id = int(tag.split("_")[-1]) if tag.split("_")[-1].isdigit() else 0
    keys = [jax.random.PRNGKey(4_000_000 + 1000 * s_id + 10 * j["seed"] + r) for r in range(8)]
    sims = []
    for r, kk in enumerate(keys):                         #: deadline checked before and after every P2 simulation
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], kk)))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    hop = Lib.branch(ho, j["branch"])
    FCho, LAGho = L.fc(hop), L.lag_fc(hop, 1, "matlab")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    FCb = np.mean([L.fc(Lib.branch(x, j["branch"])) for x in sims], 0); LAGb = np.mean([L.lag_fc(Lib.branch(x, j["branch"]), 1, "matlab") for x in sims], 0)
    np.save(j["file"].replace(".json", "_C.npy"), C)
    tfc = np.corrcoef(Lib.branch(tgt, j["branch"]))[~np.eye(100, dtype=bool)]
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"],
                fc_corr_last=info["fc_corr_last"], mse_ho=Lib.heldout_mse(FCho, LAGho, FCm, LAGm), mse_ho_branch=Lib.heldout_mse(FCho, LAGho, FCb, LAGb),
                metrics=Lib.edge_metrics(C, C_true, SC), target_fc_neg_frac=float(np.mean(tfc < 0)), target_fc_mean=float(tfc.mean()), C_sha=sha_arr(C))


# ---------------------------------------------------------------- freezing and provenance
ANALYSIS = "luppi_r4e_analyze.py"
FIXTURES = "luppi_fixtures.py"
FIXTURE_RESULT = DATA_ROOT + "results/p33/luppi_fixtures.json"
RAW_DEPS = [DATA_ROOT + "raw/luppi_demo/gec_3/FMRI.mat", DATA_ROOT + "raw/luppi_demo/gec_3/SC.mat"] + \
           [R4V2 + f"C_s{k}_header_coop_matlab_seed11.npy" for k in range(3)]


def freeze_path(group):
    return OUT + f"r4e_freeze_{group}.json"


def manifest_files():
    """relative file -> sha256 as recorded by the generating stages (truth manifest, E0m manifest, E2 calibration)."""
    exp = {}
    man = json.load(open(OUT + "r4e_manifest.json"))
    for t in man["truths"]:
        if not t.get("missing"):
            for n, h in t["sha"].items():
                exp[f"truth/{t['tag']}_{n}.npy"] = h
    for extra in ("r4e_manifest_e0m.json", "r4e_e2_calibration.json"):
        if os.path.exists(OUT + extra):
            for rec in json.load(open(OUT + extra)).get("files", []):
                exp[rec["file"]] = rec["sha"]
    return exp


def job_files(j):
    pre = cell_prefix(j)
    return [f"truth/{j['tag']}_C.npy", f"truth/{j['tag']}_f.npy", f"truth/{j['tag']}{pre}_target.npy", f"truth/{j['tag']}{pre}_heldout.npy"]


def stage_freeze(group, exps):
    """Per-group freeze: code (runner, library, estimator, analysis, fixtures), the fixture result (must be all
 passed), raw dependencies (example FMRI / SC, the R4V2 reference matrices behind the T-SG truths), the manifests, and every input
 file of every job of the group (each must match its manifest hash). Refuses if a freeze for this group already exists."""
    fp = freeze_path(group)
    if os.path.exists(fp):
        raise SystemExit(f"{fp} exists; a group is frozen once")
    fx = json.load(open(FIXTURE_RESULT))
    if not fx.get("all_passed"):
        raise SystemExit("fixtures not all passed")
    J, _ = job_list(exps); man = manifest_files()
    files = {}
    for j in J:
        for f in job_files(j):
            if f not in man:
                raise SystemExit(f"{f} is not in any manifest")
            if sha(OUT + f) != man[f]:
                raise SystemExit(f"{f} differs from its manifest hash")
            files[f] = man[f]
    fr = dict(group=group, exps=sorted(exps), n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"),
              code={c: sha(c) for c in CODE + [ANALYSIS, FIXTURES]}, fixture_result={FIXTURE_RESULT: sha(FIXTURE_RESULT)},
              raw={p: sha(p) for p in RAW_DEPS},
              manifests={m: sha(OUT + m) for m in ("r4e_manifest.json", "r4e_manifest_e0m.json", "r4e_e2_calibration.json") if os.path.exists(OUT + m)},
              files=files)
    json.dump(fr, open(fp, "w"), indent=1)
    print("frozen", group, sha(fp), "jobs", len(J), "files", len(files), flush=True)


def verify_freeze(group):
    """Refuse unless every frozen item (code, fixture result, raw dependencies, manifests, input files) is unchanged."""
    fp = freeze_path(group); fr = json.load(open(fp))
    checks = [(c, h, c) for c, h in fr["code"].items()] + [(p, h, p) for p, h in fr["fixture_result"].items()] + \
             [(p, h, p) for p, h in fr["raw"].items()] + [(m, h, OUT + m) for m, h in fr["manifests"].items()] + \
             [(f, h, OUT + f) for f, h in fr["files"].items()]
    bad = [name for name, h, path in checks if not os.path.exists(path) or sha(path) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed or missing, e.g. {bad[:3]}")
    return sha(fp), fr


def job_prov(j, freeze_sha, fr):
    files = job_files(j); missing = [f for f in files if f not in fr["files"]]
    if missing:
        raise SystemExit(f"job {j['key']} uses files not in the freeze: {missing}")
    return dict(freeze=freeze_sha, group=fr["group"], inputs={f: fr["files"][f] for f in files})


def _write(j, rec):
    json.dump(rec, open(j["file"], "w"), indent=1, default=float)
    return rec["state"]


def run_job_safe(args):
    j, prov, deadline = args
    import luppi_hopf as L
    if os.path.exists(j["file"]):
        old = json.load(open(j["file"]))
        if old.get("state") in ("done", "failed_twice", "cancelled") and old.get("prov") == prov and old.get("key") == j["key"]:
            return old["state"]
    errors = []
    for attempt in (1, 2):
        if time.time() > deadline:
            reason = "deadline before start" if attempt == 1 else "deadline before retry after a failed attempt"
            return _write(j, dict(j, state="cancelled", prov=prov, reason=reason, attempts=attempt - 1, errors=errors))
        try:
            r = run_job(j, deadline)
            if time.time() > deadline:                        #: a job finishing after the deadline is not accepted
                return _write(j, dict(j, state="cancelled", prov=prov, reason="deadline passed before acceptance", attempts=attempt, errors=errors))
            r.update(state="done", prov=prov, attempts=attempt, errors=errors)
            return _write(j, r)
        except L.Cancelled as e:
            return _write(j, dict(j, state="cancelled", prov=prov, reason=str(e), attempts=attempt, errors=errors))
        except Exception:
            errors.append(traceback.format_exc())
    return _write(j, dict(j, state="failed_twice", prov=prov, attempts=2, errors=errors))


def stage_run(exps, procs, group):
    """Runs a frozen experiment group; the 72-h hard deadline is per group (r4e_deadline_<group>.json), enforced inside
 every fit and P2 simulation and before every start, retry and acceptance."""
    freeze_sha, fr = verify_freeze(group)
    if sorted(exps) != fr["exps"]:
        raise SystemExit(f"group {group} was frozen with {fr['exps']}")
    os.makedirs(OUT + "jobs", exist_ok=True)
    dl_file = OUT + f"r4e_deadline_{group}.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, exps=sorted(exps), freeze=freeze_sha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    J, man = job_list(exps)
    if len({j["key"] for j in J}) != len(J) or len(J) != fr["n_jobs"]:
        raise SystemExit("job list differs from the freeze or keys not unique")
    J.sort(key=lambda j: (j["seed"], j["exp"], j["tag"]))
    args = [(dict(j, group=group), job_prov(j, freeze_sha, fr), dl["deadline"]) for j in J]
    print("jobs", len(J), "group", group, "deadline", time.strftime("%Y-%m-%d %H:%M", time.localtime(dl["deadline"])), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, args, chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


# ---------------------------------------------------------------- E0m: matched-noise L1 arm
def e0m_job(k):
    import luppi_r4e_lib as Lib, luppi_hopf as L
    sim = Lib.make_driven_simulator(100, Lib.T_LEN, 3000.0)
    C = np.load(OUT + f"truth/SG_{k}_C.npy"); f = np.load(OUT + f"truth/SG_{k}_f.npy"); cfg = dict(L.RELEASED)
    tgt = Lib.simulate(C, f, 6_000_000 + k, sim, cfg=cfg); ho = Lib.simulate(C, f, 7_000_000 + k, sim, cfg=cfg)
    out = []
    for n, a in (("target", tgt), ("heldout", ho)):
        p = f"truth/SG_{k}_sig0.001_{n}.npy"; np.save(OUT + p, a); out.append(dict(file=p, sha=sha(OUT + p)))
    real, mfc = realistic(tgt)
    return dict(tag=f"SG_{k}", valid=valid_series(tgt) and valid_series(ho), realistic=real, mean_fc=mfc, files=out)


def stage_e0m(procs):
    with mp.Pool(3) as p:
        res = p.map(e0m_job, range(3))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), sigma=0.001, namespaces=dict(target="6e6+k", heldout="7e6+k"),
                   records=res, files=[f for r in res for f in r["files"]], code={c: sha(c) for c in CODE}),
              open(OUT + "r4e_manifest_e0m.json", "w"), indent=1, default=float)
    for r in res:
        print(r["tag"], "valid", r["valid"], "mean FC %.3f" % r["mean_fc"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("exps", nargs="*"); ap.add_argument("--procs", type=int, default=56)
    ap.add_argument("--group", default="primary")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"truths": lambda: stage_truths(a.procs), "calibrate": lambda: stage_calibrate(a.procs), "e0m": lambda: stage_e0m(a.procs),
     "freeze": lambda: stage_freeze(a.group, a.exps), "run": lambda: stage_run(a.exps, a.procs, a.group)}[a.stage]()
