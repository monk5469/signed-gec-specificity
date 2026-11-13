"""Human fitting protocol: input preparation, job freezing and fitting of the signed and cooperative-only models (the e5 stages),
used by the PPC and the later human analyses; also the multi-mode driven simulator."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e_ext/"
R4E = DATA_ROOT + "results/p33/r4e/"
HCP = DATA_ROOT + "work/p33_ts/"
SUBJ_JSON = DATA_ROOT + "results/p33/p33_hcp_s200_v13.json"
DEMO = DATA_ROOT + "raw/luppi_demo/"
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e6.py"]
R4V2 = DATA_ROOT + "results/p33/r4v2/"
RAW = [DEMO + "gec_3/FMRI.mat", DEMO + "gec_3/SC.mat", DEMO + "single/SC.mat", SUBJ_JSON, R4E + "r4e_manifest.json"] + \
      [R4V2 + f"C_s{k}_header_signed_matlab_seed{sd}.npy" for k in range(3) for sd in (11, 12, 13)]     # reference fitted matrices
SEEDS = (11, 12, 13)
CELLS = {"S04": (1, 0.40), "S06": (1, 0.60), "M02": (3, 0.20), "M04": (3, 0.40)}
E2B_TRUTHS = (1001, 1002, 1003, 1004, 1005, 1006)
TAU = 50.0
DEADLINE_H = 72.0
DROP = 11


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------- multi-mode driven simulator
def make_multi_simulator(N, T, burnin, M):
    """As luppi_r4e_lib.make_driven_simulator (same system-noise key schedule and per-step N(0, 1) draws of shape (2, N)),
 with M independent unit-variance OU drives d_m (exact AR(1) per Euler step) entering the x equation as s * G @ d."""
    import jax, jax.numpy as jnp
    from jax import lax
    import luppi_hopf as L
    TR = L.TR; dt = 0.1 * TR / 2; spt = int(round(TR / dt)); nb = L.burn_steps(burnin, dt, spt)

    @jax.jit
    def sim(C, f, a, sigma, key, G, s, tau_d, key_d):
        om = 2 * jnp.pi * f; sumC = C.sum(1); dsig = jnp.sqrt(dt) * sigma
        rho = jnp.exp(-dt / tau_d); q = jnp.sqrt(1 - rho ** 2)
        def step(state, kk):
            (x, y, d), (k, kd) = state, kk
            e = jax.random.normal(k, (2, N), dtype=jnp.float64); r2 = x * x + y * y
            nx = x + dt * (a * x - om * y - x * r2 + C @ x - sumC * x + s * (G @ d)) + dsig * e[0]
            ny = y + dt * (a * y + om * x - y * r2 + C @ y - sumC * y) + dsig * e[1]
            nd = rho * d + q * jax.random.normal(kd, (M,), dtype=jnp.float64)
            return (nx, ny, nd), None
        k1, k2 = jax.random.split(key); kd0, kd1, kd2 = jax.random.split(key_d, 3)
        st = (0.1 * jnp.ones(N), 0.1 * jnp.ones(N), jax.random.normal(kd0, (M,), dtype=jnp.float64))
        st, _ = lax.scan(step, st, (jax.random.split(k1, nb), jax.random.split(kd1, nb)))
        def tr_step(st, ks):
            st, _ = lax.scan(step, st, ks); return st, st[0]
        ks = (jax.random.split(k2, T * spt).reshape(T, spt, 2), jax.random.split(kd2, T * spt).reshape(T, spt, 2))
        _, xs = lax.scan(tr_step, st, ks)
        return xs.T
    return sim


def simulate(sim, C, f, seed, G, s, seed_d, cfg):
    import jax, jax.numpy as jnp
    return np.asarray(sim(jnp.asarray(C), jnp.asarray(f), cfg["a"], cfg["sigma"], jax.random.PRNGKey(seed), jnp.asarray(G), float(s), TAU,
                          jax.random.PRNGKey(seed_d)))


def loadings(SC, M):
    """Eigenvectors 2..M+1 of the SC graph Laplacian, each scaled to max |g| = 1 with a fixed sign convention; (N, M)."""
    A = (SC + SC.T) / 2; Lap = np.diag(A.sum(1)) - A; w, V = np.linalg.eigh(Lap); G = V[:, 1:1 + M].copy()
    for m in range(M):
        g = G[:, m] / np.abs(G[:, m]).max(); G[:, m] = g if g[np.flatnonzero(g)[0]] > 0 else -g
    return G


# ---------------------------------------------------------------- shared-drive calibration and targets
def drive_share(sim, C, f, G, s_drive, j, cfg):
    x0 = simulate(sim, C, f, 8_500_000 + j, G, 0.0, 8_600_000 + j, cfg); xs = simulate(sim, C, f, 8_500_000 + j, G, s_drive, 8_600_000 + j, cfg)
    return float(np.median((xs.var(1) - x0.var(1)) / xs.var(1)))


def calib_job(cell):
    import luppi_hopf as L, luppi_r4e_lib as Lib
    M, q = CELLS[cell]; sim = make_multi_simulator(100, Lib.T_LEN, 3000.0, M); cfg = L.HEADER
    C = np.load(R4E + "truth/QM1_1001_C.npy"); f = np.load(R4E + "truth/QM1_1001_f.npy")
    _, SC, _ = Lib.demo_subject((1001 - 1) % 3); G = loadings(SC, M)
    lo, hi = -4.0, 1.0; trace = []
    for _ in range(30):
        mid = (lo + hi) / 2; val = float(np.mean([drive_share(sim, C, f, G, 10 ** mid, j, cfg) for j in range(4)])); trace.append((mid, val))
        lo, hi = (mid, hi) if val < q else (lo, mid)
    best = min(trace, key=lambda t: abs(t[1] - q))
    return dict(cell=cell, modes=M, q_target=q, s=10 ** best[0], q=best[1], success=bool(abs(best[1] - q) <= 0.1 * q), trace=trace)


def target_job(args):
    s, ci, cell, s_drive = args
    import luppi_hopf as L, luppi_r4e_lib as Lib
    M, _ = CELLS[cell]; sim = make_multi_simulator(100, Lib.T_LEN, 3000.0, M); cfg = L.HEADER
    tag = f"QM1_{s}"; C = np.load(R4E + f"truth/{tag}_C.npy"); f = np.load(R4E + f"truth/{tag}_f.npy")
    G = np.load(OUT + f"truth/G{M}_stratum{(s - 1) % 3}.npy"); kd = 8_000_000 + 100 * s + 10 * ci
    tgt = simulate(sim, C, f, 1_000_000 + s, G, s_drive, kd, cfg); ho = simulate(sim, C, f, 2_000_000 + s, G, s_drive, kd + 500_000, cfg)
    out = []
    for n, a in (("target", tgt), ("heldout", ho)):
        p = f"truth/{tag}_{cell}_{n}.npy"; np.save(OUT + p, a); out.append(dict(file=p, sha=sha(OUT + p)))
    Fm = np.corrcoef(L.prep_emp(tgt))[~np.eye(100, dtype=bool)]
    return dict(tag=tag, cell=cell, files=out, target_mean_fc=float(Fm.mean()), target_neg_fc=float(np.mean(Fm < 0)),
                finite=bool(np.all(np.isfinite(tgt)) and np.all(np.isfinite(ho))))


def compare_previous(name, files):
    """Provenance layer: compare regenerated file hashes with the previous manifest, if any."""
    if not os.path.exists(OUT + name):
        return dict(previous=False)
    old = {r["file"]: r["sha"] for r in json.load(open(OUT + name))["files"]}
    same = [f["file"] for f in files if old.get(f["file"]) == f["sha"]]
    return dict(previous=True, n_files=len(files), n_identical=len(same), all_identical=len(same) == len(files) == len(old))


def stage_e2b_calibrate(procs):
    import luppi_r4e_lib as Lib, luppi_hopf as L
    os.makedirs(OUT + "truth", exist_ok=True); gfiles = []
    for k in range(3):
        _, SC, _ = Lib.demo_subject(k)
        for M in (1, 3):
            p = f"truth/G{M}_stratum{k}.npy"; np.save(OUT + p, loadings(SC, M)); gfiles.append(dict(file=p, sha=sha(OUT + p)))
    # pairing check: with s = 0 the simulator must reproduce the frozen undriven target and held-out series of every
    # driven truth exactly, for both mode counts
    pair_diff = 0.0
    for M in (1, 3):
        sim = make_multi_simulator(100, Lib.T_LEN, 3000.0, M)
        for s in E2B_TRUTHS:
            C = np.load(R4E + f"truth/QM1_{s}_C.npy"); f = np.load(R4E + f"truth/QM1_{s}_f.npy"); G = np.load(OUT + f"truth/G{M}_stratum{(s - 1) % 3}.npy")
            for key, n in ((1_000_000 + s, "target"), (2_000_000 + s, "heldout")):
                x = simulate(sim, C, f, key, G, 0.0, 1, L.HEADER)
                pair_diff = max(pair_diff, float(np.max(np.abs(x - np.load(R4E + f"truth/QM1_{s}_{n}.npy")))))
    if pair_diff > 1e-12:
        raise SystemExit(f"s = 0 does not reproduce the E1 target (max diff {pair_diff})")
    with mp.Pool(4) as p:
        cal = p.map(calib_job, list(CELLS), chunksize=1)
    jobs = [(s, ci, c["cell"], c["s"]) for s in E2B_TRUTHS for ci, c in enumerate(cal)]
    with mp.Pool(procs) as p:
        tg = p.map(target_job, jobs, chunksize=1)
    files = gfiles + [f for t in tg for f in t["files"]]
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), pairing_check_max_diff=pair_diff, calibration=cal, targets=tg, files=files,
                   code={c: sha(c) for c in CODE if os.path.exists(c)}, regeneration=compare_previous("r4e_ext_e2b_manifest.json", files)),
              open(OUT + "r4e_ext_e2b_manifest.json", "w"), indent=1, default=float)
    for c in cal:
        print(c["cell"], "modes", c["modes"], "q*", c["q_target"], "-> s %.4g q %.4f success %s" % (c["s"], c["q"], c["success"]), flush=True)
    print("pairing check max diff", pair_diff, "| target mean FC range %.3f-%.3f" % (min(t["target_mean_fc"] for t in tg), max(t["target_mean_fc"] for t in tg)), flush=True)


# ---------------------------------------------------------------- PPC preparation
def subjects():
    return [s["subject"] for s in json.load(open(SUBJ_JSON))["subjects"]]


def stage_e5_prepare():
    import scipy.io as sio
    import luppi_hopf as L, luppi_r4e_lib as Lib
    os.makedirs(OUT + "e5", exist_ok=True); files = []
    np.save(OUT + "e5/SC.npy", sio.loadmat(DEMO + "single/SC.mat")["SC"].astype(float)); files.append(dict(file="e5/SC.npy", sha=sha(OUT + "e5/SC.npy")))
    for ses in ("REST1", "REST2"):
        lr = []
        for sub in subjects():
            for d, kind in (("LR", "input"), ("RL", "heldout")):
                x = np.load(HCP + f"{sub}_{ses}_{d}_s100.npy").T.astype(float)[:, DROP:]
                y = Lib.bandpass(x); p = f"e5/{sub}_{ses}_{kind}.npy"; np.save(OUT + p, y); files.append(dict(file=p, sha=sha(OUT + p)))
                if kind == "input":
                    lr.append(y)
        f = L.regional_frequencies(lr); p = f"e5/freq_{ses}.npy"; np.save(OUT + p, f); files.append(dict(file=p, sha=sha(OUT + p)))
        print(ses, "subjects", len(lr), "frames", lr[0].shape, "freq range %.4f-%.4f Hz" % (f.min(), f.max()), flush=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), subjects=subjects(), drop=DROP, band=[0.008, 0.09], files=files,
                   code={c: sha(c) for c in CODE if os.path.exists(c)}, regeneration=compare_previous("r4e_ext_e5_manifest.json", files),
                   hcp_inputs={f"{s}_{ses}_{d}": sha(HCP + f"{s}_{ses}_{d}_s100.npy") for s in subjects() for ses in ("REST1", "REST2") for d in ("LR", "RL")}),
              open(OUT + "r4e_ext_e5_manifest.json", "w"), indent=1)


# ---------------------------------------------------------------- jobs
def job_list(group):
    J = []
    if group == "e2b":
        for s in E2B_TRUTHS:
            for cell in CELLS:
                for br in "RF":
                    for sd in SEEDS:
                        for mdl in ("signed", "coop"):
                            J.append(dict(exp="E2b", tag=f"QM1_{s}", family="QM1", stratum=(s - 1) % 3, branch=br, cell=cell, seed=sd, model=mdl))
    elif group == "e5":
        for i, sub in enumerate(subjects()):
            for ses in ("REST1", "REST2"):
                for mdl in ("signed", "coop"):
                    J.append(dict(exp="E5", tag=str(sub), subject_index=i, session=ses, seed=11, model=mdl))
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j))
        j["file"] = OUT + f"jobs_{group}/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    if j["exp"] == "E2b":
        return [("r4e", f"truth/{j['tag']}_C.npy"), ("r4e", f"truth/{j['tag']}_f.npy"),
                ("ext", f"truth/{j['tag']}_{j['cell']}_target.npy"), ("ext", f"truth/{j['tag']}_{j['cell']}_heldout.npy")]
    return [("ext", "e5/SC.npy"), ("ext", f"e5/freq_{j['session']}.npy"), ("ext", f"e5/{j['tag']}_{j['session']}_input.npy"),
            ("ext", f"e5/{j['tag']}_{j['session']}_heldout.npy")]


def path_of(root, rel):
    return (R4E if root == "r4e" else OUT) + rel


_SIM = {}


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if "sim" not in _SIM:
        _SIM["sim"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["sim"]; t0 = time.time(); cfg = dict(L.HEADER)
    if j["exp"] == "E2b":
        C_true = np.load(R4E + f"truth/{j['tag']}_C.npy"); _, SC, _ = Lib.demo_subject(j["stratum"])
        tgt = np.load(OUT + f"truth/{j['tag']}_{j['cell']}_target.npy"); ho = np.load(OUT + f"truth/{j['tag']}_{j['cell']}_heldout.npy")
        inp = Lib.bandpass(tgt) if j["branch"] == "F" else tgt; f = L.regional_frequencies([inp]); hop = Lib.branch(ho, j["branch"])
        base = 4_000_000 + 1000 * int(j["tag"].split("_")[1]) + 10 * j["seed"]
    else:
        SC = np.load(OUT + "e5/SC.npy"); inp = np.load(OUT + f"e5/{j['tag']}_{j['session']}_input.npy"); f = np.load(OUT + f"e5/freq_{j['session']}.npy")
        hop = L.prep_emp(np.load(OUT + f"e5/{j['tag']}_{j['session']}_heldout.npy")); C_true = None
        base = 9_000_000 + 1000 * j["subject_index"] + (0 if j["session"] == "REST1" else 500)
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    sims = []
    for r in range(8):                                    # same evaluation keys for signed / coop (paired)
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    np.save(j["file"].replace(".json", "_C.npy"), C)
    rec = dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"],
               mse_ho=Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm), neg=L.neg_stats(C, SC), C_sha=sha_arr(C))
    if C_true is not None:
        rec["metrics"] = Lib.edge_metrics(C, C_true, SC)
    return rec


# ---------------------------------------------------------------- freezing (same contract as luppi_r4e_run, per group)
def freeze_path(group):
    return OUT + f"r4e_ext_freeze_{group}.json"


def manifest_hashes():
    exp = {}
    man = json.load(open(R4E + "r4e_manifest.json"))
    for t in man["truths"]:
        if not t.get("missing"):
            for n, h in t["sha"].items():
                exp[("r4e", f"truth/{t['tag']}_{n}.npy")] = h
    for m in ("r4e_ext_e2b_manifest.json", "r4e_ext_e5_manifest.json"):
        if os.path.exists(OUT + m):
            for rec in json.load(open(OUT + m))["files"]:
                exp[("ext", rec["file"])] = rec["sha"]
    return exp


def stage_freeze(group):
    fp = freeze_path(group)
    if os.path.exists(fp):
        raise SystemExit(f"{fp} exists")
    J = job_list(group); man = manifest_hashes(); files = {}
    for j in J:
        for root, rel in job_files(j):
            if (root, rel) not in man or sha(path_of(root, rel)) != man[(root, rel)]:
                raise SystemExit(f"{root}:{rel} missing from manifests or changed")
            files[f"{root}:{rel}"] = man[(root, rel)]
    if group == "e2b":                                   # loadings used by the analysis
        for (root, rel), h in man.items():
            if root == "ext" and rel.startswith("truth/G"):
                if sha(path_of(root, rel)) != h:
                    raise SystemExit(f"{rel} changed")
                files[f"{root}:{rel}"] = h
    mans = {m: sha(OUT + m) for m in ("r4e_ext_e2b_manifest.json", "r4e_ext_e5_manifest.json") if os.path.exists(OUT + m)}
    fr = dict(group=group, n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE},
              raw={p: sha(p) for p in RAW}, manifests=mans, files=files)
    json.dump(fr, open(fp, "x"), indent=1)
    print("frozen", group, sha(fp), "jobs", len(J), "files", len(files), flush=True)


def verify_freeze(group):
    fp = freeze_path(group); fr = json.load(open(fp))
    checks = [(c, h, c) for c, h in fr["code"].items()] + [(p, h, p) for p, h in fr["raw"].items()] + \
             [(m, h, OUT + m) for m, h in fr["manifests"].items()] + [(k, h, path_of(*k.split(":", 1))) for k, h in fr["files"].items()]
    bad = [n for n, h, p in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed or missing, e.g. {bad[:3]}")
    return sha(fp), fr


def job_prov(j, fsha, fr):
    keys = [f"{r}:{p}" for r, p in job_files(j)]
    return dict(freeze=fsha, group=fr["group"], inputs={k: fr["files"][k] for k in keys})


def _write(j, rec):
    json.dump(rec, open(j["file"], "w"), indent=1, default=float); return rec["state"]


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
            return _write(j, dict(j, state="cancelled", prov=prov, reason="deadline before start" if attempt == 1 else "deadline before retry", attempts=attempt - 1, errors=errors))
        try:
            r = run_job(j, deadline)
            if time.time() > deadline:
                return _write(j, dict(j, state="cancelled", prov=prov, reason="deadline passed before acceptance", attempts=attempt, errors=errors))
            r.update(state="done", prov=prov, attempts=attempt, errors=errors); return _write(j, r)
        except L.Cancelled as e:
            return _write(j, dict(j, state="cancelled", prov=prov, reason=str(e), attempts=attempt, errors=errors))
        except Exception:
            errors.append(traceback.format_exc())
    return _write(j, dict(j, state="failed_twice", prov=prov, attempts=2, errors=errors))


def stage_run(group, procs):
    fsha, fr = verify_freeze(group); os.makedirs(OUT + f"jobs_{group}", exist_ok=True)
    dl_file = OUT + f"r4e_ext_deadline_{group}.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file)); J = job_list(group)
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze or exceeds start + 72 h")
    if len(J) != fr["n_jobs"] or len({j["key"] for j in J}) != len(J):
        raise SystemExit("job list differs from the freeze or keys not unique")
    J.sort(key=lambda j: (j["seed"], j.get("cell", j.get("session")), j["tag"]))
    args = [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J]
    print("jobs", len(J), "group", group, flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, args, chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("group", nargs="?"); ap.add_argument("--procs", type=int, default=56)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"e2b_calibrate": lambda: stage_e2b_calibrate(a.procs), "e5_prepare": stage_e5_prepare, "freeze": lambda: stage_freeze(a.group),
     "run": lambda: stage_run(a.group, a.procs)}[a.stage]()
