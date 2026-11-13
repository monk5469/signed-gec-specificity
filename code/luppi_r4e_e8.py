"""Observation-noise and hidden-drive functions used by the known-truth simulations."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e_e8/"
E7 = DATA_ROOT + "results/p33/r4e_e7/"
DEMO = DATA_ROOT + "raw/luppi_demo/"
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py", "luppi_r4e_e8_analyze.py"]
ATLAS = DATA_ROOT + "raw/p33_hcp/Schaefer2018_100Parcels_7Networks_order.dlabel.nii"
R4V2 = DATA_ROOT + "results/p33/r4v2/"
RAW = [DEMO + "gec_3/FMRI.mat", DEMO + "gec_3/SC.mat", E7 + "r4e_e7_manifest.json", E7 + "r4e_e7_freeze.json", ATLAS] + \
      [R4V2 + f"C_s{k}_header_signed_matlab_seed{sd}.npy" for k in range(3) for sd in (11, 12, 13)]       #: real-data reference fits
GEN_CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_r4e_ext.py", "luppi_r4e_e8.py"]                     #: generation lineage
SNR = {"1": 1.0, "2": 2.0, "4": 4.0, "8": 8.0}
CELLS = {"N1": (0, "1"), "N2": (0, "2"), "N4": (0, "4"), "N8": (0, "8"),
         "S1": (1, "1"), "S2": (1, "2"), "S4": (1, "4"), "M1": (3, "1"), "M2": (3, "2"), "M4": (3, "4")}
DRIVE_SHARE = 0.40
SEEDS = tuple(range(5001, 5031))
T = 1189
DEADLINE_H = 72.0
FREEZE = OUT + "r4e_e8_freeze.json"; PREREG = OUT + "r4e_e8_analysis_prereg.json"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


_SIM = {}


def msim(M):
    import luppi_r4e_ext as X
    k = max(M, 1)
    if k not in _SIM:
        _SIM[k] = X.make_multi_simulator(100, T, 3000.0, k)
    return _SIM[k]


def truth(s):
    return np.load(E7 + f"truth/SCG_{s}_C.npy"), np.load(E7 + f"truth/SCG_{s}_f.npy"), (s - 1) % 3


def G_of(stratum, M):
    import luppi_r4e_lib as Lib, luppi_r4e_ext as X
    _, SC, _ = Lib.demo_subject(stratum); return X.loadings(SC, max(M, 1))


def add_noise(x, snr, rng):
    """White observation noise with in-band (0.008-0.09 Hz) SNR = snr per region."""
    import luppi_r4e_lib as Lib
    e = rng.standard_normal(x.shape); scale = np.sqrt(Lib.bandpass(x).var(1) / (snr * Lib.bandpass(e).var(1)))
    return x + scale[:, None] * e


def generate(s, cell, held, s_drive):
    import luppi_hopf as L, luppi_r4e_ext as X
    M, snr_key = CELLS[cell]; C, f, k = truth(s); G = G_of(k, M)
    key = 14_000_000 + s + (500_000 if held else 0); kd = 15_000_000 + 100 * s + 10 * {0: 0, 1: 1, 3: 3}[M] + (500_000 if held else 0)
    x = X.simulate(msim(M), C, f, key, G, s_drive if M else 0.0, kd, L.HEADER)
    return add_noise(x, SNR[snr_key], np.random.default_rng([s, int(snr_key), 2 if held else 1]))


def calibrate(M):
    import luppi_hopf as L, luppi_r4e_ext as X
    C, f, k = truth(5001); G = G_of(k, M); lo, hi = -4.0, 1.0; trace = []
    for _ in range(30):
        mid = (lo + hi) / 2
        val = float(np.mean([X.drive_share(msim(M), C, f, G, 10 ** mid, j, L.HEADER) for j in range(4)])); trace.append((mid, val))
        lo, hi = (mid, hi) if val < DRIVE_SHARE else (lo, mid)
    best = min(trace, key=lambda t: abs(t[1] - DRIVE_SHARE))
    return dict(modes=M, s=10 ** best[0], q=best[1], success=bool(abs(best[1] - DRIVE_SHARE) <= 0.1 * DRIVE_SHARE), trace=trace)


def drive_band_share(s, cell, s_drive):
    """Realised IN-BAND (0.008-0.09 Hz) drive share of the target, before observation noise:
 median over regions of (var_bp(x_drive) - var_bp(x_0)) / var_bp(x_drive), same system and drive keys."""
    import luppi_hopf as L, luppi_r4e_ext as X, luppi_r4e_lib as Lib
    M, _ = CELLS[cell]; C, f, k = truth(s); G = G_of(k, M); kd = 15_000_000 + 100 * s + 10 * M
    x0 = Lib.bandpass(X.simulate(msim(M), C, f, 14_000_000 + s, G, 0.0, kd, L.HEADER)); xs = Lib.bandpass(X.simulate(msim(M), C, f, 14_000_000 + s, G, s_drive, kd, L.HEADER))
    return float(np.median((xs.var(1) - x0.var(1)) / xs.var(1)))


def target_job(args):
    s, cell, sd = args
    out = []
    band_share = drive_band_share(s, cell, sd) if CELLS[cell][0] else 0.0
    for held, n in ((False, "target"), (True, "heldout")):
        x = generate(s, cell, held, sd)
        if not (np.all(np.isfinite(x)) and np.all(x.std(1) > 0)):
            return dict(tag=f"SCG_{s}", cell=cell, valid=False, files=[], drive_band_share=band_share)
        p = f"truth/SCG_{s}_{cell}_{n}.npy"; np.save(OUT + p, x); out.append(dict(file=p, sha=sha(OUT + p)))
    return dict(tag=f"SCG_{s}", cell=cell, valid=True, files=out, drive_band_share=band_share)


def stage_prepare(procs):
    import luppi_hopf as L, luppi_r4e_ext as X
    os.makedirs(OUT + "truth", exist_ok=True)
    # pairing check: s = 0 with 1 or 3 modes reproduces the same Hopf series (drive draws use separate keys)
    C, f, k = truth(5001)
    a = X.simulate(msim(1), C, f, 14_005_001, G_of(k, 1), 0.0, 1, L.HEADER); b = X.simulate(msim(3), C, f, 14_005_001, G_of(k, 3), 0.0, 2, L.HEADER)
    pair = float(np.max(np.abs(a - b)))
    if pair > 1e-12:
        raise SystemExit(f"pairing check failed ({pair})")
    with mp.Pool(2) as p:
        cal = dict(zip((1, 3), p.map(calibrate, (1, 3))))
    jobs = [(s, c, cal[CELLS[c][0]]["s"] if CELLS[c][0] else 0.0) for s in SEEDS for c in CELLS]
    with mp.Pool(procs) as p:
        tg = p.map(target_job, jobs, chunksize=1)
    import nibabel as nib                                # Yeo-7 network of each Schaefer parcel
    lt = nib.load(ATLAS).header.get_axis(0).label[0]; nets = [lt[i][0].split("_")[2] for i in range(1, 101)]
    json.dump(dict(networks=nets, source=ATLAS), open(OUT + "networks.json", "w"), indent=1)
    files = [f for t in tg for f in t["files"]] + [dict(file="networks.json", sha=sha(OUT + "networks.json"))]
    old = {r["file"]: r["sha"] for r in json.load(open(OUT + "r4e_e8_manifest.json"))["files"]} if os.path.exists(OUT + "r4e_e8_manifest.json") else {}
    regen = dict(previous=bool(old), n_identical=sum(old.get(f["file"]) == f["sha"] for f in files), n_files=len(files))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), pairing_check_max_diff=pair, calibration={str(k): v for k, v in cal.items()},
                   targets=tg, files=files, code={c: sha(c) for c in CODE if os.path.exists(c)}, generation_code={c: sha(c) for c in GEN_CODE}, regeneration=regen,
                   namespaces=dict(system="14e6+s (+5e5 held-out)", drive="15e6+100s+10M (+5e5)", noise="default_rng([s, snr, 1|2])")),
              open(OUT + "r4e_e8_manifest.json", "w"), indent=1, default=float)
    print("pairing", pair, "| calibration", {k: (round(v["s"], 5), round(v["q"], 3), v["success"]) for k, v in cal.items()},
          "| targets valid", sum(t["valid"] for t in tg), "/", len(tg), "| in-band drive share S %.3f M %.3f" % (
              np.median([t["drive_band_share"] for t in tg if t["cell"].startswith("S")]), np.median([t["drive_band_share"] for t in tg if t["cell"].startswith("M")])),
          "| networks", sorted(set(nets)), "| regeneration", regen, flush=True)


# ---------------------------------------------------------------- jobs / freeze / run
def job_list():
    J = []
    for s in SEEDS:
        for c in CELLS:
            for mdl in ("signed", "coop"):
                J.append(dict(exp="E8", family="SCG", tag=f"SCG_{s}", stratum=(s - 1) % 3, cell=c, branch="F", seed=11, model=mdl))
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    return [E7 + f"truth/{j['tag']}_C.npy", E7 + f"truth/{j['tag']}_f.npy", OUT + f"truth/{j['tag']}_{j['cell']}_target.npy", OUT + f"truth/{j['tag']}_{j['cell']}_heldout.npy"]


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if "fit" not in _SIM:
        _SIM["fit"] = L.make_simulator(100, T, 3000.0)
    sim = _SIM["fit"]; t0 = time.time(); cfg = dict(L.HEADER)
    C_true, _, k = truth(int(j["tag"].split("_")[1])); _, SC, _ = Lib.demo_subject(k)
    tgt = np.load(OUT + f"truth/{j['tag']}_{j['cell']}_target.npy"); ho = np.load(OUT + f"truth/{j['tag']}_{j['cell']}_heldout.npy")
    inp = Lib.bandpass(tgt); f = L.regional_frequencies([inp]); hop = Lib.branch(ho, "F")
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    base = 16_000_000 + 1000 * int(j["tag"].split("_")[1]) + 10 * list(CELLS).index(j["cell"]); sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm); met = Lib.edge_metrics(C, C_true, SC)
    if not (np.all(np.isfinite(C)) and np.isfinite(mse) and np.isfinite(met["frac_neg"])):
        raise ValueError("non-finite fitted matrix or endpoint input")
    np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"], mse_ho=mse, metrics=met, C_sha=sha_arr(C))


def stage_freeze():
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("luppi_r4e_e8_analyze.py"):
        raise SystemExit("analysis pre-registration missing or not matching")
    mf = json.load(open(OUT + "r4e_e8_manifest.json"))
    if not all(v["success"] for v in mf["calibration"].values()):
        raise SystemExit("drive calibration unsuccessful")
    if mf.get("generation_code") != {c: sha(c) for c in GEN_CODE}:                      #: lineage
        raise SystemExit("generation code changed since the targets were prepared")
    man = {OUT + r["file"]: r["sha"] for r in mf["files"]}
    e7 = {E7 + r["file"]: r["sha"] for r in json.load(open(E7 + "r4e_e7_manifest.json"))["files"]}; man.update(e7)
    valid = {(t["tag"], t["cell"]) for t in json.load(open(OUT + "r4e_e8_manifest.json"))["targets"] if t["valid"]}
    J = [j for j in job_list() if (j["tag"], j["cell"]) in valid]; files = {}
    for j in J:
        for p in job_files(j):
            if p not in man or sha(p) != man[p]:
                raise SystemExit(f"{p} missing from manifests or changed")
            files[p] = man[p]
    fr = dict(group="e8", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), code={c: sha(c) for c in CODE}, raw={p: sha(p) for p in RAW},
              manifest=sha(OUT + "r4e_e8_manifest.json"), prereg=sha(PREREG), eligible=sorted(f"{a}|{b}" for a, b in valid),
              files=dict(files, **{OUT + "networks.json": man[OUT + "networks.json"]}))
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen e8", sha(FREEZE), "jobs", len(J), "files", len(files), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(OUT + "r4e_e8_manifest.json", fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="e8", inputs={p: fr["files"][p] for p in job_files(j)})


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
            return w(dict(j, state="cancelled", prov=prov, reason="deadline before start" if attempt == 1 else "deadline before retry", attempts=attempt - 1, errors=errors))
        try:
            r = run_job(j, deadline)
            if time.time() > deadline:
                return w(dict(j, state="cancelled", prov=prov, reason="deadline passed before acceptance", attempts=attempt, errors=errors))
            r.update(state="done", prov=prov, attempts=attempt, errors=errors); return w(r)
        except L.Cancelled as e:
            return w(dict(j, state="cancelled", prov=prov, reason=str(e), attempts=attempt, errors=errors))
        except Exception:
            errors.append(traceback.format_exc())
    return w(dict(j, state="failed_twice", prov=prov, attempts=2, errors=errors))


def stage_run(procs):
    fsha, fr = verify_freeze(); os.makedirs(OUT + "jobs", exist_ok=True)
    dl_file = OUT + "r4e_e8_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    ok = set(fr["eligible"]); J = [j for j in job_list() if f"{j['tag']}|{j['cell']}" in ok]; J.sort(key=lambda j: (j["cell"], j["tag"]))
    print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"prepare": lambda: stage_prepare(a.procs), "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
