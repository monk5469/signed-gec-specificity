"""Participant-mask analysis (Section 2.4.1): builds accS, the acc condition with WM and CSF pools taken from each participant's own
wmparc.2 segmentation, and fits the signed and cooperative-only models with the PPC protocol."""
from paths import DATA_ROOT
import sys, os, json, time, hashlib, argparse, traceback, multiprocessing as mp
import numpy as np

OUT = DATA_ROOT + "results/p33/b1/"; P33 = DATA_ROOT + "results/p33/"; RAW = DATA_ROOT + "raw/p33_hcp_vol/"; WP = DATA_ROOT + "raw/p33_wmparc2/"; TEMPLATE = DATA_ROOT + "raw/p33_hcp_vol/100307_Atlas_wmparc.2.nii.gz"
WORK = DATA_ROOT + "work/"; TS = WORK + "p33_ts_accS2/"; LINEAGE = P33 + "ppc_lineage_manifest.json"; SUBJ_JSON = P33 + "p33_hcp_s200_v13.json"; DEMOSET = {"100307", "100408", "101107"}
SERIES_MAN = OUT + "b1_series_manifest.json"; MAN = OUT + "b1_manifest.json"; FREEZE = OUT + "b1_freeze.json"; PREREG = P33 + "b1_analyze_prereg.json"
SC_SRC = DATA_ROOT + "raw/luppi_demo/single/SC.mat"; DROP = 11; DEADLINE_H = 72.0
CODE = ["luppi_hopf.py", "luppi_r4e_lib.py", "luppi_pipeline_vol.py", "luppi_ppc_prep.py", "luppi_r4e_ext.py", "b1_run.py", "b1_analyze.py"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def subjects():
    return [s["subject"] for s in json.load(open(SUBJ_JSON))["subjects"]]


def mask_path(s):
    return WP + f"{s}_wmparc.2.nii.gz"


# ---------------------------------------------------------------- stage 1: accS series (copy of the luppi_ppc_prep 'acc' path; mask swapped)
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def series_job(args):
    s, d, lin, etag, tmpl_sha = args
    import nibabel as nib
    import luppi_pipeline_vol as P, luppi_ppc_prep as PP
    vol = RAW + f"{s}_rfMRI_REST1_{d}.nii.gz"; bmf = RAW + f"{s}_REST1_{d}_brainmask_fs.2.nii.gz"; movf = RAW + f"{s}_REST1_{d}_Movement_Regressors.txt"
    mk = mask_path(s); motref = WORK + f"p33_ts_mot/{s}_REST1_{d}_s100.npy"
    try:
        if lin is None or not lin.get("ok"):
            raise ValueError("no successful PPC lineage record for this run")
        for p in (vol, bmf, movf):
            h = lin["inputs"].get(os.path.basename(p))
            if h is None or sha(p) != h:
                raise ValueError(f"lineage mismatch or missing hash: {os.path.basename(p)}")
        if sha(motref) != lin["cond"]["mot"]["npy_sha"]:
            raise ValueError("PPC mot reference differs from its lineage npy_sha")
        if etag is None or "-" in etag or md5(mk) != etag:
            raise ValueError("tissue mask md5 differs from the S3 ETag recorded at presign")
        ms = sha(mk); mask_check = "S3 ETag (md5)"
        if sha(TEMPLATE) != tmpl_sha:
            raise ValueError("template differs from the PPC lineage mask_source")
        imgs = {n: nib.load(p) for n, p in (("vol", vol), ("mask", mk), ("bm", bmf), ("atlas", P.ATLAS))}
        if len({tuple(i.shape[:3]) for i in imgs.values()}) != 1 or not all(np.allclose(i.affine, imgs["vol"].affine, atol=1e-3) for i in imgs.values()):
            raise ValueError("voxel grid / affine mismatch")
        X = np.asarray(imgs["vol"].dataobj, dtype=np.float32); lab = np.asarray(imgs["mask"].dataobj).astype(int)
        tl = np.asarray(nib.load(TEMPLATE).dataobj).astype(int); dice = lambda a, b: float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        sw, sc, tw, tc = np.isin(lab, P.WM_LABELS), np.isin(lab, P.CSF_LABELS), np.isin(tl, P.WM_LABELS), np.isin(tl, P.CSF_LABELS)
        if tl.shape != lab.shape or np.array_equal(sw, tw) or np.array_equal(sc, tc):
            raise ValueError("subject tissue sets equal the template (or grid differs): not a subject-specific mask")
        dice_wm, dice_csf = dice(sw, tw), dice(sc, tc)
        bm = np.asarray(imgs["bm"].dataobj) > 0; atl = np.asarray(imgs["atlas"].dataobj).astype(int); mot = np.loadtxt(movf); T = X.shape[3]
        if T != 1200 or mot.shape != (T, 12):
            raise ValueError(f"frames {T} motion {mot.shape}")
        Y, gs, pools = P.prepare(X, lab, bm, atl); del X
        t = np.arange(T, dtype=float); base = np.c_[np.ones(T), t]; art = P.art_dummies(gs, mot, "int"); Vw, Vc = pools[0]
        mplus = np.c_[base, mot[:, :12], art]; acc_t = np.c_[P.comps(Vw, "mean4", 5), P.comps(Vc, "mean4", 5)]
        nz = lambda M: M / np.maximum(np.linalg.norm(M, axis=0), 1e-300)
        rk = lambda M: int(np.linalg.matrix_rank(nz(M), tol=PP.RANK_TOL * np.linalg.svd(nz(M), compute_uv=False)[0]))
        inc = rk(np.c_[mplus, acc_t]) - rk(mplus)
        if inc != 10:
            raise ValueError(f"incremental design rank {inc} (expected 10)")
        resid = lambda C: Y.T - C @ np.linalg.lstsq(C, Y.T, rcond=None)[0]
        Rm = resid(mplus); ref = np.load(motref)
        idd = float(np.max(np.abs(Rm - ref)) / max(float(ref.std()), 1e-300))
        if idd > 1e-6:
            raise ValueError(f"identity check failed: mot residual differs from PPC (rel {idd:.2e})")
        R = resid(np.c_[mplus, acc_t])
        if not (np.all(np.isfinite(R)) and R.std(0).min() > 0):
            raise ValueError("non-finite or constant output")
        os.makedirs(TS, exist_ok=True); p = TS + f"{s}_REST1_{d}_s100.npy"; np.save(p, R)
        return dict(subject=s, run=d, ok=True, mask=os.path.basename(mk), mask_sha=ms, mask_check=mask_check, dice_wm_template=dice_wm, dice_csf_template=dice_csf, n_wm=int(Vw.shape[1]), n_csf=int(Vc.shape[1]),
                    n_wm_ppc=lin.get("n_wm"), n_csf_ppc=lin.get("n_csf"), incremental_rank=inc, identity_rel=idd, npy_sha=sha(p))
    except Exception as e:
        return dict(subject=s, run=d, ok=False, error=repr(e))


def lineage():
    """The PPC lineage manifest, verified against the copy pinned in the PPC acc freeze; code and atlas hashes must match the files used."""
    import luppi_r4e_ppc as W, luppi_pipeline_vol as P
    X = W.X; W.point("acc"); _, fr = X.verify_freeze("e5")
    if fr["raw"].get(LINEAGE) != sha(LINEAGE):
        raise SystemExit("PPC lineage manifest is not the one pinned in the PPC acc freeze")
    m = json.load(open(LINEAGE))
    if any(sha(c) != h for c, h in m["code"].items()) or sha(P.ATLAS) != m["atlas"]["sha"] or os.path.basename(P.ATLAS) != m["atlas"]["file"]:
        raise SystemExit("PPC preprocessing code or atlas differs from the lineage record")
    return m


def stage_series(procs):
    v = json.load(open(WP + "wmparc2_etag_verify.json"))
    if not v.get("all_ok") or v["n"] != 98 or v["etags_sha"] != sha(WP + "wmparc2_etags.txt"):
        raise SystemExit("wmparc download verification missing, failed, or not for the recorded ETag list")
    et = {l.split("\t")[0]: l.split("\t")[1] for l in open(WP + "wmparc2_etags.txt") if l.strip()}
    m = lineage(); rec = {(r["subject"], r["run"]): r for r in m["records"]}
    os.makedirs(OUT, exist_ok=True)
    args = [(s, d, rec.get((s, d)), et.get(f"{s}_wmparc.2.nii.gz"), m["mask_source"]["sha"]) for s in subjects() for d in ("LR", "RL")]
    with mp.Pool(procs) as p:
        res = p.map(series_job, args, chunksize=1)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), n=len(res), n_ok=sum(r["ok"] for r in res), etag_verify_sha=sha(WP + "wmparc2_etag_verify.json"),
                   etags_sha=sha(WP + "wmparc2_etags.txt"), template_sha=sha(TEMPLATE), lineage_sha=sha(LINEAGE), code={c: sha(c) for c in CODE if os.path.exists(c)}, records=res),
              open(SERIES_MAN, "w"), indent=1)
    print("runs", len(res), "ok", sum(r["ok"] for r in res), "bad", [(r["subject"], r["run"], r.get("error")) for r in res if not r["ok"]][:5], flush=True)


def verify_series():
    """ every series output equals its manifest record, and the producer code is unchanged since the series stage."""
    sm = json.load(open(SERIES_MAN))
    if sm["n_ok"] != 2 * len(subjects()) or sm["n"] != sm["n_ok"]:
        raise SystemExit("series incomplete")
    for c in ("b1_run.py", "luppi_pipeline_vol.py", "luppi_ppc_prep.py"):
        if sm["code"].get(c) != sha(c):
            raise SystemExit(f"producer code changed since the series stage: {c}")
    bad = [(r["subject"], r["run"]) for r in sm["records"] if sha(TS + f"{r['subject']}_REST1_{r['run']}_s100.npy") != r["npy_sha"]]
    if bad:
        raise SystemExit(f"{len(bad)} series outputs differ from the series manifest, e.g. {bad[:3]}")
    return sm


# ---------------------------------------------------------------- stage 2: PPC-style REST1 inputs (copy of luppi_r4e_ppc.stage_prepare_rest1)
def stage_prepare():
    import scipy.io as sio
    import luppi_hopf as L, luppi_r4e_lib as Lib
    verify_series(); os.makedirs(OUT + "e5", exist_ok=True); files = []
    np.save(OUT + "e5/SC.npy", sio.loadmat(SC_SRC)["SC"].astype(float)); files.append(dict(file="e5/SC.npy", sha=sha(OUT + "e5/SC.npy")))
    lr = []
    for sub in subjects():
        for d, kind in (("LR", "input"), ("RL", "heldout")):
            x = np.load(TS + f"{sub}_REST1_{d}_s100.npy").T.astype(float)[:, DROP:]
            y = Lib.bandpass(x); p = f"e5/{sub}_REST1_{kind}.npy"; np.save(OUT + p, y); files.append(dict(file=p, sha=sha(OUT + p)))
            if kind == "input":
                lr.append(y)
    f = L.regional_frequencies(lr); p = "e5/freq_REST1.npy"; np.save(OUT + p, f); files.append(dict(file=p, sha=sha(OUT + p)))
    import luppi_r4e_ppc as W                                            # frequency path, compared with the frozen PPC acc frequencies
    X = W.X; W.point("acc"); _, fr = X.verify_freeze("e5"); fa_path = X.OUT + "e5/freq_REST1.npy"
    if fr["files"].get("ext:e5/freq_REST1.npy") != sha(fa_path):
        raise SystemExit("PPC acc frequencies differ from their freeze")
    fa = np.load(fa_path); df = f - fa
    freq = dict(identical=bool(np.array_equal(f, fa)), max_abs_diff_hz=float(np.max(np.abs(df))), median_abs_diff_hz=float(np.median(np.abs(df))),
                per_region_diff_hz=df.tolist(), acc_freq_sha=sha(fa_path))
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), subjects=subjects(), series_manifest=sha(SERIES_MAN), drop=DROP, band=[0.008, 0.09], files=files,
                   frequency_vs_acc=freq, code={c: sha(c) for c in CODE if os.path.exists(c)}), open(MAN, "w"), indent=1)
    print("accS REST1 subjects", len(lr), "frames", lr[0].shape, "freq range %.4f-%.4f Hz" % (f.min(), f.max()),
          "| vs acc: identical", freq["identical"], "max |diff| %.5f Hz" % freq["max_abs_diff_hz"], flush=True)


# ---------------------------------------------------------------- jobs / fit (identical to luppi_r4e_ext.run_job, PPC branch)
def job_list():
    J = []
    for i, sub in enumerate(subjects()):
        if sub in DEMOSET:
            continue
        for mdl in ("signed", "coop"):
            J.append(dict(exp="B1", cond="accS", tag=str(sub), subject_index=i, session="REST1", seed=11, model=mdl))
    for j in J:
        j["key"] = "|".join(f"{k}={j[k]}" for k in sorted(j)); j["file"] = OUT + "jobs/" + hashlib.sha1(j["key"].encode()).hexdigest()[:16] + ".json"
    return J


def job_files(j):
    return [OUT + "e5/SC.npy", OUT + "e5/freq_REST1.npy", OUT + f"e5/{j['tag']}_REST1_input.npy", OUT + f"e5/{j['tag']}_REST1_heldout.npy"]


_SIM = {}


def run_job(j, stop_at):
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if "sim" not in _SIM:
        _SIM["sim"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["sim"]; t0 = time.time(); cfg = dict(L.HEADER); fs = job_files(j)
    SC = np.load(fs[0]); f = np.load(fs[1]); inp = np.load(fs[2]); hop = L.prep_emp(np.load(fs[3])); base = 9_000_000 + 1000 * j["subject_index"]
    C, info = L.fit(inp, SC, f, cfg, seed=j["seed"], cooperative_only=(j["model"] == "coop"), lag="matlab", sim=sim, stop_at=stop_at)
    sims = []
    for r in range(8):
        L.check_deadline(stop_at, f"before P2 simulation {r}"); sims.append(np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))))
        L.check_deadline(stop_at, f"after P2 simulation {r}")
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    mse = Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm)
    if not (C.shape == (100, 100) and np.all(np.isfinite(C)) and np.isfinite(mse)):
        raise ValueError("non-finite or mis-shaped fitted matrix or held-out MSE")
    np.save(j["file"].replace(".json", "_C.npy"), C)
    return dict(j, secs=round(time.time() - t0, 1), stop=info["stop"], iterations=info["iterations"], fc_corr=info["fc_corr"], mse_ho=mse,
                neg=L.neg_stats(C, SC), C_sha=sha_arr(C))


def stage_freeze():
    if os.path.exists(FREEZE):
        raise SystemExit("frozen once")
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha("b1_analyze.py"):
        raise SystemExit("analysis pre-registration missing or not matching")
    mf = json.load(open(MAN)); man = {OUT + r["file"]: r["sha"] for r in mf["files"]}; J = job_list(); files = {}
    if mf["series_manifest"] != sha(SERIES_MAN):
        raise SystemExit("series manifest changed")
    verify_series()
    bad = [p for p, h in man.items() if sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} prepare outputs changed, e.g. {bad[:3]}")
    for j in J:
        for p in job_files(j):
            if p not in man or sha(p) != man[p]:
                raise SystemExit(f"{p} missing from the manifest or changed")
            files[p] = man[p]
    fr = dict(group="b1", n_jobs=len(J), created=time.strftime("%Y-%m-%d %H:%M:%S"), host=os.environ.get("COMPUTERNAME"), code={c: sha(c) for c in CODE},
              raw={p: sha(p) for p in (SC_SRC, SUBJ_JSON, WP + "wmparc2_etag_verify.json", TEMPLATE)}, series_manifest=sha(SERIES_MAN), manifest=sha(MAN), prereg=sha(PREREG), files=files)
    json.dump(fr, open(FREEZE, "x"), indent=1); print("frozen b1", sha(FREEZE), "jobs", len(J), flush=True)


def verify_freeze():
    fr = json.load(open(FREEZE))
    checks = list(fr["code"].items()) + list(fr["raw"].items()) + [(SERIES_MAN, fr["series_manifest"]), (MAN, fr["manifest"]), (PREREG, fr["prereg"])] + list(fr["files"].items())
    bad = [p for p, h in checks if not os.path.exists(p) or sha(p) != h]
    if bad:
        raise SystemExit(f"{len(bad)} frozen items changed, e.g. {bad[:3]}")
    return sha(FREEZE), fr


def job_prov(j, fsha, fr):
    return dict(freeze=fsha, group="b1", inputs={p: fr["files"][p] for p in job_files(j)})


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
    dl_file = OUT + "b1_deadline.json"
    if not os.path.exists(dl_file):
        json.dump(dict(start=time.time(), deadline=time.time() + DEADLINE_H * 3600, freeze=fsha), open(dl_file, "w"))
    dl = json.load(open(dl_file))
    if dl.get("freeze") != fsha or dl["deadline"] > dl["start"] + DEADLINE_H * 3600 + 1:
        raise SystemExit("deadline record does not belong to this freeze")
    J = job_list()
    if len(J) != fr["n_jobs"]:
        raise SystemExit("job count differs from the freeze")
    print("jobs", len(J), flush=True)
    with mp.Pool(procs) as p:
        states = p.map(run_job_safe, [(j, job_prov(j, fsha, fr), dl["deadline"]) for j in J], chunksize=1)
    print({s: states.count(s) for s in set(states)}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=56); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"series": lambda: stage_series(min(a.procs, 16)), "prepare": stage_prepare, "freeze": stage_freeze, "run": lambda: stage_run(a.procs)}[a.stage]()
