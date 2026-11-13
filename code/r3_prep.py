"""232-region time series (Section 2.4.2): the frozen PPC denoising with a label-generic regional averaging step for the
Schaefer-200 + Tian S2 atlas.
Usage: python r3_prep.py gts | run [--procs 12]"""
import sys, os, json, time, hashlib, argparse, shutil, multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; WORK = DATA_ROOT + "work/"; ATL = DATA_ROOT + "raw/p33_atlas232/"; QSDR = DATA_ROOT + "raw/p33_qsdr/"
A100 = ATL + "mni_schaefer100.nii.gz"; A232 = ATL + "mni_schaefer232.nii.gz"; CONDS = ("mot", "null10", "acc", "gsr"); OUT = P33 + "r23/"
GTS_SUBJ = ("101309", "101915"); DROP = 11


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def generic_prepare(X, lab, bm, atl):
    """luppi_pipeline_vol.prepare with the ROI loop over all labels 1..max(atl) instead of the hard-coded 1..100."""
    import luppi_pipeline_vol as P
    T = X.shape[3]; Xf = X.reshape(-1, T); a = atl.ravel(); n = int(a.max())
    Y = np.stack([Xf[a == p].mean(0) for p in range(1, n + 1)]).astype(float); gs = Xf[bm.ravel()].mean(0).astype(float)
    pools = {e: (Xf[P.erode(np.isin(lab, P.WM_LABELS) & bm, e).ravel()].T.astype(float), Xf[P.erode(np.isin(lab, P.CSF_LABELS) & bm, e).ravel()].T.astype(float)) for e in (0, 1)}
    return Y, gs, pools


_CFG = {}


def _init(atlas, work):
    _CFG.update(atlas=atlas, work=work)


def job(args):
    """Frozen PPC job with the atlas, the ROI step and the output root redirected (module attributes set per process)."""
    import luppi_ppc_prep as PP, luppi_pipeline_vol as P
    P.ATLAS = _CFG["atlas"]; P.prepare = generic_prepare; PP.WORK = _CFG["work"]
    return PP.job(args)


def e5(R):
    import luppi_r4e_lib as Lib
    return Lib.bandpass(R.T.astype(float)[:, DROP:])


def stage_gts():
    import nibabel as nib
    import luppi_ppc_prep as PP
    tmp = OUT + "gts_tmp/"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp + "frozen/"); os.makedirs(tmp + "ad/")
    res = dict(equivalence={}, atlas={}); pairs = [(s, d) for s in GTS_SUBJ for d in ("LR", "RL")]
    PP.WORK = tmp + "frozen/"                                    # (b) unmodified frozen job first, before any attribute is patched
    for s, d in pairs:
        r = PP.job((s, d))
        if not r.get("ok"):
            raise SystemExit(f"frozen job failed {s} {d}: {r.get('error')}")
    _init(A100, tmp + "ad/")                                    # (a) adapter, same process and environment
    for s, d in pairs:
        r = job((s, d))
        if not r.get("ok"):
            raise SystemExit(f"adapter job failed {s} {d}: {r.get('error')}")
    rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    for s, d in pairs:
        for c in CONDS:
            f = f"p33_ts_{c}/{s}_REST1_{d}_s100.npy"; ad = np.load(tmp + "ad/" + f); fr = np.load(tmp + "frozen/" + f); old = np.load(WORK + f)
            kind = "input" if d == "LR" else "heldout"; e5_old = np.load(P33 + f"r4e_ppc_{c}/e5/{s}_REST1_{kind}.npy")
            res["equivalence"][f"{s}_{d}_{c}"] = dict(adapter_vs_frozen_raw=float(np.max(np.abs(ad - fr))), adapter_vs_frozen_e5=float(np.max(np.abs(e5(ad) - e5(fr)))),
                                                      frozen_vs_archive_rel_raw=rel(fr, old), frozen_vs_archive_rel_e5=rel(e5(fr), e5_old),
                                                      frozen_vs_archive_abs_raw=float(np.max(np.abs(fr - old))))
    shutil.rmtree(tmp, ignore_errors=True)
    a = nib.load(A232); f = nib.load(f"{DATA_ROOT}raw/p33_hcp_vol/{GTS_SUBJ[0]}_rfMRI_REST1_LR.nii.gz"); L = np.asarray(a.dataobj).astype(int)
    res["atlas"] = dict(grid_equal=bool(a.shape[:3] == f.shape[:3]), affine_equal=bool(np.allclose(a.affine, f.affine, atol=1e-3)),
                        labels_complete=bool(sorted(np.unique(L[L > 0]).tolist()) == list(range(1, 233))))
    V = res["equivalence"].values(); ma = max(max(v["adapter_vs_frozen_raw"], v["adapter_vs_frozen_e5"]) for v in V)
    mr = max(max(v["frozen_vs_archive_rel_raw"], v["frozen_vs_archive_rel_e5"]) for v in V)
    res.update(spec="time-series equivalence check", max_adapter_vs_frozen=ma, max_rel_frozen_vs_archive=mr, gate=bool(ma < 1e-12 and mr < 1e-9 and all(res["atlas"].values())))
    json.dump(res, open(OUT + "r3_gts.json", "w"), indent=1)
    print("G-TS | series", len(res["equivalence"]), "| adapter vs frozen", ma, "| rel frozen vs archive", mr, "| atlas", res["atlas"], "| gate", res["gate"], flush=True)


def stage_run(procs):
    if not json.load(open(OUT + "r3_gts.json"))["gate"]:
        raise SystemExit("G-TS gate not passed")
    subs = [x["subject"] for x in json.load(open(P33 + "p33_hcp_s200_v13.json"))["subjects"]]; have = {f.split(".")[0] for f in os.listdir(QSDR) if f.endswith(".qsdr.fz")}
    subs = [s for s in subs if s in have]; work = WORK + "r3tmp/"; os.makedirs(work, exist_ok=True)
    with mp.Pool(procs, initializer=_init, initargs=(A232, work)) as p:
        res = p.map(job, [(s, d) for s in subs for d in ("LR", "RL")], chunksize=1)
    bad = [(r["subject"], r["run"], r.get("error")) for r in res if not r.get("ok")]
    if bad:
        raise SystemExit(f"failed runs: {bad[:5]}")
    import luppi_hopf as L
    files = {}
    for c in CONDS:
        tdir = WORK + f"p33_ts232_{c}/"; idir = OUT + f"r3inp/{c}/"; os.makedirs(tdir, exist_ok=True); os.makedirs(idir, exist_ok=True); lr = []
        for s in subs:
            for d, kind in (("LR", "input"), ("RL", "heldout")):
                R = np.load(work + f"p33_ts_{c}/{s}_REST1_{d}_s100.npy")
                if R.shape != (1200, 232) or not np.all(np.isfinite(R)) or R.std(0).min() <= 0:
                    raise SystemExit(f"bad series {s} {d} {c} {R.shape}")
                np.save(tdir + f"{s}_REST1_{d}.npy", R); y = e5(R); p = idir + f"{s}_REST1_{kind}.npy"; np.save(p, y); files[p] = sha(p)
                if kind == "input":
                    lr.append(y)
        f = L.regional_frequencies(lr); p = idir + "freq_REST1.npy"; np.save(p, f); files[p] = sha(p)
    shutil.rmtree(work, ignore_errors=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), subjects=subs, conditions=CONDS, atlas_sha=sha(A232), files=files,
                   code={c: sha(c) for c in ("r3_prep.py", "luppi_ppc_prep.py", "luppi_pipeline_vol.py")}), open(OUT + "r3_prep_manifest.json", "w"), indent=1)
    print("R3 prep | subjects", len(subs), "| files", len(files), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=12); a = ap.parse_args()
    stage_gts() if a.stage == "gts" else stage_run(a.procs)
