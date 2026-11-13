"""Spin test (exploratory; Sections 2.7.3, 3.6; Supplementary Note 9): 10,000 rotations of the parcel centroids and the rotation
null of the hierarchy-gap coefficient E_h; the rotations are reused for O_h by vavf_vd.py."""
import sys, os, json, time, hashlib
import multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; OUT = P33 + "r4/"; PREREG = OUT + "r4_prereg.json"
HIER = P33 + "e13_hier.json"; E13RES = P33 + "e13_analysis.json"; FITMAN = P33 + "e13_fit_manifest.json"
SPH = DATA_ROOT + "raw/p33_sphere/100307.%s.sphere.32k_fs_LR.surf.gii"
ATLAS = DATA_ROOT + "raw/p33_hcp/Schaefer2018_100Parcels_7Networks_order.dlabel.nii"
SETS = ("real_acc", "real_gsr", "real_mot", "e10c_COOP_d136", "e11_COOP_N49d136", "e10c_CMPN_d136", "e11_CMPN_N49d136", "e11_CMPH_N49d136")
N_SPIN = 10_000; SEED = 20261016; KEYS = ("E_h", "E_h_inp")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hashes():
    return dict(script=sha(os.path.abspath(__file__)), hier=sha(HIER), fitman=sha(FITMAN), e13_result=sha(E13RES),
                sphere_L=sha(SPH % "L"), sphere_R=sha(SPH % "R"), atlas=sha(ATLAS))


def centroids():
    """(100 x 3) unit-sphere parcel centroids and hemisphere labels (0 = left, 1 = right)."""
    import nibabel as nib
    lab = nib.load(ATLAS); Lv = np.asarray(lab.get_fdata())[0].astype(int); ax = lab.header.get_axis(1)
    sph = {"CIFTI_STRUCTURE_CORTEX_LEFT": nib.load(SPH % "L").darrays[0].data, "CIFTI_STRUCTURE_CORTEX_RIGHT": nib.load(SPH % "R").darrays[0].data}
    acc = {p: [] for p in range(1, 101)}; hemi = np.full(100, -1)
    for name, sl, bm in ax.iter_structures():
        if name in sph:
            for k, v in enumerate(bm.vertex):
                p = Lv[sl.start + k]
                if p > 0:
                    acc[p].append(sph[name][int(v)]); hemi[p - 1] = 0 if "LEFT" in name else 1
    if any(len(v) == 0 for v in acc.values()) or np.any(hemi < 0):
        raise SystemExit("a parcel has no surface vertices")
    c = np.array([np.mean(acc[p], 0) for p in range(1, 101)]); c /= np.linalg.norm(c, axis=1, keepdims=True)
    return c, hemi


def spin_perms(c, hemi, n, seed):
    from scipy.spatial.transform import Rotation
    from scipy.optimize import linear_sum_assignment
    Rs = Rotation.random(n, random_state=seed).as_matrix(); F = np.diag([-1.0, 1.0, 1.0]); out = np.zeros((n, 100), int)
    idx = [np.where(hemi == 0)[0], np.where(hemi == 1)[0]]
    for s in range(n):
        for h, R in ((0, Rs[s]), (1, F @ Rs[s] @ F)):
            ii = idx[h]; rot = c[ii] @ R.T; cost = -rot @ c[ii].T       # maximise cosine = minimise angular distance
            r, col = linear_sum_assignment(cost); out[s, ii[r]] = ii[col]
    return out


def load_raw(names):
    """hierarchy analysis admission, capturing the admitted fits' raw inputs instead of computing metrics."""
    import e13_analyze as E
    hier = json.load(open(HIER)); fitman = json.load(open(FITMAN)); dist = np.array(hier["dist_mm"]); orig = E.metrics; data = {}
    E.metrics = lambda C, SC, inp, h, d: dict(C=C, SC=SC, inp=inp)
    try:
        for n in names:
            rows, den = E.load_set(n, {"raw": np.array(hier["h_A2"])}, dist, fitman); data[n] = (rows, den)
    finally:
        E.metrics = orig
    return data, np.array(hier["h_A2"]), dist


def design(data, dist):
    """Per fit: fixed pieces of the hierarchy analysis regressions (support mask, outcomes, covariate columns)."""
    import e13_analyze as E, luppi_hopf as L, luppi_r4e_lib as Lib
    D = {}
    for n, (rows, _) in data.items():
        D[n] = []
        for tag in sorted(rows):
            C, SC, inp = rows[tag]["raw"]["C"], rows[tag]["raw"]["SC"], rows[tag]["raw"]["inp"]; m = Lib.support(SC)
            f = np.corrcoef(L.prep_emp(inp))[m]; cov = np.column_stack(E.quart(SC[m]) + E.quart(dist[m]))
            D[n].append(dict(m=m, y=C[m] < 0, yi=f < 0, cov=cov))
    return D


def eh(fit, h):
    import e13_analyze as E
    from scipy.stats import rankdata
    r = rankdata(h); d = (np.abs(r[:, None] - r[None, :]) / 99.0)[fit["m"]]; z = (d - d.mean()) / d.std()
    X = np.column_stack([np.ones(len(z)), z, fit["cov"]])
    return {k: E.irls(fit[y], X) for k, y in (("E_h", "y"), ("E_h_inp", "yi"))}


def medians(fits, h):
    out = {}
    for k in KEYS:
        v = [x for x, st in (eh(f, h)[k] for f in fits) if st == "ok"]
        out[k] = (float(np.median(v)) if v else np.nan, len(v))
    return out


_G = {}


def _init(D, h):
    _G["D"], _G["h"] = D, h


def _work(perms):
    D, h = _G["D"], _G["h"]
    return [{n: {k: v[0] for k, v in medians(fits, h[p]).items()} for n, fits in D.items()} for p in perms]


def observed_gate(data, D, h):
    e13 = json.load(open(E13RES))["summary"]; obs = {}
    for n, fits in D.items():
        obs[n] = medians(fits, h); ref = e13[n]["A2"]["E_h"].get("median", np.nan)
        if not abs(obs[n]["E_h"][0] - ref) < 1e-12:
            raise SystemExit(f"{n}: observed median E_h {obs[n]['E_h'][0]} != E13 {ref}")
    return obs


def stage_fixture():
    c, hemi = centroids(); P = spin_perms(c, hemi, 5, SEED)
    ok = all(sorted(p) == list(range(100)) for p in P) and all(np.all(hemi[P[s]] == hemi) for s in range(5))
    data, h, dist = load_raw(("real_acc", "e11_COOP_N49d136")); D = {n: v[:3] for n, v in design(data, dist).items()}
    for n in D:                                                   # gate on the full sets is in run; here: per-fit values vs hierarchy analysis metrics
        import e13_analyze as E
        rows = data[n][0]; t = sorted(rows)[:3]
        for tag, fit in zip(t, D[n]):
            ref = E.metrics(rows[tag]["raw"]["C"], rows[tag]["raw"]["SC"], rows[tag]["raw"]["inp"], h, dist)["val"]["E_h"]
            assert abs(eh(fit, h)["E_h"][0] - ref) < 1e-12 or (np.isnan(ref) and np.isnan(eh(fit, h)["E_h"][0]))
    print("fixture | permutations valid and hemisphere-preserving:", ok, "| fixed points per spin:", [int((P[s] == np.arange(100)).sum()) for s in range(5)],
          "| per-fit E_h equals E13 metrics on 2 sets x 3 fits: True | admitted", {n: data[n][1]["admitted"] for n in data}, flush=True)


def stage_run(procs):
    pre = json.load(open(PREREG))
    if pre["hashes"] != hashes():
        raise SystemExit("script or inputs differ from the registration")
    t0 = time.time(); data, h, dist = load_raw(SETS); D = design(data, dist); obs = observed_gate(data, D, h)
    c, hemi = centroids(); P = spin_perms(c, hemi, N_SPIN, SEED)
    hs = {i: h[P[i]] for i in range(N_SPIN)}; chunks = np.array_split(np.arange(N_SPIN), procs * 4)
    with mp.Pool(procs, initializer=_init, initargs=(D, hs)) as pool:
        res = [r for part in pool.map(_work, [list(ch) for ch in chunks]) for r in part]
    out = {}
    for n in SETS:
        out[n] = dict(admitted=data[n][1]["admitted"])
        for k in KEYS:
            null = np.array([r[n][k] for r in res], float); null = null[np.isfinite(null)]; o = obs[n][k][0]
            out[n][k] = dict(observed_median=o, n_fits_ok=obs[n][k][1], null_n=int(null.size), null_median=float(np.median(null)),
                             null_q95=float(np.quantile(null, 0.95)), p_spin=float((1 + np.sum(null >= o)) / (1 + null.size)))
        print(n, {k: (round(out[n][k]["observed_median"], 3), round(out[n][k]["null_median"], 3), round(out[n][k]["null_q95"], 3), round(out[n][k]["p_spin"], 4)) for k in KEYS}, flush=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), prereg=pre, n_spin=N_SPIN, secs=round(time.time() - t0, 1),
                   fixed_points_mean=float(np.mean((P == np.arange(100)).sum(1))), results=out), open(OUT + "r4_result.json", "x"), indent=1, default=float)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); st = sys.argv[1]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 56
    if st == "fixture":
        stage_fixture()
    elif st == "prereg":
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script="r4_spin.py", hashes=hashes(), spec="spin test"), open(PREREG, "x"), indent=1)
        print("prereg", hashes()["script"][:12])
    elif st == "run":
        stage_run(procs)
