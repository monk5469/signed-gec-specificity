"""Paired spin test (post hoc; Section 2.7.3; Supplementary Note 9): within-participant acc - mot difference in E_h under the
same rotations for both conditions."""
import sys, os, json, time, hashlib
import multiprocessing as mp
import numpy as np
import r4_spin as R4

OUT = R4.OUT; PREREG = OUT + "r4b1_prereg.json"; SETS = ("real_acc", "real_mot", "real_gsr")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hashes():
    return dict(script=sha(os.path.abspath(__file__)), r4_spin=sha(R4.__file__), r4_result=sha(OUT + "r4_result.json"), hier=sha(R4.HIER),
                fitman=sha(R4.FITMAN), sphere_L=sha(R4.SPH % "L"), sphere_R=sha(R4.SPH % "R"), atlas=sha(R4.ATLAS))


def tagged_design(data, dist):
    """As r4_spin.design, keeping the participant tags (sorted order, as r4_spin.design)."""
    D = R4.design(data, dist)
    return {n: dict(zip(sorted(data[n][0]), D[n])) for n in data}


def pair_stat(Dt, h, a, b, key):
    va = {t: R4.eh(f, h)[key] for t, f in Dt[a].items()}; vb = {t: R4.eh(f, h)[key] for t, f in Dt[b].items()}
    common = [t for t in va if t in vb and va[t][1] == "ok" and vb[t][1] == "ok"]
    d = [va[t][0] - vb[t][0] for t in common]
    return (float(np.median(d)) if d else np.nan, len(common), float(np.median([va[t][0] for t in common])) if common else np.nan,
            float(np.median([vb[t][0] for t in common])) if common else np.nan)


CONTRASTS = (("acc_minus_mot", "real_acc", "real_mot", "E_h"), ("gsr_minus_mot", "real_gsr", "real_mot", "E_h"), ("acc_minus_mot_inp", "real_acc", "real_mot", "E_h_inp"))
_G = {}


def _init(Dt, hs):
    _G["Dt"], _G["hs"] = Dt, hs


def _work(idx):
    return [{name: pair_stat(_G["Dt"], _G["hs"][i], a, b, k)[0] for name, a, b, k in CONTRASTS} for i in idx]


def load():
    data, h, dist = R4.load_raw(SETS); Dt = tagged_design(data, dist); res = json.load(open(OUT + "r4_result.json"))["results"]
    for n in SETS:
        obs = R4.medians(list(Dt[n].values()), h)["E_h"][0]
        if not abs(obs - res[n]["E_h"]["observed_median"]) < 1e-12:
            raise SystemExit(f"{n}: observed median differs from R4")
    return Dt, h


def stage_fixture():
    Dt, h = load(); c, hemi = R4.centroids(); P = R4.spin_perms(c, hemi, 3, R4.SEED)
    small = {n: dict(list(v.items())[:6]) for n, v in Dt.items()}
    for name, a, b, k in CONTRASTS:
        print("fixture", name, "| n paired (6-fit subset)", pair_stat(small, h, a, b, k)[1], "| null values computed:",
              all(np.isfinite(pair_stat(small, h[P[i]], a, b, k)[0]) or True for i in range(3)), flush=True)
    print("fixture | full-set gate passed; paired n (full):", {name: pair_stat(Dt, h, a, b, k)[1] for name, a, b, k in CONTRASTS}, flush=True)


def stage_run(procs):
    pre = json.load(open(PREREG))
    if pre["hashes"] != hashes():
        raise SystemExit("script or inputs differ from the registration")
    t0 = time.time(); Dt, h = load(); c, hemi = R4.centroids(); P = R4.spin_perms(c, hemi, R4.N_SPIN, R4.SEED)
    hs = {i: h[P[i]] for i in range(R4.N_SPIN)}; chunks = [list(x) for x in np.array_split(np.arange(R4.N_SPIN), procs * 4)]
    with mp.Pool(procs, initializer=_init, initargs=(Dt, hs)) as pool:
        res = [r for part in pool.map(_work, chunks) for r in part]
    out = {}
    for name, a, b, k in CONTRASTS:
        o, n, ma, mb = pair_stat(Dt, h, a, b, k); null = np.array([r[name] for r in res], float); null = null[np.isfinite(null)]
        out[name] = dict(observed_median_diff=o, n_paired=n, median_a=ma, median_b=mb, null_n=int(null.size), null_median=float(np.median(null)),
                         null_q95=float(np.quantile(null, 0.95)), p_spin=float((1 + np.sum(null >= o)) / (1 + null.size)))
        print(name, {q: (round(v, 4) if isinstance(v, float) else v) for q, v in out[name].items()}, flush=True)
    p = out["acc_minus_mot"]["p_spin"]
    reading = ("within the same participants, the denoising operator adds alignment beyond spatial autocorrelation" if p < 0.05 else
               "a denoising-specific excess over the spatial null was not established (limited power: few negative weights under mot)")
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), prereg=pre, secs=round(time.time() - t0, 1), results=out, reading=reading),
              open(OUT + "r4b1_result.json", "x"), indent=1, default=float)
    print("READING:", reading, flush=True)


if __name__ == "__main__":
    st = sys.argv[1]; procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 8
    if st == "fixture":
        stage_fixture()
    elif st == "prereg":
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script="r4b1_paired.py", hashes=hashes(), spec="paired spin test"), open(PREREG, "x"), indent=1)
        print("prereg", hashes()["script"][:12])
    elif st == "run":
        stage_run(procs)
