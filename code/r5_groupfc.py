"""Group-level FC correlation (descriptive; Sections 2.3.3, 3.2; Supplementary Note 3): Pearson and Spearman correlations between
participant-mean empirical and simulated FC for each model, comparable with the group-level values of Luppi et al. (2026)."""
import sys, os, json, glob, time, hashlib
import multiprocessing as mp
import numpy as np
from paths import DATA_ROOT

P33 = DATA_ROOT + "results/p33/"; OUT = P33 + "r5/"; PREREG = OUT + "r5_prereg.json"
CONDS = ("vol0", "mot", "null10", "acc1", "acc3", "acc", "accero", "gsr", "fix"); FIX_OUT = P33 + "r4e_e5fix/"
DEMO = {"100307", "100408", "101107"}; MIN_N = 86; B = 2000; SEED = 20261015; TOL = 1e-9


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_prereg():
    if not os.path.exists(PREREG) or json.load(open(PREREG))["sha256"] != sha(os.path.abspath(__file__)):
        sys.exit("r5_groupfc.py does not match the registered hash")


def admitted(cond):
    """PPC admission without the descriptor endpoints; returns X (re-pointed module) and [(tag, idx, {model: (job, rec, C)})]."""
    import luppi_r4e_ppc as W, luppi_r4e_run as RUN
    X = W.X
    if cond == "fix":
        X.OUT = FIX_OUT; X.HCP = DATA_ROOT + "work/p33_ts_fix/"
    else:
        W.point(cond)
    fsha, fr = X.verify_freeze("e5"); expected = {j["key"]: j for j in X.job_list("e5")}; R = {}; rej = {}
    for p in sorted(glob.glob(X.OUT + "jobs_e5/*.json")):
        try:
            r = json.load(open(p)); assert isinstance(r, dict)
        except Exception:
            rej["malformed"] = rej.get("malformed", 0) + 1; continue
        j = expected.get(r.get("key"))
        why = ("unexpected" if j is None or os.path.basename(j["file"]) != os.path.basename(p) else
               "field_mismatch" if any(r.get(k) != v for k, v in j.items()) else
               "not_done" if r.get("state") != "done" else
               "wrong_prov" if r.get("prov") != X.job_prov(j, fsha, fr) else
               "duplicate" if j["model"] in R.get(j["tag"], {}) else None)
        if why:
            rej[why] = rej.get(why, 0) + 1; continue
        C = np.load(j["file"].replace(".json", "_C.npy"))
        if C.shape != (100, 100) or not np.all(np.isfinite(C)) or RUN.sha_arr(C) != r.get("C_sha"):
            rej["bad_matrix"] = rej.get("bad_matrix", 0) + 1; continue
        R.setdefault(j["tag"], {})[j["model"]] = (j, r, C)
    pairs = [(t, d["signed"][0]["subject_index"], d) for t, d in sorted(R.items()) if t not in DEMO and "signed" in d and "coop" in d]
    return X, pairs, rej, fsha


_SIM = {}


def sim_job(args):
    """Regenerate the 8 held-out evaluation simulations of one fit; return mean FC and the recomputed held-out MSE."""
    import jax
    import luppi_hopf as L, luppi_r4e_lib as Lib
    C, f, base, hop_path = args
    if "sim" not in _SIM:
        _SIM["sim"] = L.make_simulator(100, Lib.T_LEN, 3000.0)
    sim = _SIM["sim"]; cfg = dict(L.HEADER)
    sims = [np.asarray(sim(C, f, cfg["a"], cfg["sigma"], jax.random.PRNGKey(base + r))) for r in range(8)]
    FCm = np.mean([L.fc(x) for x in sims], 0); LAGm = np.mean([L.lag_fc(x, 1, "matlab") for x in sims], 0)
    hop = L.prep_emp(np.load(hop_path))
    return FCm, float(Lib.heldout_mse(L.fc(hop), L.lag_fc(hop, 1, "matlab"), FCm, LAGm))


def run_cond(cond, procs, limit=None):
    import luppi_hopf as L
    X, pairs, rej, fsha = admitted(cond)
    if limit:
        pairs = pairs[:limit]
    f = np.load(X.OUT + "e5/freq_REST1.npy"); tasks = []; meta = []
    for tag, idx, d in pairs:
        base = 9_000_000 + 1000 * idx
        for mdl in ("signed", "coop"):
            tasks.append((d[mdl][2], f, base, X.OUT + f"e5/{tag}_REST1_heldout.npy")); meta.append((tag, mdl, d[mdl][1]))
    with mp.Pool(procs) as p:
        res = p.map(sim_job, tasks, chunksize=1)
    tags = [t for t, _, _ in pairs]; FCe = np.stack([np.corrcoef(L.prep_emp(np.load(X.OUT + f"e5/{t}_REST1_input.npy"))) for t in tags])
    FCs = {m: np.stack([res[i][0] for i, (_, mm, _) in enumerate(meta) if mm == m]) for m in ("signed", "coop")}
    dev = [abs(res[i][1] - rec["mse_ho"]) / abs(rec["mse_ho"]) for i, (_, _, rec) in enumerate(meta)]
    fc_corr = {m: np.array([rec["fc_corr"] for (_, mm, rec) in meta if mm == m], float) for m in ("signed", "coop")}
    return dict(tags=tags, FCe=FCe, FCs=FCs, fc_corr=fc_corr, max_rel_dev=float(max(dev)) if dev else np.nan, rej=rej, freeze=fsha)


def stage_fixture():
    r = run_cond("mot", 4, limit=2)
    print("fixture mot | pairs", len(r["tags"]), "| FC_emp", r["FCe"].shape, "| FC_sim", {m: v.shape for m, v in r["FCs"].items()},
          "| max relative deviation of recomputed held-out MSE", r["max_rel_dev"], "| gate", r["max_rel_dev"] < TOL, flush=True)


def stage_run(procs):
    check_prereg(); summary = {}
    for cond in CONDS:
        t0 = time.time(); r = run_cond(cond, procs)
        if not r["max_rel_dev"] < TOL:
            sys.exit(f"{cond}: held-out MSE not reproduced (max relative deviation {r['max_rel_dev']})")
        np.savez(OUT + f"{cond}.npz", tags=np.array(r["tags"]), FCe=r["FCe"], FCs_signed=r["FCs"]["signed"], FCs_coop=r["FCs"]["coop"],
                 fc_corr_signed=r["fc_corr"]["signed"], fc_corr_coop=r["fc_corr"]["coop"])
        summary[cond] = dict(n_pairs=len(r["tags"]), rejected=r["rej"], freeze=r["freeze"], max_rel_dev=r["max_rel_dev"],
                             npz_sha=sha(OUT + f"{cond}.npz"), secs=round(time.time() - t0, 1))
        print(cond, summary[cond], flush=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script_sha=sha(os.path.abspath(__file__)), conds=summary), open(OUT + "r5_run.json", "w"), indent=1)


def gcorr(FCe, FCs, idx):
    from scipy.stats import spearmanr
    iu = np.triu_indices(FCe.shape[1], 1); a = FCe[idx].mean(0)[iu]; b = FCs[idx].mean(0)[iu]
    return float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b)[0])


def stage_analyze():
    check_prereg(); run = json.load(open(OUT + "r5_run.json")); res = {}
    for cond in CONDS:
        if run["conds"][cond]["npz_sha"] != sha(OUT + f"{cond}.npz"):
            sys.exit(f"{cond}.npz changed after the run")
        z = np.load(OUT + f"{cond}.npz"); n = len(z["tags"]); all_ = np.arange(n)
        if n < MIN_N:
            res[cond] = dict(n=n, analysable=False); continue
        FCe, S, Co = z["FCe"], z["FCs_signed"], z["FCs_coop"]
        ps, ss = gcorr(FCe, S, all_); pc, sc = gcorr(FCe, Co, all_)
        rng = np.random.default_rng(SEED); bs = []
        for _ in range(B):
            i = rng.integers(0, n, n); a = gcorr(FCe, S, i); b = gcorr(FCe, Co, i); bs.append((a[0], b[0], a[0] - b[0]))
        bs = np.array(bs); ci = lambda k: [float(np.quantile(bs[:, k], 0.025)), float(np.quantile(bs[:, k], 0.975))]
        iu = np.triu_indices(100, 1); ind = {m: [float(np.corrcoef(FCe[k][iu], F[k][iu])[0, 1]) for k in range(n)] for m, F in (("signed", S), ("coop", Co))}
        res[cond] = dict(n=n, analysable=True,
                         group_pearson=dict(signed=ps, coop=pc, diff=ps - pc, ci_signed=ci(0), ci_coop=ci(1), ci_diff=ci(2)),
                         group_spearman=dict(signed=ss, coop=sc, diff=ss - sc),
                         participant_median_r=dict(signed=float(np.median(ind["signed"])), coop=float(np.median(ind["coop"]))),
                         stored_fc_corr_median=dict(signed=float(np.median(z["fc_corr_signed"])), coop=float(np.median(z["fc_corr_coop"]))))
        g = res[cond]["group_pearson"]
        print(f"{cond:7s} n={n} | group r signed {g['signed']:.3f} coop {g['coop']:.3f} diff {g['diff']:+.3f} {np.round(g['ci_diff'], 3).tolist()}"
              f" | participant median r {res[cond]['participant_median_r']['signed']:.3f} / {res[cond]['participant_median_r']['coop']:.3f}", flush=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), prereg=json.load(open(PREREG)), run=sha(OUT + "r5_run.json"), results=res),
              open(OUT + "r5_result.json", "x"), indent=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); st = sys.argv[1]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 56
    if st == "fixture":
        stage_fixture()
    elif st == "prereg":
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script="r5_groupfc.py", sha256=sha(os.path.abspath(__file__)),
                       spec="group-level FC correlation"), open(PREREG, "x"), indent=1); print("prereg", sha(os.path.abspath(__file__))[:12])
    elif st == "run":
        stage_run(procs)
    elif st == "analyze":
        stage_analyze()
