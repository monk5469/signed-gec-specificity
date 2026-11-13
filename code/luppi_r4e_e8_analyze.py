"""Shared analysis helpers: stratum-equal-weight means and stratified t intervals (Section 2.9), used by the simulation analyses."""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np
from scipy import stats

OUT = DATA_ROOT + "results/p33/r4e_e8/"; R4V2 = DATA_ROOT + "results/p33/r4v2/"
PREREG = OUT + "r4e_e8_analysis_prereg.json"


def strat_t(vals, alpha=0.05):
    S = {}
    for h, v in vals:
        if np.isfinite(v):
            S.setdefault(h, []).append(v)
    hs = sorted(S); w = 1.0 / 3
    if len(hs) < 3 or any(len(S[h]) < 2 for h in hs):
        return dict(theta=np.nan, lo=np.nan, hi=np.nan, degenerate=True, n={h: len(S.get(h, [])) for h in range(3)})
    m = {h: np.mean(S[h]) for h in hs}; s2 = {h: np.var(S[h], ddof=1) for h in hs}; V = sum(w ** 2 * s2[h] / len(S[h]) for h in hs)
    th = float(sum(w * m[h] for h in hs))
    if V == 0:
        return dict(theta=th, lo=np.nan, hi=np.nan, degenerate=True, n={h: len(S[h]) for h in hs})
    nu = V ** 2 / sum((w ** 2 * s2[h] / len(S[h])) ** 2 / (len(S[h]) - 1) for h in hs); t = stats.t.ppf(1 - alpha / 2, nu)
    return dict(theta=th, lo=th - t * np.sqrt(V), hi=th + t * np.sqrt(V), degenerate=False, n={h: len(S[h]) for h in hs})


def logor(y, X, j=1, ridge=1e-6, iters=100):
    x = X[:, j]
    if x.min() == x.max() or min(((x > 0) & y).sum(), ((x > 0) & ~y).sum(), ((x == 0) & y).sum(), ((x == 0) & ~y).sum()) == 0:
        return np.nan
    beta = np.zeros(X.shape[1]); yf = y.astype(float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))); Wd = p * (1 - p) + 1e-12
        H = X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1]); step = np.linalg.solve(H, X.T @ (yf - p) - ridge * beta); beta += step
        if np.max(np.abs(step)) < 1e-8:
            return float(beta[j]) if np.all(np.isfinite(beta)) else np.nan
    return np.nan                                          # not converged -> unusable


def design(SC, m, pred):
    sq = np.searchsorted(np.quantile(SC[m], [0.25, 0.5, 0.75]), SC[m])
    return np.column_stack([np.ones(m.sum()), pred.astype(float)] + [(sq == k).astype(float) for k in (1, 2, 3)])


def clustering(A):
    A = ((A + A.T) > 0).astype(float); np.fill_diagonal(A, 0); d = A.sum(1); tri = np.diag(A @ A @ A) / 2
    c = np.where(d > 1, 2 * tri / np.maximum(d * (d - 1), 1), np.nan); return float(np.nanmean(c)) if np.isfinite(c).any() else np.nan


def descriptors(C, SC, inp, nets, G=None, C_true=None):
    import luppi_hopf as L, luppi_r4e_lib as Lib
    m = Lib.support(SC); y = C[m] < 0
    net = np.asarray(nets); between = (net[:, None] != net[None, :])[m]
    F = np.corrcoef(L.prep_emp(inp)); N = len(F); Hc = np.eye(N) - np.ones((N, N)) / N; w, V = np.linalg.eigh(Hc @ F @ Hc); v = V[:, -1]
    opp = (np.sign(v)[:, None] * np.sign(v)[None, :] < 0)[m]
    neg = np.where(m & (C < 0), 1, 0); pos = np.where(m & (C > 0), 1, 0); wn = np.abs(C[m][y]); wp = C[m][~y & (C[m] > 0)]
    out = dict(fp=float(y.mean()), E_bn=logor(y, design(SC, m, between)), E_ctr=logor(y, design(SC, m, opp)),
               CR=(clustering(neg) / clustering(pos)) if clustering(pos) else np.nan,
               WR=float(np.median(wn) / np.median(wp)) if wn.size and wp.size else np.nan)
    if G is not None and C_true is not None:
        pp = ((G @ G.T) < 0)[m]; t = np.abs(C_true[m]); tq = np.searchsorted(np.quantile(t, [0.25, 0.5, 0.75]), t)
        X = np.column_stack([design(SC, m, pp)] + [(tq == k).astype(float) for k in (1, 2, 3)]); out["E_pp"] = logor(y, X)
    return out


def load():
    import luppi_r4e_e8 as E8
    fsha, fr = E8.verify_freeze(); expected = {j["key"]: j for j in E8.job_list()}
    R, seen, rej = [], set(), dict(not_done=0, wrong_prov=0, unexpected=0, duplicate=0, nonfinite=0)
    for p in sorted(glob.glob(OUT + "jobs/*.json")):
        r = json.load(open(p)); j = expected.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != E8.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if not (np.isfinite(r.get("mse_ho", np.nan)) and np.isfinite(r.get("metrics", {}).get("frac_neg", np.nan))):
            rej["nonfinite"] += 1; continue
        if r["key"] in seen:
            rej["duplicate"] += 1; continue
        seen.add(r["key"]); R.append(r)
    return R, rej, fsha, fr


def main():
    pre = json.load(open(PREREG))
    if hashlib.sha256(open(__file__, "rb").read()).hexdigest() != pre["sha256"]:
        raise SystemExit("E8 analysis script differs from the pre-registered version")
    import scipy.io as sio
    import luppi_r4e_e8 as E8, luppi_r4e_lib as Lib, luppi_r4e_run as RUN
    R, rej, fsha, fr = load(); nets = json.load(open(OUT + "networks.json"))["networks"]
    pairs = {}
    for r in R:
        pairs.setdefault((r["cell"], r["tag"]), {})[r["model"]] = r
    rows = {}
    for (cell, tag), d in pairs.items():
        if "signed" not in d or "coop" not in d:
            continue
        C = np.load(d["signed"]["file"].replace(".json", "_C.npy"))
        if RUN.sha_arr(C) != d["signed"]["C_sha"]:
            raise SystemExit(f"C_sha mismatch {d['signed']['key']}")
        s = int(tag.split("_")[1]); C_true, _, k = E8.truth(s); _, SC, _ = Lib.demo_subject(k); M = E8.CELLS[cell][0]
        inp = Lib.bandpass(np.load(OUT + f"truth/{tag}_{cell}_target.npy"))
        dsc = descriptors(C, SC, inp, nets, E8.G_of(k, M if M else 1), C_true)
        rows.setdefault(cell, []).append(dict(tag=tag, stratum=k, thr5=d["signed"]["metrics"]["frac_neg_thr0.05"], p2=d["signed"]["mse_ho"] - d["coop"]["mse_ho"], **dsc))
    out = {}
    for cell in E8.CELLS:
        rr = rows.get(cell, []); n_by = {h: sum(1 for r in rr if r["stratum"] == h) for h in range(3)}; complete = all(n_by[h] >= 8 for h in range(3))
        rec = dict(n_by_stratum=n_by, complete=complete, **{k: strat_t([(r["stratum"], r[k]) for r in rr]) for k in ("fp", "thr5", "p2", "E_bn", "E_ctr", "CR", "WR", "E_pp")},
                   p2_negative_share=float(np.mean([r["p2"] < 0 for r in rr])) if rr else np.nan)
        M = E8.CELLS[cell][0]
        if M:
            base = {r["tag"]: r for r in rows.get("N" + cell[1:], [])}
            rec["d_E_pp_vs_N"] = strat_t([(r["stratum"], r["E_pp"] - base[r["tag"]]["E_pp"]) for r in rr if r["tag"] in base])
            rec["d_E_bn_vs_N"] = strat_t([(r["stratum"], r["E_bn"] - base[r["tag"]]["E_bn"]) for r in rr if r["tag"] in base])
            usable = all(rec["E_bn"]["n"].get(h, 0) >= 8 for h in range(3))
            if not (complete and usable):
                rec["label"] = "not analysable"
            else:
                c = dict(fp_ge_020=bool(rec["fp"]["theta"] >= 0.20), signed_better_heldout=bool(np.isfinite(rec["p2"]["hi"]) and rec["p2"]["hi"] < 0),
                         between_network_enriched=bool(np.isfinite(rec["E_bn"]["lo"]) and rec["E_bn"]["lo"] > 0))
                rec["criteria"] = c
                rec["label"] = "Luppi-type evidence combination reproduced (descriptive)" if all(c.values()) else "not reproduced; satisfied: " + (",".join(k for k, v in c.items() if v) or "none")
        out[cell] = rec
    dose = [c for c in ("N1", "N2", "N4", "N8") if out[c]["complete"] and np.isfinite(out[c]["fp"]["hi"]) and out[c]["fp"]["hi"] <= 0.10]
    # real-data reference (pinned reference fits of the example participants, R4V2)
    ts = sio.loadmat(DATA_ROOT + "raw/luppi_demo/gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel(); scs = sio.loadmat(DATA_ROOT + "raw/luppi_demo/gec_3/SC.mat")["DTI_fibers_VolNorm_HCP"].ravel()
    real = []
    for k in range(3):
        for sd in (11, 12, 13):
            p = R4V2 + f"C_s{k}_header_signed_matlab_seed{sd}.npy"
            if hashlib.sha256(open(p, "rb").read()).hexdigest() != fr["raw"][p]:
                raise SystemExit(f"{p} differs from its frozen hash")
            real.append(dict(subject=k, seed=sd, **descriptors(np.load(p), scs[k].astype(float), ts[k].astype(float), nets)))
    real_summary = {key: float(np.nanmedian([r[key] for r in real])) for key in ("fp", "E_bn", "E_ctr", "CR", "WR")}
    res = dict(prereg=pre, freeze=fsha, admitted=len(R), rejected=rej, cells=out,
               dose_response=dict(smallest_tested_snr_with_fp_upper_le_010=E8.SNR[dose[0][1:]] if dose else None),
               real_demo_reference=dict(rows=real, median=real_summary), note="pointwise unadjusted intervals; descriptive labels")
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
    f3 = lambda d: "%.3f [%.3f, %.3f]" % (d["theta"], d["lo"], d["hi"])
    for c, r in out.items():
        print(c, "FP", f3(r["fp"]), "| thr5", f3(r["thr5"]), "| P2", f3(r["p2"]), "neg-share %.2f" % r["p2_negative_share"], "| E_bn", f3(r["E_bn"]),
              "| CR", f3(r["CR"]), "| WR", f3(r["WR"]), "| E_pp", f3(r["E_pp"]), "|", r.get("label", ""), flush=True)
    print("dose-response:", res["dose_response"], "| real demo reference (median):", {k: round(v, 3) for k, v in real_summary.items()}, flush=True)


if __name__ == "__main__":
    main()
