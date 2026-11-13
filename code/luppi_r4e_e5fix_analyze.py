"""Analysis of the surface ICA-FIX condition (descriptive; Sections 3.2, 3.3).
Usage: python luppi_r4e_e5fix_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np
from scipy import stats

import luppi_r4e_e5fix as W                     # re-points luppi_r4e_ext to the FIX paths
X = W.X
OUT = X.OUT; PREREG = OUT + "r4e_e5fix_analysis_prereg.json"
E6_JSON = DATA_ROOT + "results/p33/r4e_ext_e6_analysis.json"; E8_JSON = DATA_ROOT + "results/p33/r4e_e8_analysis.json"
E5_MIN_JOBS = DATA_ROOT + "results/p33/r4e_ext/jobs_e5/"; NETS = DATA_ROOT + "results/p33/r4e_e8/networks.json"


def load():
    fsha, fr = X.verify_freeze("e5"); expected = {j["key"]: j for j in X.job_list("e5")}
    R, seen, rej = [], set(), dict(not_done=0, wrong_prov=0, unexpected=0, duplicate=0, nonfinite=0)
    for p in sorted(glob.glob(OUT + "jobs_e5/*.json")):
        r = json.load(open(p)); j = expected.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != X.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if not np.isfinite(r.get("mse_ho", np.nan)):
            rej["nonfinite"] += 1; continue
        if r["key"] in seen:
            rej["duplicate"] += 1; continue
        seen.add(r["key"]); R.append(r)
    return R, rej, fsha


def q(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return dict(n=int(v.size), median=float(np.median(v)) if v.size else np.nan, q25=float(np.quantile(v, 0.25)) if v.size else np.nan,
                q75=float(np.quantile(v, 0.75)) if v.size else np.nan)


def main():
    pre = json.load(open(PREREG))
    if hashlib.sha256(open(__file__, "rb").read()).hexdigest() != pre["sha256"]:
        raise SystemExit("E5-FIX analysis script differs from the pre-registered version")
    import luppi_r4e_run as RUN, luppi_r4e_e6 as E6, luppi_r4e_e8_analyze as A8
    R, rej, fsha = load(); SC = np.load(OUT + "e5/SC.npy"); nets = json.load(open(NETS))["networks"]
    def C_of(r):
        C = np.load(r["file"].replace(".json", "_C.npy"))
        if RUN.sha_arr(C) != r["C_sha"]:
            raise SystemExit(f"C_sha mismatch {r['key']}")
        return C
    summ, sig, desc = {}, {}, {}
    for ses in ("REST1", "REST2"):
        sg = {r["tag"]: r for r in R if r["session"] == ses and r["model"] == "signed"}; cp = {r["tag"]: r for r in R if r["session"] == ses and r["model"] == "coop"}
        summ[ses] = dict(n_signed=len(sg), n_coop=len(cp), neg=q([r["neg"]["frac_neg_thr0.0"] for r in sg.values()]),
                         neg5=q([r["neg"]["frac_neg_thr0.05"] for r in sg.values()]), fc_corr_signed=q([r["fc_corr"] for r in sg.values()]),
                         fc_corr_coop=q([r["fc_corr"] for r in cp.values()]),
                         p2_negative_share=float(np.mean([sg[t]["mse_ho"] - cp[t]["mse_ho"] < 0 for t in sg if t in cp])) if sg else np.nan)
        sig[ses], desc[ses] = {}, {}
        for t, r in sg.items():
            C = C_of(r); inp = np.load(OUT + f"e5/{t}_{ses}_input.npy")
            sig[ses][t] = {ld: E6.signature(C, SC, inp, ld) for ld in E6.LOADINGS}
            desc[ses][t] = A8.descriptors(C, SC, inp, nets)
    common = sorted(set(sig["REST1"]) & set(sig["REST2"]))
    neg1 = {r["tag"]: r["neg"]["frac_neg_thr0.0"] for r in R if r["session"] == "REST1" and r["model"] == "signed"}
    neg2 = {r["tag"]: r["neg"]["frac_neg_thr0.0"] for r in R if r["session"] == "REST2" and r["model"] == "signed"}
    retest = float(stats.spearmanr([neg1[t] for t in common], [neg2[t] for t in common])[0]) if len(common) > 2 else np.nan
    # paired change versus minimally preprocessed PPC (descriptive; those records are frozen in the r4e_ext e5 group)
    mini = {}
    for p in glob.glob(E5_MIN_JOBS + "*.json"):
        r = json.load(open(p))
        if r.get("state") == "done" and r["model"] == "signed":
            mini[(r["tag"], r["session"])] = r["neg"]["frac_neg_thr0.0"]
    d_vs_min = {ses: q([n - mini[(t, ses)] for t, n in (neg1 if ses == "REST1" else neg2).items() if (t, ses) in mini]) for ses in ("REST1", "REST2")}
    # contrast-loading signature on FIX REST1 against the frozen reference families
    e6 = json.load(open(E6_JSON)); fam = e6["families"]
    rows = [v["contrast"] for v in sig["REST1"].values()]; ok = [r["E"] for r in rows if r["status"] == "ok"]
    analysable = len(ok) >= 0.8 * 98; med = float(np.median(ok)) if ok else np.nan
    e2m = fam["E2_tau50_q1_F"]["contrast"]["E"]["median"]; cmph25 = fam["E3_CMPH_F"]["contrast"]["E"]["q25"]; cmpm = fam["E3_CMP_F"]["contrast"]["E"]["median"]
    fam_ok = all(fam[k]["contrast"]["analysable"] for k in ("E2_tau50_q1_F", "E3_CMPH_F", "E3_CMP_F"))     #: families must be analysable too
    if not (analysable and fam_ok) or not np.isfinite(med):
        label = "not analysable"
    elif med >= e2m and med >= cmph25:
        label = "ordinal label A: at or above the hidden-drive (q 0.2) median and the aligned-competition 25th percentile"
    elif med <= cmpm:
        label = "ordinal label C: at or below the random-competition median"
    else:
        label = "ordinal label B: between"
    pct_e6 = {k: float(np.mean(np.array([r["E"] for r in v["contrast"]["rows"]], float) <= med) * 100) for k, v in fam.items() if k.endswith("_F") and np.isfinite(med)}
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    # descriptors of Luppi et al. Fig. 3 against the simulation reference cells (descriptive)
    dsum = {ses: {k: q([d[k] for d in desc[ses].values()]) for k in ("fp", "E_bn", "CR", "WR", "E_ctr")} for ses in ("REST1", "REST2")}
    e8 = json.load(open(E8_JSON)) if os.path.exists(E8_JSON) else None
    e8_pos = {}
    if e8:
        for cell, rec in e8["cells"].items():
            e8_pos[cell] = {k: dict(cell_estimate=rec[k]["theta"], real_REST1_median=dsum["REST1"][k]["median"]) for k in ("E_bn", "CR", "WR", "E_ctr", "fp")}
    res = dict(prereg=pre, freeze=fsha, admitted=len(R), rejected=rej, fit_summary=summ, retest_neg_spearman=retest, change_vs_minimal=d_vs_min,
               E6_fix=dict(primary_loading="contrast", n_ok=len(ok), status=st, analysable=analysable, real_median_E=med, label=label, percentile_in_E6_families=pct_e6),
               fig3_descriptors=dsum, vs_E8_cells=e8_pos)
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
    for ses, s in summ.items():
        print("E5-FIX", ses, "neg median %.3f [%.3f, %.3f]" % (s["neg"]["median"], s["neg"]["q25"], s["neg"]["q75"]), "neg5 %.3f" % s["neg5"]["median"],
              "FC corr signed/coop %.3f/%.3f" % (s["fc_corr_signed"]["median"], s["fc_corr_coop"]["median"]), "P2<0 share %.2f" % s["p2_negative_share"], flush=True)
    print("retest neg spearman %.3f | change vs minimal" % retest, {k: round(v["median"], 3) for k, v in d_vs_min.items()}, flush=True)
    print("E6-FIX:", label, "| n_ok", len(ok), "| median E %.3f" % med if np.isfinite(med) else "", "| status", st, flush=True)
    print("Fig3 descriptors REST1:", {k: round(v["median"], 3) for k, v in dsum["REST1"].items()}, flush=True)


if __name__ == "__main__":
    main()
