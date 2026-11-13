"""Summary functions imported by the human fitting protocol (luppi_r4e_ext.py)."""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np
from scipy import stats

R4E = DATA_ROOT + "results/p33/r4e/"; EXT = DATA_ROOT + "results/p33/r4e_ext/"; R4V2 = DATA_ROOT + "results/p33/r4v2/"
PREREG = EXT + "r4e_ext_e6_prereg.json"
LOADINGS = ("contrast", "fcpc1", "fiedler")
COMPLETE_FRAC = 0.8


def logistic_logor(y, X, j, ridge=1e-6, iters=50):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))); Wd = p * (1 - p) + 1e-12
        H = X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1]); step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta); beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[j]) if np.all(np.isfinite(beta)) else np.nan


def loading_vector(inp, SC, loading):
    import luppi_hopf as L, luppi_r4e_lib as Lib
    if loading == "fiedler":
        return Lib.fiedler(SC), False
    F = np.corrcoef(L.prep_emp(inp))
    if loading == "contrast":
        N = len(F); Hc = np.eye(N) - np.ones((N, N)) / N; F = Hc @ F @ Hc
    w, V = np.linalg.eigh(F)
    return V[:, -1], bool(abs(w[-1] - w[-2]) <= 1e-9 * max(abs(w[-1]), 1e-30))


def signature(C, SC, inp, loading):
    import luppi_r4e_lib as Lib
    v, tie = loading_vector(inp, SC, loading)
    nz = np.abs(v) >= 1e-12; m = Lib.support(SC) & nz[:, None] & nz[None, :]
    opp = (np.sign(v)[:, None] * np.sign(v)[None, :] < 0)[m]; y = (C[m] < 0)
    cnt = dict(n_edges=int(m.sum()), n_opp=int(opp.sum()), n_nonopp=int((~opp).sum()), minority_nodes=int(min((v > 0).sum(), (v < 0).sum())),
               zero_nodes=int((~nz).sum()), neg_opp=int((y & opp).sum()), neg_nonopp=int((y & ~opp).sum()))
    out = dict(S=float((y & opp).sum() / max(y.sum(), 1)), b=float(opp.mean()) if opp.size else np.nan, neg=float(y.mean()), **cnt)
    if tie:
        status = "tie"
    elif opp.size == 0 or opp.all() or not opp.any():
        status = "constant_predictor"
    elif min(cnt["neg_opp"], cnt["neg_nonopp"], int((~y & opp).sum()), int((~y & ~opp).sum())) == 0:
        status = "separation"
    else:
        sq = np.searchsorted(np.quantile(SC[m], [0.25, 0.5, 0.75]), SC[m])
        X = np.column_stack([np.ones(m.sum()), opp.astype(float)] + [(sq == k).astype(float) for k in (1, 2, 3)])
        E = logistic_logor(y.astype(float), X, 1); status = "ok" if np.isfinite(E) else "failed"
        out["E"] = E
    out["status"] = status
    if status != "ok":
        out["E"] = np.nan
    return out


def ext_load(group):
    import luppi_r4e_ext as X
    fsha, fr = X.verify_freeze(group); expected = {j["key"]: j for j in X.job_list(group)}
    R, seen, rej = [], set(), dict(not_done=0, wrong_prov=0, unexpected=0, duplicate=0)
    for p in sorted(glob.glob(EXT + f"jobs_{group}/*.json")):
        r = json.load(open(p)); j = expected.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != X.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if r["key"] in seen:
            rej["duplicate"] += 1; continue
        seen.add(r["key"]); R.append(r)
    return R, rej, fsha, fr


def load_C(r):
    import luppi_r4e_run as RUN
    C = np.load(r["file"].replace(".json", "_C.npy"))
    if RUN.sha_arr(C) != r["C_sha"]:
        raise SystemExit(f"C_sha mismatch for {r['key']}")
    return C


def sim_sig(r, root, cellpart):
    import luppi_r4e_lib as Lib
    tgt = np.load(root + f"truth/{r['tag']}{cellpart}_target.npy"); inp = Lib.bandpass(tgt) if r["branch"] == "F" else tgt
    _, SC, _ = Lib.demo_subject(r["stratum"]); C = load_C(r)
    return {ld: signature(C, SC, inp, ld) for ld in LOADINGS}


def summarize(rows, key="E"):
    x = np.array([r[key] for r in rows], float); x = x[np.isfinite(x)]
    return dict(n=int(x.size), median=float(np.median(x)) if x.size else np.nan, q25=float(np.quantile(x, 0.25)) if x.size else np.nan,
                q75=float(np.quantile(x, 0.75)) if x.size else np.nan)


def family_table(recs, root, cellpart_fn, n_expected):
    per = {}
    for r in recs:
        if r["model"] == "signed":
            per.setdefault(r["tag"], []).append(sim_sig(r, root, cellpart_fn(r)))
    out = {}
    for ld in LOADINGS:
        rows, status = [], {}
        for tag, v in per.items():
            ok = len(v) == 3 and all(s[ld]["status"] == "ok" for s in v)
            for s in v:
                status[s[ld]["status"]] = status.get(s[ld]["status"], 0) + 1
            row = {k: float(np.mean([s[ld][k] for s in v])) for k in ("S", "b", "neg", "minority_nodes", "n_opp")}
            row["E"] = float(np.mean([s[ld]["E"] for s in v])) if ok else np.nan; row["complete"] = ok; rows.append(row)
        n_ok = sum(r["complete"] for r in rows)
        out[ld] = dict(n_truths_expected=n_expected, n_truths_complete=n_ok, analysable=bool(n_ok >= COMPLETE_FRAC * n_expected), fit_status=status,
                       E=summarize(rows), S=summarize(rows, "S"), b=summarize(rows, "b"), neg=summarize(rows, "neg"), rows=rows)
    return out


def real_cohort(sigs, n_expected):
    out = {}
    for ld in LOADINGS:
        rows = [s[ld] for s in sigs]; ok = [r for r in rows if r["status"] == "ok"]
        st = {}
        for r in rows:
            st[r["status"]] = st.get(r["status"], 0) + 1
        out[ld] = dict(n_expected=n_expected, n_ok=len(ok), analysable=bool(len(ok) >= COMPLETE_FRAC * n_expected), status=st,
                       E=summarize(ok), S=summarize(rows, "S"), b=summarize(rows, "b"), neg=summarize(rows, "neg"),
                       minority_nodes=summarize(rows, "minority_nodes"))
    return out


def pct(x, ref):
    ref = np.asarray(ref, float); ref = ref[np.isfinite(ref)]
    return float(np.mean(ref <= x) * 100) if ref.size and np.isfinite(x) else np.nan


def pushpull_enrichment(C_list, C_ctrl, C_true, SC, G):
    """Paired enrichment increment of the driven truths: log-OR of the drive push-pull indicator
 (sum_m g_im g_jm < 0) in a logistic model of (C < 0) with true-weight and SC-weight quartile effects, seeds pooled;
 returns (log-OR with drive, log-OR without drive, difference)."""
    import luppi_r4e_lib as Lib
    m = Lib.support(SC); pp = ((G @ G.T) < 0)[m].astype(float); t = np.abs(C_true[m]); sc = SC[m]
    tq = np.searchsorted(np.quantile(t, [0.25, 0.5, 0.75]), t); sq = np.searchsorted(np.quantile(sc, [0.25, 0.5, 0.75]), sc)
    X = np.column_stack([np.ones(m.sum()), pp] + [(tq == k).astype(float) for k in (1, 2, 3)] + [(sq == k).astype(float) for k in (1, 2, 3)])
    XX = np.vstack([X] * len(C_list))
    a = logistic_logor(np.concatenate([(C[m] < 0).astype(float) for C in C_list]), XX, 1)
    b = logistic_logor(np.concatenate([(C[m] < 0).astype(float) for C in C_ctrl]), XX, 1)
    return a, b, a - b


def main():
    pre = json.load(open(PREREG))
    if hashlib.sha256(open(__file__, "rb").read()).hexdigest() != pre["sha256"]:
        raise SystemExit("E6 script differs from the pre-registered version")
    import scipy.io as sio
    import luppi_r4e_analyze as A, luppi_r4e_lib as Lib
    Rp, _, _ = A.load(R4E + "jobs", "primary"); Re, _, _ = A.load(R4E + "jobs", "e23")
    Rb, rejb, fb, _ = ext_load("e2b"); R5, rej5, f5, fr5 = ext_load("e5")
    # pinned reference fits (R4V2)
    r4v2 = {}
    for k in range(3):
        for sd in (11, 12, 13):
            p = R4V2 + f"C_s{k}_header_signed_matlab_seed{sd}.npy"
            if hashlib.sha256(open(p, "rb").read()).hexdigest() != fr5["raw"].get(p):
                raise SystemExit(f"{p} differs from its frozen hash")
            r4v2[(k, sd)] = np.load(p)
    n_e2 = 6; fam = {}
    for br in "FR":
        sel = lambda R, **kw: [r for r in R if r["branch"] == br and all(r.get(k) == v for k, v in kw.items())]
        fam[f"E1_QM1_{br}"] = family_table(sel(Rp, exp="E1", family="QM1"), R4E, lambda r: "", 30)
        fam[f"E1_SG_{br}"] = family_table(sel(Rp, exp="E1", family="SG"), R4E, lambda r: "", 3)
        for cell in sorted({r["cell"] for r in Re if r["exp"] == "E2"}):
            fam[f"E2_{cell}_{br}"] = family_table(sel(Re, exp="E2", cell=cell), R4E, lambda r: f"_{r['cell']}", n_e2)
        for cell in sorted({r["cell"] for r in Rb}):
            fam[f"E2b_{cell}_{br}"] = family_table(sel(Rb, cell=cell), EXT, lambda r: f"_{r['cell']}", n_e2)
        fam[f"E3_CMP_{br}"] = family_table(sel(Re, exp="E3", family="CMP"), R4E, lambda r: "", 15)
        fam[f"E3_CMPH_{br}"] = family_table(sel(Re, exp="E3", family="CMP-H"), R4E, lambda r: "", 4)
    fiedler_truth_info = sorted(k for k in fam if k.startswith("E2_") or k.startswith("E2b_S"))
    # real cohorts
    ts = sio.loadmat(DATA_ROOT + "raw/luppi_demo/gec_3/FMRI.mat")["BOLD_timeseries_HCP"].ravel()
    scs = sio.loadmat(DATA_ROOT + "raw/luppi_demo/gec_3/SC.mat")["DTI_fibers_VolNorm_HCP"].ravel()
    demo = []
    for k in range(3):
        sig = [{ld: signature(r4v2[(k, sd)], scs[k].astype(float), ts[k].astype(float), ld) for ld in LOADINGS} for sd in (11, 12, 13)]
        demo.append({ld: dict({q: float(np.mean([s[ld][q] for s in sig])) for q in ("S", "b", "neg", "minority_nodes")},
                              E=float(np.mean([s[ld]["E"] for s in sig])) if all(s[ld]["status"] == "ok" for s in sig) else np.nan,
                              status="ok" if all(s[ld]["status"] == "ok" for s in sig) else "incomplete") for ld in LOADINGS})
    SC5 = np.load(EXT + "e5/SC.npy"); e5sig = {}
    for ses in ("REST1", "REST2"):
        e5sig[ses] = {r["tag"]: {ld: signature(load_C(r), SC5, np.load(EXT + f"e5/{r['tag']}_{ses}_input.npy"), ld) for ld in LOADINGS}
                      for r in R5 if r["session"] == ses and r["model"] == "signed"}
    real = dict(R4V2_demo=real_cohort(demo, 3), E5_REST1=real_cohort(list(e5sig["REST1"].values()), 98), E5_REST2=real_cohort(list(e5sig["REST2"].values()), 98))
    # judgement (primary: PPC REST1, F-branch families, contrast loading)
    ld = "contrast"; rc = real["E5_REST1"][ld]; e2 = fam["E2_tau50_q1_F"][ld]; cmph = fam["E3_CMPH_F"][ld]; cmp_ = fam["E3_CMP_F"][ld]
    med = rc["E"]["median"]
    if not (rc["analysable"] and e2["analysable"] and cmph["analysable"] and cmp_["analysable"]) or not np.isfinite(med):
        label = "not analysable"
    elif med >= e2["E"]["median"] and med >= cmph["E"]["q25"]:
        label = "ordinal label A: real median E at or above the hidden-drive (q 0.2) median and the aligned-competition 25th percentile"
    elif med <= cmp_["E"]["median"]:
        label = "ordinal label C: real median E at or below the random-competition median"
    else:
        label = "ordinal label B: between"
    percentiles = {k: pct(med, [r["E"] for r in v[ld]["rows"]]) for k, v in fam.items() if k.endswith("_F")}
    common = sorted(set(e5sig["REST1"]) & set(e5sig["REST2"]))
    def rt(key, ldg):
        a = np.array([e5sig["REST1"][s][ldg][key] for s in common], float); b = np.array([e5sig["REST2"][s][ldg][key] for s in common], float)
        ok = np.isfinite(a) & np.isfinite(b)
        return dict(n=int(ok.sum()), spearman=float(stats.spearmanr(a[ok], b[ok])[0]) if ok.sum() > 2 else np.nan)
    retest = {ldg: dict(E=rt("E", ldg), neg=rt("neg", ldg)) for ldg in LOADINGS}
    # shared-drive simulation summary
    ctrl = {(r["tag"], r["branch"], r["seed"]): r for r in Rp if r["exp"] == "E1" and r["family"] == "QM1" and r["model"] == "signed"}
    e2b = {}
    for cell in sorted({r["cell"] for r in Rb}):
        M = 1 if cell.startswith("S") else 3
        for br in "RF":
            per = {}
            for r in Rb:
                if r["cell"] == cell and r["branch"] == br:
                    per.setdefault(r["tag"], {}).setdefault(r["seed"], {})[r["model"]] = r
            rows = []
            for tag, sd in per.items():
                s = int(tag.split("_")[1]); st = (s - 1) % 3; full = [x for x in sd.values() if "signed" in x and "coop" in x]
                if len(full) < 3:
                    rows.append(dict(tag=tag, complete=False)); continue
                G = np.load(EXT + f"truth/G{M}_stratum{st}.npy"); _, SC, _ = Lib.demo_subject(st); m = Lib.support(SC)
                C_s = [load_C(x["signed"]) for x in full]; C_c = [load_C(ctrl[(tag, br, x["signed"]["seed"])]) for x in full]
                pp = ((G @ G.T) < 0)[m]
                lE, l0, dl = pushpull_enrichment(C_s, C_c, np.load(R4E + f"truth/{tag}_C.npy"), SC, G)
                rows.append(dict(tag=tag, stratum=st, complete=True, neg=float(np.mean([x["signed"]["metrics"]["frac_neg"] for x in full])),
                                 d_neg=float(np.mean([x["signed"]["metrics"]["frac_neg"] - ctrl[(tag, br, x["signed"]["seed"])]["metrics"]["frac_neg"] for x in full])),
                                 dP2=float(np.mean([x["signed"]["mse_ho"] - x["coop"]["mse_ho"] for x in full])),
                                 pushpull_share=float(np.mean([((C[m] < 0) & pp).sum() / max((C[m] < 0).sum(), 1) for C in C_s])),
                                 pushpull_base=float(pp.mean()), logor_E2b=lE, logor_E1=l0, d_enrich=dl))
            ok = [r for r in rows if r["complete"]]
            mean_neg = float(np.mean([r["neg"] for r in ok])) if ok else np.nan
            e2b[f"{cell}_{br}"] = dict(n_expected=n_e2, n_complete=len(ok), analysable=bool(len(ok) >= COMPLETE_FRAC * n_e2), rows=rows,
                                       neg=mean_neg, reaches_real_count=bool(len(ok) >= COMPLETE_FRAC * n_e2 and mean_neg >= 0.20),
                                       **{k: float(np.mean([r[k] for r in ok])) if ok else np.nan for k in ("d_neg", "dP2", "pushpull_share", "pushpull_base", "d_enrich")})
    # PPC summary
    e5 = {}
    for ses in ("REST1", "REST2"):
        S = [r for r in R5 if r["session"] == ses]; sg = {r["tag"]: r for r in S if r["model"] == "signed"}; cp = {r["tag"]: r for r in S if r["model"] == "coop"}
        neg = [r["neg"]["frac_neg_thr0.0"] for r in sg.values()]
        e5[ses] = dict(n_signed=len(sg), n_coop=len(cp), n_expected=98, neg=summarize([dict(E=x) for x in neg]),
                       neg5=summarize([dict(E=r["neg"]["frac_neg_thr0.05"]) for r in sg.values()]),
                       fc_corr_signed=float(np.median([r["fc_corr"] for r in sg.values()])) if sg else np.nan,
                       fc_corr_coop=float(np.median([r["fc_corr"] for r in cp.values()])) if cp else np.nan,
                       dP2_negative_share=float(np.mean([sg[t]["mse_ho"] - cp[t]["mse_ho"] < 0 for t in sg if t in cp])) if sg else np.nan)
    out = dict(prereg=pre, admitted=dict(primary=len(Rp), e23=len(Re), e2b=len(Rb), e5=len(R5)), rejected=dict(e2b=rejb, e5=rej5),
               freezes=dict(e2b=fb, e5=f5), families=fam, fiedler_uses_truth_information=fiedler_truth_info, real=real,
               E6=dict(primary_loading=ld, real_median_E_REST1=med, label=label, percentile_in_families=percentiles, retest=retest),
               E2b=e2b, E5=e5)
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
    print("E6 (contrast loading):", label, "| real median E %.3f" % med if np.isfinite(med) else "", flush=True)
    for k, v in real.items():
        print("  real", k, {l: dict(n_ok=v[l]["n_ok"], E=round(v[l]["E"]["median"], 3) if np.isfinite(v[l]["E"]["median"]) else None,
                                   b=round(v[l]["b"]["median"], 3), neg=round(v[l]["neg"]["median"], 3)) for l in LOADINGS}, flush=True)
    for k in sorted(percentiles):
        f = fam[k]["contrast"]
        print("  family", k, "analysable", f["analysable"], "median E %.3f" % f["E"]["median"] if np.isfinite(f["E"]["median"]) else "median E nan",
              "neg %.3f" % f["neg"]["median"], "| real median E percentile %.0f" % percentiles[k] if np.isfinite(percentiles[k]) else "", flush=True)
    print("retest", retest, flush=True)
    for k, v in e2b.items():
        print("E2b", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "rows"}, flush=True)
    for k, v in e5.items():
        print("E5", k, v, flush=True)


if __name__ == "__main__":
    main()
