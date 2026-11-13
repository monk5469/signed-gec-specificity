"""Simulation 2 analysis (Sections 2.6.5, 3.4, 3.5): counterexample replication, positive controls, edge AUC of -C and -FC,
matched-input equivalence and RAWX classification.
Usage: python luppi_r4e_e11_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e_e11/"; P33 = DATA_ROOT + "results/p33/"; NETS = P33 + "r4e_e8/networks.json"
PREREG = OUT + "r4e_e11_analysis_prereg.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def extras(C, SC, inp, nets):
    import luppi_hopf as L, luppi_r4e_lib as Lib, luppi_r4e_e8_analyze as A8
    m = Lib.support(SC); y = C[m] < 0; nn = np.asarray(nets); b = (nn[:, None] != nn[None, :])[m]; F = np.corrcoef(L.prep_emp(inp))
    q = np.searchsorted(np.quantile(F[m], [0.25, 0.5, 0.75]), F[m]); Xf = np.column_stack([A8.design(SC, m, b)] + [(q == k).astype(float) for k in (1, 2, 3)])
    return dict(E_bn_fc=A8.logor(y, Xf))


def load():
    import luppi_r4e_e11 as E, luppi_r4e_run as RUN
    fsha, fr = E.verify_freeze(); expected = {j["key"]: j for j in E.job_list()}
    rej = dict(malformed=0, unexpected=0, field_mismatch=0, not_done=0, wrong_prov=0, duplicate=0, nonfinite=0, bad_matrix=0); seen = set(); R = {}
    for p in sorted(glob.glob(OUT + "jobs/*.json")):
        try:
            r = json.load(open(p)); assert isinstance(r, dict)
        except Exception:
            rej["malformed"] += 1; continue
        j = expected.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if any(r.get(k) != v for k, v in j.items()):
            rej["field_mismatch"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != E.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if not np.isfinite(r.get("mse_ho", np.nan)):
            rej["nonfinite"] += 1; continue
        if j["key"] in seen:
            rej["duplicate"] += 1; continue
        if j["model"] == "signed":
            try:
                C = np.load(j["file"].replace(".json", "_C.npy"))
            except Exception:
                C = None
            if C is None or C.shape != (100, 100) or not np.all(np.isfinite(C)) or RUN.sha_arr(C) != r.get("C_sha"):
                rej["bad_matrix"] += 1; continue
            r["_C"] = C
        seen.add(j["key"]); R.setdefault((j["ttype"], j["cond"], j["tag"]), {})[j["model"]] = (j, r)
    return R, rej, fsha


def edge_auc(score, Ct, SC):
    """P(score on a true-negative support entry > score on a true-positive support entry), average ranks for ties."""
    from scipy.stats import rankdata
    import luppi_r4e_lib as Lib
    m = Lib.support(SC); w, t = score[m], Ct[m]; a, b = w[t < 0], w[t > 0]
    if a.size == 0 or b.size == 0:
        return np.nan
    r = rankdata(np.r_[a, b], method="average")
    return float((r[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size))


def enough(stat, k=8):
    """Per-estimand minimum data rule: >= k finite observations in every stratum."""
    return all(int(stat["n"].get(h, stat["n"].get(str(h), 0))) >= k for h in range(3))


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("E11 analysis script differs from the pre-registered version")
    import luppi_r4e_e11 as E
    R, rej, fsha = load(); man = json.load(open(E.MAN))
    res = compute(R, rej, fsha, pre, man)
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float); report(res)


def compute(R, rej, fsha, pre, man):
    import luppi_hopf as L, luppi_r4e_e11 as E, luppi_r4e_e8_analyze as A8, luppi_r4e_lib as Lib
    nets = json.load(open(NETS))["networks"]; tg = json.load(open(E.TARGETS)); ref_in = tg["input_reference"]
    valid = {t["tag"]: t for t in man["targets"] if t.get("valid")}
    rows = {}
    for (tt, cond, tag), d in R.items():
        if "signed" not in d or "coop" not in d or tag not in valid:
            continue
        j, r = d["signed"]; C = r["_C"]; s = int(tag.split("_")[1]); Ct, _, k, _ = E.truth_c(s, tt); _, SC, _ = Lib.demo_subject(k)
        inp = Lib.bandpass(np.load(E.job_files(j)[2])); m = Lib.support(SC)
        try:
            dsc = A8.descriptors(C, SC, inp, nets); ex = extras(C, SC, inp, nets)
        except Exception:
            dsc = dict(E_bn=np.nan); ex = dict(E_bn_fc=np.nan)
        mt = r["metrics"]; mm = valid[tag]["measures"][tt][cond]; F = np.corrcoef(L.prep_emp(inp))
        rows.setdefault((tt, cond), {})[tag] = dict(stratum=k, fp=float((C[m] < 0).mean()), thr5=float((C[m] < -0.05 * np.abs(C[m]).max()).mean()),
                                                    p2=float(r["mse_ho"] - d["coop"][1]["mse_ho"]), E_bn=dsc["E_bn"], E_bn_fc=ex["E_bn_fc"],
                                                    sensitivity=mt.get("sensitivity", np.nan), specificity=mt.get("specificity", np.nan), precision=mt.get("precision", np.nan),
                                                    auc=edge_auc(-C, Ct, SC) if tt != "COOP" else np.nan, auc_fc=edge_auc(-F, Ct, SC) if tt != "COOP" else np.nan,
                                                    nf=mm["neg_fc"], gs=mm["gs"])
    st = lambda pairs: A8.strat_t(pairs); fin = np.isfinite
    out = {}
    for tt in E.TTYPES:
        for cond in E.CONDS:
            rr = rows.get((tt, cond), {}); n_by = {h: sum(1 for v in rr.values() if v["stratum"] == h) for h in range(3)}
            rec = dict(n_by_stratum=n_by, complete=all(n_by[h] >= 8 for h in range(3)),
                       **{key: st([(v["stratum"], v[key]) for v in rr.values() if fin(v[key])]) for key in ("fp", "thr5", "p2", "E_bn", "E_bn_fc", "auc", "auc_fc", "sensitivity", "precision")},
                       p2_negative_share=float(np.mean([v["p2"] < 0 for v in rr.values()])) if rr else np.nan,
                       input={key: float(np.median([valid[t]["measures"][tt][cond][key] for t in valid])) for key in ("gs", "neg_fc", "mean_fc", "sd", "mod")} if valid else None,
                       var_fractions={q: float(np.median([valid[t]["var_fractions"][tt][cond][q] for t in valid])) for q in ("neural", "source", "noise", "cross")} if valid else None)
            nu, dose = E.cond_split(cond)
            if dose != "raw":
                base = rows.get((tt, f"{nu}-raw"), {})
                rec["d_fp_vs_raw"] = st([(v["stratum"], v["fp"] - base[t]["fp"]) for t, v in rr.items() if t in base])
            if tt != "COOP":
                co = rows.get(("COOP", cond), {}); p0 = rows.get((tt, "N0-raw"), {})
                rec["total_effect"] = dict(d_fp=st([(v["stratum"], v["fp"] - co[t]["fp"]) for t, v in rr.items() if t in co]),
                                           d_p2=st([(v["stratum"], v["p2"] - co[t]["p2"]) for t, v in rr.items() if t in co]))
                rec["d_auc_vs_N0raw"] = st([(v["stratum"], v["auc"] - p0[t]["auc"]) for t, v in rr.items() if t in p0 and fin(v["auc"]) and fin(p0[t]["auc"])])
                rec["auc_minus_fc_baseline"] = st([(v["stratum"], v["auc"] - v["auc_fc"]) for v in rr.values() if fin(v["auc"]) and fin(v["auc_fc"])])
            out[f"{tt}_{cond}"] = rec
    # ---------------- G (manifest, all valid targets; independent of job admission)
    G = {}
    for tt in E.TTYPES:
        if not all(sum(1 for t in valid.values() if t["stratum"] == h) >= 8 for h in range(3)):
            G[tt] = dict(label="not analysable"); continue
        a = float(np.median([t["measures"][tt]["N49-d136"]["neg_fc"] for t in valid.values()])); b = float(np.median([t["input_only"][tt]["d042"]["neg_fc"] for t in valid.values()]))
        G[tt] = dict(d136=dict(sim=a, real=ref_in["d136"]["neg_fc"]), d042=dict(sim=b, real=ref_in["gsr"]["neg_fc"]),
                     label=bool(abs(a - ref_in["d136"]["neg_fc"]) <= 0.08 and abs(b - ref_in["gsr"]["neg_fc"]) <= 0.08))
    # ---------------- RAWX: mismatch of the fixed simulated scenario with the real raw (mot) negative-FC benchmark
    RAWX = {}
    for tt in E.TTYPES:
        if not valid:
            RAWX[tt] = dict(label="not analysable"); continue
        a = float(np.median([t["measures"][tt]["N49-raw"]["neg_fc"] for t in valid.values()]))
        RAWX[tt] = dict(sim=a, real=ref_in["raw"]["neg_fc"], mismatch_with_real_raw=bool(a > ref_in["raw"]["neg_fc"] + 0.08))
    # ---------------- Q1 (COOP replication; confirmatory)
    r = out["COOP_N49-d136"]
    if G["COOP"]["label"] is not True:
        Q1 = dict(label="not interpretable" if G["COOP"]["label"] is False else "not analysable")
    elif enough(r["fp"]) and enough(r["p2"]) and enough(r["d_fp_vs_raw"]):
        c = dict(fp_ge_020=bool(r["fp"]["theta"] >= 0.20), signed_better=bool(fin(r["p2"]["hi"]) and r["p2"]["hi"] < 0),
                 d_fp_lower_gt_010=bool(fin(r["d_fp_vs_raw"]["lo"]) and r["d_fp_vs_raw"]["lo"] > 0.10))
        Q1 = dict(criteria=c, label=all(c.values()))
    else:
        Q1 = dict(label="not analysable")
    # ---------------- Q3 (positive control and degradation boundary)
    Q3 = {}
    for tt in ("CMPN", "CMPH"):
        p0 = out[f"{tt}_N0-raw"]; b = out[f"{tt}_N49-d136"]; rec = {}
        if enough(p0["auc"]) and enough(p0["total_effect"]["d_fp"]):
            pc = bool(p0["auc"]["theta"] >= 0.80 and fin(p0["auc"]["lo"]) and p0["auc"]["lo"] >= 0.75
                      and fin(p0["total_effect"]["d_fp"]["lo"]) and p0["total_effect"]["d_fp"]["lo"] > 0.05)
            rec["positive_control"] = pc if pc else "the joint positive-control criterion was not established"
        else:
            rec["positive_control"] = "not analysable"
        rec["N0raw"] = dict(auc=p0["auc"], auc_fc_baseline=p0["auc_fc"], auc_minus_fc=p0.get("auc_minus_fc_baseline"), d_fp=p0["total_effect"]["d_fp"])
        rec["N0raw_beyond_fc"] = (bool(fin(p0["auc_minus_fc_baseline"]["lo"]) and p0["auc_minus_fc_baseline"]["lo"] > 0) if enough(p0["auc_minus_fc_baseline"]) else "not analysable")
        rec["by_condition"] = {q: dict(auc=out[f"{tt}_{q}"]["auc"], auc_fc_baseline=out[f"{tt}_{q}"]["auc_fc"], sensitivity=out[f"{tt}_{q}"]["sensitivity"],
                                       precision=out[f"{tt}_{q}"]["precision"], d_fp=out[f"{tt}_{q}"]["total_effect"]["d_fp"], d_p2=out[f"{tt}_{q}"]["total_effect"]["d_p2"],
                                       d_auc_vs_N0raw=out[f"{tt}_{q}"]["d_auc_vs_N0raw"]) for q in E.CONDS}
        if rec["positive_control"] is True:
            if enough(b["auc"]) and enough(b["d_auc_vs_N0raw"]):
                rec["substantial_degradation_of_edge_ranking_under_COOP_calibrated_policy"] = bool(fin(b["auc"]["hi"]) and b["auc"]["hi"] < 0.70
                                                                                              and fin(b["d_auc_vs_N0raw"]["hi"]) and b["d_auc_vs_N0raw"]["hi"] < -0.10)
            else:
                rec["substantial_degradation_of_edge_ranking_under_COOP_calibrated_policy"] = "not analysable"
        Q3[tt] = rec
    # ---------------- Q4 (secondary; per competition type vs COOP, gated on G for both, within common input-negative-FC support)
    rng = np.random.default_rng(11); Q4 = {}; MARGIN = 0.023
    for T in ("CMPN", "CMPH"):
        if not (G["COOP"]["label"] is True and G[T]["label"] is True):
            Q4[T] = dict(label="not interpretable (G)" if "not analysable" not in (G["COOP"]["label"], G[T]["label"]) else "not analysable"); continue
        sel = [(tt, t, v) for (tt, q), rr in rows.items() if tt in ("COOP", T) and not q.startswith("N0") for t, v in rr.items()]
        if not any(a == T for a, *_ in sel) or not any(a == "COOP" for a, *_ in sel):
            Q4[T] = dict(label="not analysable"); continue
        nfc = np.array([v["nf"] for a, _, v in sel if a == "COOP"]); nft = np.array([v["nf"] for a, _, v in sel if a == T])
        lo_s, hi_s = max(np.quantile(nfc, 0.05), np.quantile(nft, 0.05)), min(np.quantile(nfc, 0.95), np.quantile(nft, 0.95))
        sel = [x for x in sel if lo_s <= x[2]["nf"] <= hi_s]; tags = sorted({t for _, t, _ in sel}); recs = {}
        for key in ("fp", "p2"):
            y = np.array([v[key] for *_, v in sel]); nf = np.array([v["nf"] for *_, v in sel]); ind = np.array([float(a == T) for a, *_ in sel])
            X = np.column_stack([np.ones_like(nf), nf, nf ** 2, ind])
            if len(sel) < 20 or np.linalg.matrix_rank(X) < 4 or ind.sum() < 10 or (1 - ind).sum() < 10:
                recs[key] = dict(label="not analysable", n=len(sel)); continue
            b = np.linalg.lstsq(X, y, rcond=None)[0]; idx = {t: [i for i, s in enumerate(sel) if s[1] == t] for t in tags}; B = []; deficient = 0
            for _ in range(2000):
                ii = np.concatenate([idx[t] for t in rng.choice(tags, len(tags))])
                if np.linalg.matrix_rank(X[ii]) < 4:
                    deficient += 1; continue
                B.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0][3])
            if deficient > 100:
                recs[key] = dict(label="not analysable (rank-deficient resamples)", deficient=deficient); continue
            lo, hi = float(np.quantile(B, 0.05)), float(np.quantile(B, 0.95)); rr = dict(coef=float(b[3]), ci90=[lo, hi], n=len(sel), support=[float(lo_s), float(hi_s)], deficient=deficient)
            if key == "fp":
                rr["label"] = "equivalent" if (lo >= -MARGIN and hi <= MARGIN) else ("not equivalent" if (hi < -MARGIN or lo > MARGIN) else "inconclusive")
            recs[key] = rr
        Q4[T] = dict(recs, margin=MARGIN, note="conditional on input negative FC within common support: incremental information beyond input negative FC, not the total effect")
    return dict(prereg=pre, freeze=fsha, admitted=sum(len(v) for v in rows.values()), rejected=rej, input_reference=ref_in, cells=out, G=G, RAWX=RAWX, Q1=Q1, Q3=Q3, Q4=Q4,
                note="descriptive except the pre-specified labels; constructed setting; observation-level cooperative sources; independence = new stochastic truths "
                     "conditional on the same three structural subjects; no attribution beyond the construction")


def report(res):
    f3 = lambda d: "%.3f [%.3f, %.3f]" % (d["theta"], d["lo"], d["hi"])
    for key, r in res["cells"].items():
        print(key, "| in gs %.3f nf %.3f neural %.2f" % (r["input"]["gs"], r["input"]["neg_fc"], r["var_fractions"]["neural"]) if r["input"] else "", "| FP", f3(r["fp"]),
              "| P2", f3(r["p2"]), "share<0 %.2f" % r["p2_negative_share"], "| AUC", f3(r["auc"]), "| AUC_FC", f3(r["auc_fc"]),
              "| dFPc", f3(r["total_effect"]["d_fp"]) if "total_effect" in r else "-", flush=True)
    print("G", res["G"], flush=True); print("RAWX", res["RAWX"], flush=True); print("Q1", res["Q1"], flush=True)
    print("Q3", {t: {k: v for k, v in q.items() if k != "by_condition"} for t, q in res["Q3"].items()}, flush=True); print("Q4", res["Q4"], flush=True)


if __name__ == "__main__":
    main()
