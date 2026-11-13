"""Simulation 1 analysis (Sections 2.6.5, 3.4): FP, P2 and input statistics per dose with stratified t intervals, and the
counterexample and dose-curve criteria.
Usage: python luppi_r4e_e10c_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

OUT = DATA_ROOT + "results/p33/r4e_e10c/"; P33 = DATA_ROOT + "results/p33/"; NETS = P33 + "r4e_e8/networks.json"
PREREG = OUT + "r4e_e10c_analysis_prereg.json"


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
    nb, nw = int((y & b).sum()), int((y & ~b).sum())
    sep = "none_negative" if y.sum() == 0 else "pos_sep" if nw == 0 else "neg_sep" if nb == 0 else None
    return dict(E_bn_fc=A8.logor(y, Xf), RD=float(y[b].mean() - y[~b].mean()), sep=sep)


def load():
    import luppi_r4e_e10c as E, luppi_r4e_run as RUN
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
        seen.add(j["key"]); R.setdefault((j["ttype"], j["cell"], j["dose"], j["tag"]), {})[j["model"]] = (j, r)
    return R, rej, fsha


def auc(a, b):
    """P(a > b) + 1/2 P(a = b) over all pairs (a from CMPN, b from COOP), finite values only."""
    a = np.asarray([v for v in a if np.isfinite(v)]); b = np.asarray([v for v in b if np.isfinite(v)])
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (a.size * b.size))


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("E10c analysis script differs from the pre-registered version")
    import luppi_r4e_e10c as E, luppi_r4e_e8_analyze as A8, luppi_r4e_lib as Lib
    R, rej, fsha = load(); nets = json.load(open(NETS))["networks"]; man = json.load(open(E.MAN))
    tg = json.load(open(E.TARGETS))
    if tg["source_sha"] != sha(tg["source"]):
        raise SystemExit("real reference does not match the verified PPC snapshot")
    real = tg["reference"]
    meas = {(t["tag"], t["ttype"], t["cell"]): t["measures"] for t in man["targets"] if t.get("valid")}
    rows = {}
    for (tt, cell, dose, tag), d in R.items():
        if "signed" not in d or "coop" not in d:
            continue
        j, r = d["signed"]; C = r["_C"]; s = int(tag.split("_")[1]); _, _, k, _ = E.truth_c(s, tt); _, SC, _ = Lib.demo_subject(k)
        inp = Lib.bandpass(np.load(E.job_files(j)[2])); m = Lib.support(SC)
        try:
            dsc = A8.descriptors(C, SC, inp, nets); ex = extras(C, SC, inp, nets); why = None
        except Exception as e:
            dsc = dict(E_bn=np.nan); ex = dict(E_bn_fc=np.nan, RD=np.nan, sep=None); why = repr(e)[:120]
        mt = r["metrics"]; mm = meas.get((tag, tt, cell), {}).get(dose, {})
        rows.setdefault((tt, cell, dose), {})[tag] = dict(stratum=k, fp=float((C[m] < 0).mean()), thr5=float((C[m] < -0.05 * np.abs(C[m]).max()).mean()),
                                                          p2=float(r["mse_ho"] - d["coop"][1]["mse_ho"]), E_bn=dsc["E_bn"], E_bn_fc=ex["E_bn_fc"], RD=ex["RD"], sep=ex["sep"],
                                                          sensitivity=mt.get("sensitivity", np.nan), specificity=mt.get("specificity", np.nan), precision=mt.get("precision", np.nan),
                                                          gs=mm.get("gs", np.nan), neg_fc=mm.get("neg_fc", np.nan), mean_fc=mm.get("mean_fc", np.nan), sd=mm.get("sd", np.nan),
                                                          mod=mm.get("mod", np.nan), descriptor_error=why)
    out = {}
    for tt in E.TTYPES:
        for cell in E.CELLS:
            for dose in E.CELL_DOSES[cell]:
                rr = rows.get((tt, cell, dose), {}); n_by = {h: sum(1 for v in rr.values() if v["stratum"] == h) for h in range(3)}
                fps = [v["fp"] for v in rr.values()]
                rec = dict(n_by_stratum=n_by, complete=all(n_by[h] >= 8 for h in range(3)),
                           **{key: A8.strat_t([(v["stratum"], v[key]) for v in rr.values()]) for key in ("fp", "thr5", "p2", "E_bn", "E_bn_fc", "RD")},
                           fp_q10_q90=[float(np.quantile(fps, 0.1)), float(np.quantile(fps, 0.9))] if fps else None,
                           p2_negative_share=float(np.mean([v["p2"] < 0 for v in rr.values()])) if rr else np.nan,
                           sep_counts={x: sum(v["sep"] == x for v in rr.values()) for x in ("pos_sep", "neg_sep", "none_negative")},
                           input={key: float(np.nanmedian([v[key] for v in rr.values()])) for key in ("gs", "neg_fc", "mean_fc", "sd", "mod")} if rr else None,
                           recovery=({key: float(np.nanmedian([v[key] for v in rr.values()])) for key in ("sensitivity", "specificity", "precision")} if tt != "COOP" and rr else None))
                if dose != "raw":
                    base = rows.get((tt, cell, "raw"), {})
                    rec["d_fp_vs_raw"] = A8.strat_t([(v["stratum"], v["fp"] - base[t]["fp"]) for t, v in rr.items() if t in base])
                    rec["n_paired_by_stratum"] = {h: sum(1 for t, v in rr.items() if t in base and v["stratum"] == h) for h in range(3)}
                out[f"{tt}_{cell}_{dose}"] = rec
    # ------------------------------------------------ input level
    ref_in = tg["input_reference"]; valid = [t for t in man["targets"] if t.get("valid")]
    def tvals(tt, cell, f):
        sel = [t for t in valid if t["ttype"] == tt and t["cell"] == cell]
        return [(int(t["tag"].split("_")[1]) - 1) % 3 for t in sel], [f(t) for t in sel]
    G, G042, INP = {}, {}, {}
    for tt in E.TTYPES:
        for cell in E.CELLS:
            st, _ = tvals(tt, cell, lambda t: 0)
            if not all(st.count(h) >= 8 for h in range(3)):
                G[f"{tt}_{cell}"] = G042[f"{tt}_{cell}"] = dict(label="not analysable"); continue
            per = {}
            for d in ("d284", "d189", "d136", "gsr"):
                _, v = tvals(tt, cell, lambda t, d=d: t["measures"][d]["neg_fc"])
                per[d] = dict(sim_median=float(np.median(v)), sim_q25_q75=[float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))], real=ref_in[d]["neg_fc"])
                per[d]["ok"] = bool(abs(per[d]["sim_median"] - per[d]["real"]) <= 0.08)
            G[f"{tt}_{cell}"] = dict(per_dose=per, label=all(v["ok"] for v in per.values()))
            _, v = tvals(tt, cell, lambda t: t["input_only"]["d042"]["neg_fc"])
            G042[f"{tt}_{cell}"] = dict(sim_median=float(np.median(v)), real=ref_in["gsr"]["neg_fc"], label=bool(abs(float(np.median(v)) - ref_in["gsr"]["neg_fc"]) <= 0.08))
            INP[f"{tt}_{cell}"] = {d: dict(realised_gs=float(np.median(tvals(tt, cell, lambda t, d=d: t["realised_gs"][d])[1])),
                                           var_fractions={q: float(np.median(tvals(tt, cell, lambda t, d=d, q=q: t["var_fractions"][d][q])[1])) for q in ("neural", "source", "noise", "cross")},
                                           heldout_neg_fc=float(np.median(tvals(tt, cell, lambda t, d=d: t["heldout_measures"][d]["neg_fc"])[1])),
                                           heldout_gs=float(np.median(tvals(tt, cell, lambda t, d=d: t["heldout_measures"][d]["gs"])[1])))
                                   for d in E.CELL_DOSES[cell]}
    # paired input-level FC / lag-1 FC differences: target residuals, median over truths of the mean |difference|
    import luppi_hopf as L
    PAIRDIFF = {}
    for cell in E.CELLS:
        for other in ("CMPN", "CMPNp"):
            for d in E.CELL_DOSES[cell]:
                fc, lg = [], []
                for s in E.SEEDS:
                    pa = E.OUT + f"truth/SCG_{s}_COOP_{cell}_{d}_target.npy"; pb = E.OUT + f"truth/SCG_{s}_{other}_{cell}_{d}_target.npy"
                    if os.path.exists(pa) and os.path.exists(pb):
                        xa, xb = Lib.bandpass(np.load(pa)), Lib.bandpass(np.load(pb)); iu = np.triu_indices(100, 1)
                        fc.append(float(np.abs(np.corrcoef(L.prep_emp(xa))[iu] - np.corrcoef(L.prep_emp(xb))[iu]).mean()))
                        lg.append(float(np.abs(L.lag_fc(xa, 1, "matlab") - L.lag_fc(xb, 1, "matlab")).mean()))
                PAIRDIFF[f"{other}_vs_COOP_{cell}_{d}"] = dict(n=len(fc), fc_mean_abs_diff=float(np.median(fc)) if fc else np.nan,
                                                               lagfc_mean_abs_diff=float(np.median(lg)) if lg else np.nan)
    # ------------------------------------------------ estimator level
    L1, L2, L3, L3p, reading = {}, {}, {}, {}, {}
    for tt in E.TTYPES:
        for cell in E.CELLS:
            r = out[f"{tt}_{cell}_d136"]
            if r["complete"] and all(r["n_paired_by_stratum"][h] >= 8 for h in range(3)):
                c = dict(fp_ge_020=bool(r["fp"]["theta"] >= 0.20), signed_better=bool(np.isfinite(r["p2"]["hi"]) and r["p2"]["hi"] < 0),
                         d_fp_lower_gt_010=bool(np.isfinite(r["d_fp_vs_raw"]["lo"]) and r["d_fp_vs_raw"]["lo"] > 0.10))
                L1[f"{tt}_{cell}"] = dict(criteria=c, label=all(c.values()))
            else:
                L1[f"{tt}_{cell}"] = dict(label="not analysable")
            recs = {d: out[f"{tt}_{cell}_{d}"] for d in E.DOSES}
            if not all(v["complete"] for v in recs.values()):
                L2[f"{tt}_{cell}"] = dict(label="not analysable"); continue
            per = {}
            for d, v in recs.items():
                rf = real[d]["fp"]["median"]; rs = real[d]["p2_negative_share"]
                per[d] = dict(real_fp=rf, sim_fp_q10_q90=v["fp_q10_q90"], fp_in_band=bool(v["fp_q10_q90"][0] <= rf <= v["fp_q10_q90"][1]),
                              real_share=rs, sim_share=v["p2_negative_share"], share_within_020=bool(abs(rs - v["p2_negative_share"]) <= 0.20),
                              note="gsr: simulated exact parcel-mean GSR vs real upstream GSR (operator-mismatched)" if d == "gsr" else None)
                per[d]["satisfied"] = per[d]["fp_in_band"] and per[d]["share_within_020"]
            n_sat = sum(p["satisfied"] for p in per.values())
            L2[f"{tt}_{cell}"] = dict(per_dose=per, n_satisfied=n_sat, label=bool(n_sat >= 4 and per["raw"]["satisfied"] and per["d136"]["satisfied"]))
    rng = np.random.default_rng(20261011)
    def finite_by(rr, key):
        return {h: sum(1 for v in rr.values() if v["stratum"] == h and np.isfinite(v[key])) for h in range(3)}
    for cell in E.CELLS:
        for other in ("CMPN", "CMPNp"):
            for dose in ("raw", "d136"):
                a, b = rows.get((other, cell, dose), {}), rows.get(("COOP", cell, dose), {}); rec = {}
                for key in ("fp", "p2", "E_bn_fc"):                                                  #: score-specific completeness
                    fa, fb = finite_by(a, key), finite_by(b, key)
                    if not all(fa[h] >= 8 and fb[h] >= 8 for h in range(3)):
                        rec[key] = dict(label="not analysable", n_finite=dict(other=fa, COOP=fb)); continue
                    pa = {h: [v[key] for v in a.values() if v["stratum"] == h and np.isfinite(v[key])] for h in range(3)}
                    pb = {h: [v[key] for v in b.values() if v["stratum"] == h and np.isfinite(v[key])] for h in range(3)}
                    A = auc(sum(pa.values(), []), sum(pb.values(), [])); boots = []
                    for _ in range(2000):                                                              # stratified bootstrap over truths
                        boots.append(auc(np.concatenate([rng.choice(pa[h], len(pa[h])) for h in range(3)]),
                                         np.concatenate([rng.choice(pb[h], len(pb[h])) for h in range(3)])))
                    rec[key] = dict(auc=A, ci95=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], distinguishable=bool(abs(A - 0.5) >= 0.30))
                    if other == "CMPNp":                                                               # paired: same seed / streams / c / sigma
                        rec[key]["paired_diff"] = A8.strat_t([(v["stratum"], v[key] - b[t][key]) for t, v in a.items()
                                                              if t in b and np.isfinite(v[key]) and np.isfinite(b[t][key])])
                (L3 if other == "CMPN" else L3p)[f"{cell}_{dose}"] = rec
    for cell in E.CELLS:
        labs = {f"{tt} {nm}": gg[f"{tt}_{cell}"]["label"] for tt in ("COOP", "CMPN") for nm, gg in (("G", G), ("G042", G042))}
        if any(v == "not analysable" for v in labs.values()):
            reading[cell] = dict(statement="not analysable (input completeness)"); continue
        fails = [k for k, v in labs.items() if v is not True]
        if fails:
            reading[cell] = dict(statement=f"not interpretable: input realism gate failed ({', '.join(fails)})"); continue
        l1c, l3 = L1[f"COOP_{cell}"]["label"], L3[f"{cell}_d136"]
        if l1c == "not analysable":
            reading[cell] = dict(statement="not analysable"); continue
        ra = ("construction-conditional counterexample: the operational L1 conjunction (FP point >= 0.20, P2 upper < 0, paired dFP lower > 0.10 "
              "at the acc-matched dose) is met in the cooperative construction" if l1c else
              "the operational L1 conjunction is not met in the cooperative construction")
        by = [k for k, v in l3.items() if v.get("distinguishable") is True]; na = [k for k, v in l3.items() if v.get("label") == "not analysable"]
        rb = (f"descriptively, the sign-flipped (CMPN) world differs from the cooperative world at the acc dose by {', '.join(by)} (point AUC)" if by else
              "descriptively, no tested score separates the CMPN and COOP worlds at the acc dose at the |AUC - 0.5| >= 0.30 threshold; "
              "this is NOT evidence of non-identifiability")
        reading[cell] = dict(R_a=ra, R_b=rb, R_b_not_analysable=na, R_b_paired_control=L3p.get(f"{cell}_d136"),
                             dilution=dict(neural_fraction_d136={tt: INP[f"{tt}_{cell}"]["d136"]["var_fractions"]["neural"] for tt in E.TTYPES}),
                             L2_secondary=dict(COOP=L2[f"COOP_{cell}"]["label"], CMPN=L2[f"CMPN_{cell}"]["label"]),
                             statement="matched on selected input summaries only (not FC-equivalent worlds); CMPN is one sign-flip construction, not a "
                                       "faithful competition world; observation-level sources; Cn not run (failure of the tested parameterisation)")
    res = dict(prereg=pre, freeze=fsha, admitted=sum(len(v) for v in rows.values()), rejected=rej,
               real_reference=dict(source=tg["source"], source_sha=tg["source_sha"], reference=real),
               calibration={c: {k: v for k, v in cal.items() if not k.endswith("trace")} for c, cal in man["calibration"].items()},
               input_reference=ref_in, input=INP, G=G, G042=G042, pair_input_diff=PAIRDIFF, cells=out, L1=L1, L2=L2, L3=L3, L3_paired_control=L3p,
               reading=reading, note="descriptive; pointwise unadjusted; constructed setting; no attribution to the drive")
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float)
    f3 = lambda d: "%.3f [%.3f, %.3f]" % (d["theta"], d["lo"], d["hi"])
    for key, r in out.items():
        print(key, "gs %.3f" % r["input"]["gs"] if r["input"] else "-", "| FP", f3(r["fp"]), "| P2", f3(r["p2"]), "share<0 %.2f" % r["p2_negative_share"],
              "| E_bn_fc", f3(r["E_bn_fc"]), "| rec", r["recovery"], flush=True)
    print("G", {k: v["label"] for k, v in G.items()}, "| G042", {k: v["label"] for k, v in G042.items()}, flush=True)
    print("INPUT", {k: {d: (round(v["realised_gs"], 3), {q: round(x, 3) for q, x in v["var_fractions"].items()}) for d, v in dd.items()} for k, dd in INP.items()}, flush=True)
    print("PAIRDIFF", PAIRDIFF, flush=True); print("L1", {k: v["label"] for k, v in L1.items()}, flush=True)
    print("L2", {k: (v["label"], v.get("n_satisfied")) for k, v in L2.items()}, flush=True)
    print("L3", L3, flush=True); print("L3p", L3p, flush=True); print("READING", reading, flush=True)


if __name__ == "__main__":
    main()
