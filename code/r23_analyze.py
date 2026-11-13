"""Participant-specific SC and 232-region analysis (Sections 2.4.2, 3.7.1, 3.7.2; Supplementary Note 12): the prespecified
robustness requirements, equivalence with the reference-SC fits, group-level comparisons and frozen-law predictions.
Usage: python r23_analyze.py OUT_JSON"""
import sys, json
import numpy as np
from paths import DATA_ROOT
import r23_run as R

B = 2000; MIN_COMPLETE = 80


def admit():
    fsha, fr = R.verify_freeze(); J = R.job_list(); rows = {}; rej = {}
    for j in J:
        why = None
        try:
            r = json.load(open(j["file"]))
        except Exception:
            why = "missing"
        else:
            why = ("field_mismatch" if any(r.get(q) != v for q, v in j.items()) else "not_done" if r.get("state") != "done" else
                   "wrong_prov" if r.get("prov") != R.job_prov(j, fsha, fr) else None)
            if why is None:
                try:
                    C = np.load(j["file"].replace(".json", "_C.npy"))
                    if not np.all(np.isfinite(C)) or R.sha_arr(C) != r.get("C_sha"):
                        why = "bad_matrix"
                except Exception:
                    why = "bad_matrix"
        if why:
            k = f"{j['res']}|{j['cond']}|{j['arm']}"; rej.setdefault(k, {}).setdefault(why, 0); rej[k][why] += 1; continue
        rows[(j["res"], j["cond"], j["arm"], j["tag"])] = (j, r, C)
    return rows, rej, fsha


def boot(vals, rng):
    v = np.array([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    bs = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(B)]
    return dict(n=int(v.size), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def inp_stats(j):
    import luppi_hopf as L
    x = np.load(R.job_files(j)[0]); F = np.corrcoef(L.prep_emp(x)); iu = np.triu_indices(len(F), 1)
    return float((F[iu] < 0).mean()), float(np.var(x.mean(0)) / np.mean(x.var(1)))


def main():
    from scipy.stats import spearmanr
    import e12a_g_freeze as GF, r5_groupfc as R5
    R.check_rules(); rows, rej, fsha = admit(); P = json.load(open(R.PANEL)); subs = P["subjects"]; rng = np.random.default_rng(20261023)
    FP, FPx, P2b, P2a, DRb, DRa, NFGS = {}, {}, {}, {}, {}, {}, {}
    ref_sup = (np.load(DATA_ROOT + "results/p33/r4e_ppc_acc/e5/SC.npy") != 0) & ~np.eye(100, dtype=bool)
    for (res, c, a, s), (j, r, C) in rows.items():
        if a != "S-ind":
            continue
        SC = np.load(R.job_files(j)[3]); m = (SC != 0) & ~np.eye(res, dtype=bool); FP[(res, c, s)] = float((C[m] < 0).mean())
        if res == 100:
            FPx[(c, s)] = float((C[m & ref_sup] < 0).mean())
        NFGS[(res, c, s)] = inp_stats(j)
        for arm, P2, DR in (("C-ind", P2b, DRb), ("C-grp", P2a, DRa)):
            o = rows.get((res, c, arm, s))
            if o:
                P2[(res, c, s)] = float(r["mse_ho"] - o[1]["mse_ho"]); DR[(res, c, s)] = float(r["fc_corr"] - o[1]["fc_corr"])
    out = dict(freeze=fsha, panel=dict(n=len(subs), gsc=P["gsc_selected"], label=P["gsc_label"], density=P["density"]), rejected=rej)
    pdiff = lambda D, res, c1, c2: boot([D[(res, c1, s)] - D[(res, c2, s)] for s in subs if (res, c1, s) in D and (res, c2, s) in D], rng)
    cell = lambda D, res, c: boot([D[(res, c, s)] for s in subs if (res, c, s) in D], rng)
    for res in (100, 232):
        comp = sum(all((res, c, a, s) in rows for c in ("mot", "acc") for a in ("S-ind", "C-ind")) for s in subs); ok = comp >= MIN_COMPLETE
        conds = R.R2_FULL + (R.R2_SONLY if res == 100 else ())
        o = dict(complete_participants=comp, analysable=ok, FP={c: cell(FP, res, c) for c in conds},
                 acc_minus_mot=pdiff(FP, res, "acc", "mot"), acc_minus_null10=pdiff(FP, res, "acc", "null10"), gsr_minus_acc=pdiff(FP, res, "gsr", "acc"),
                 P2_design_b={c: cell(P2b, res, c) for c in R.R2_FULL}, P2_design_a={c: cell(P2a, res, c) for c in R.R2_FULL},
                 P2b_acc_minus_mot=pdiff(P2b, res, "acc", "mot"), P2a_acc_minus_mot=pdiff(P2a, res, "acc", "mot"),
                 dr_design_b={c: cell(DRb, res, c) for c in R.R2_FULL}, dr_design_a={c: cell(DRa, res, c) for c in R.R2_FULL})
        T, Pc = o["acc_minus_mot"], o["P2b_acc_minus_mot"]
        if res == 100:
            rho = []
            for s in subs:
                v = [FP.get((100, c, s)) for c in ("mot", "acc1", "acc3", "acc")]
                if all(x is not None for x in v):
                    rho.append(float(spearmanr([0, 1, 3, 5], v)[0]))
            o["rank_trend_rho"] = boot(rho, rng)
            o["R2-T"] = bool(T["ci95"][0] > 0.10 and o["FP"]["acc"]["ci95"][0] > 0.15) if ok else "not analysable"
            o["R2-N"] = bool(o["acc_minus_null10"]["ci95"][0] > 0.10) if ok else "not analysable"
            o["R2-K"] = bool(o["rank_trend_rho"]["ci95"][0] > 0) if ok and o["rank_trend_rho"].get("n", 0) >= MIN_COMPLETE else "not analysable"
            o["R2-P"] = bool(Pc["ci95"][1] < 0) if ok else "not analysable"
            sens = {}
            for c in conds:
                X, pairs, _, _ = R5.admitted(c); ref = {t: d["signed"][2] for t, _, d in pairs}; refsc = np.load(X.OUT + "e5/SC.npy"); mr = (refsc != 0) & ~np.eye(100, dtype=bool)
                fpref = {t: float((ref[t][mr] < 0).mean()) for t in ref}
                d = boot([FP[(100, c, s)] - fpref[s] for s in subs if (100, c, s) in FP and s in fpref], rng)
                xs = []
                for s in subs:
                    if (c, s) in FPx and s in ref:
                        mi = (np.load(R.OUT + f"scsel/{s}_sc100.npy") != 0) & mr                     # intersection of individual and reference supports
                        xs.append(FPx[(c, s)] - float((ref[s][mi] < 0).mean()))
                dx = boot(xs, rng)
                sens[c] = dict(native=d, intersection=dx, equivalent_native=bool(d.get("ci95") and -0.023 < d["ci95"][0] and d["ci95"][1] < 0.023))
            o["SC_sensitivity"] = sens
        else:
            o["R3-T"] = bool(T["ci95"][0] > 0.10) if ok else "not analysable"
            o["R3-P"] = bool(Pc["ci95"][1] < 0) if ok else "not analysable"
        G = json.load(open(DATA_ROOT + "results/p33/e12a_g.json")); err = []
        for c in conds:
            ks = [s for s in subs if (res, c, s) in FP]
            if ks:
                nf = [NFGS[(res, c, s)][0] for s in ks]; gs = [NFGS[(res, c, s)][1] for s in ks]
                err += list(np.array([FP[(res, c, s)] for s in ks]) - GF.predict(G, "g", "fp", nf, gs))
        o["readout_g_residual"] = dict(mae=float(np.mean(np.abs(err))) if err else np.nan, median=float(np.median(err)) if err else np.nan, n=len(err))
        gl = {}
        for c in R.R2_FULL:
            for arm in R.ARMS:
                ks = [s for s in subs if (res, c, arm, s) in rows and (res, c, "S-ind", s) in rows]
                if len(ks) >= MIN_COMPLETE:
                    import luppi_hopf as L
                    E = np.mean([np.corrcoef(L.prep_emp(np.load(R.job_files(rows[(res, c, "S-ind", s)][0])[0]))) for s in ks], 0)
                    S = np.mean([np.load(rows[(res, c, arm, s)][0]["file"].replace(".json", "_FCsim.npy")).astype(float) for s in ks], 0)
                    iu = np.triu_indices(res, 1); gl[f"{c}|{arm}"] = dict(pearson=float(np.corrcoef(E[iu], S[iu])[0, 1]), spearman=float(spearmanr(E[iu], S[iu])[0]), n=len(ks))
        o["group_level_fc_corr"] = gl
        out[f"R{2 if res == 100 else 3}"] = o
    json.dump(out, open(sys.argv[1], "x"), indent=1, default=float)
    r2, r3 = out["R2"], out["R3"]
    print("R2 | T", r2["R2-T"], "N", r2["R2-N"], "K", r2["R2-K"], "P", r2["R2-P"], "| acc-mot", r2["acc_minus_mot"], "| P2b acc-mot", r2["P2b_acc_minus_mot"], flush=True)
    print("R2 SC sensitivity", {c: (v["native"].get("median"), v["native"].get("ci95"), v["equivalent_native"]) for c, v in r2["SC_sensitivity"].items()}, flush=True)
    print("R3 | T", r3["R3-T"], "P", r3["R3-P"], "| acc-mot", r3["acc_minus_mot"], "| FP acc", r3["FP"]["acc"], flush=True)


if __name__ == "__main__":
    main()
