"""Exploratory zero-fit checks (Supplementary Notes 2 and 7), including the Simulation 1 comparison of COOP and CMPN at matched
input (FP ~ 1 + nf + nf^2 + type within the common nf support).
Usage: python factcheck_extra.py OUT_JSON"""
from paths import DATA_ROOT
import sys, json, glob
import numpy as np

P33 = DATA_ROOT + "results/p33/"


def fits(mod, key_ok):
    fsha, fr = mod.verify_freeze(); exp = {j["key"]: j for j in mod.job_list()}; out = []
    for p in glob.glob(mod.OUT + "jobs/*.json"):
        r = json.load(open(p)); j = exp.get(r.get("key"))
        if j is None or j["model"] != "signed" or r.get("state") != "done" or r.get("prov") != mod.job_prov(j, fsha, fr) or not key_ok(j):
            continue
        C = np.load(j["file"].replace(".json", "_C.npy"))
        if mod.sha_arr(C) == r.get("C_sha") if hasattr(mod, "sha_arr") else True:
            out.append((j, C))
    return out


def main():
    import luppi_r4e_e10c as E10, luppi_r4e_e11 as E11, luppi_r4e_lib as Lib, luppi_r4e_e8 as E8, e12a_validate as V
    E10.sha_arr = E8.sha_arr; E11.sha_arr = E8.sha_arr
    rows = []
    for j, C in fits(E10, lambda j: j["ttype"] in ("COOP", "CMPN") and j["cell"] == "Co"):
        SC = Lib.demo_subject(j["stratum"])[1]; m = Lib.support(SC)
        nf, gs = V.inp_measures(Lib.bandpass(np.load(E10.job_files(j)[2])))
        rows.append(dict(tag=j["tag"], stratum=j["stratum"], ttype=j["ttype"], dose=j["dose"], fp=float((C[m] < 0).mean()), nf=nf, gs=gs))
    res = dict(n_rows=len(rows))
    res["per_dose"] = {d: {t: dict(fp=float(np.mean([r["fp"] for r in rows if r["dose"] == d and r["ttype"] == t])),
                                   nf=float(np.mean([r["nf"] for r in rows if r["dose"] == d and r["ttype"] == t]))) for t in ("COOP", "CMPN")}
                       for d in ("raw", "d284", "d189", "d136", "gsr")}
    src = [r for r in rows if r["dose"] in ("d284", "d189", "d136", "gsr")]
    lo = max(np.quantile([r["nf"] for r in src if r["ttype"] == t], 0.05) for t in ("COOP", "CMPN"))
    hi = min(np.quantile([r["nf"] for r in src if r["ttype"] == t], 0.95) for t in ("COOP", "CMPN"))
    sup = [r for r in src if lo <= r["nf"] <= hi]
    def coef(rs):
        X = np.c_[np.ones(len(rs)), [r["nf"] for r in rs], [r["nf"] ** 2 for r in rs], [r["ttype"] == "CMPN" for r in rs]].astype(float)
        return float(np.linalg.lstsq(X, np.array([r["fp"] for r in rs]), rcond=None)[0][3])
    tags = sorted({r["tag"] for r in sup}); st = {r["tag"]: r["stratum"] for r in sup}; rng = np.random.default_rng(161); B = []
    for _ in range(2000):
        pick = []
        for s in (0, 1, 2):
            ts = [t for t in tags if st[t] == s]; pick += list(rng.choice(ts, len(ts)))
        rs = [r for t in pick for r in sup if r["tag"] == t]; B.append(coef(rs))
    res["matched_input"] = dict(support=[float(lo), float(hi)], n=len(sup), cmpn_coef=coef(sup), ci95=[float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))])
    # (2) normalisation
    import luppi_r4e_ppc as W, luppi_r4e_run as RUN
    stats = []
    for cond in ("vol0", "mot", "null10", "accero", "acc1", "acc3", "acc", "gsr"):
        X = W.X; W.point(cond); fsha, fr = X.verify_freeze("e5"); exp = {j["key"]: j for j in X.job_list("e5")}; SC = np.load(X.OUT + "e5/SC.npy"); m = (SC != 0) & ~np.eye(100, dtype=bool)
        for p in glob.glob(X.OUT + "jobs_e5/*.json"):
            r = json.load(open(p)); j = exp.get(r.get("key"))
            if j is None or j["model"] != "signed" or r.get("state") != "done":
                continue
            C = np.load(j["file"].replace(".json", "_C.npy")); w = C[m]; stats.append(("PPC", float(w.max()), float(-w.min())))
    for name, mod, ok in (("E10c", E10, lambda j: True), ("E11", E11, lambda j: True)):
        for j, C in fits(mod, ok):
            SC = Lib.demo_subject(j["stratum"])[1]; w = C[Lib.support(SC)]; stats.append((name, float(w.max()), float(-w.min())))
    arr = np.array([(s[1], s[2]) for s in stats])
    res["normalisation"] = {g: dict(n=sum(s[0] == g for s in stats), maxpos_min=float(min(s[1] for s in stats if s[0] == g)), maxpos_max=float(max(s[1] for s in stats if s[0] == g)),
                                    n_neg_exceeds_maxpos=int(sum(s[2] > s[1] + 1e-12 for s in stats if s[0] == g)), max_abs_neg=float(max(s[2] for s in stats if s[0] == g)))
                            for g in ("PPC", "E10c", "E11")}
    json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
