"""Validation of the frozen readout law on V1 (Sections 2.5.1, 3.3): MAE, median residual and per-operator residuals against the
prespecified tolerance 0.023, with participant-cluster bootstrap CIs.
Usage: python e12a_validate.py SET OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

P33 = DATA_ROOT + "results/p33/"; G = P33 + "e12a_g.json"; PREREG = P33 + "e12a_validate_prereg.json"; MARGIN = 0.023
DEMO = {"100307", "100408", "101107"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def inp_measures(x):
    import luppi_hopf as L
    F = np.corrcoef(L.prep_emp(x)); iu = np.triu_indices(len(F), 1)
    return float((F[iu] < 0).mean()), float(np.var(x.mean(0)) / np.mean(x.var(1)))


def rows_v1():
    import e12a_rl as V, luppi_r4e_lib as Lib
    fsha, fr = V.verify_freeze(); exp = {j["key"]: j for j in V.job_list()}; R = {}; rej = dict(unexpected=0, not_done=0, wrong_prov=0, nonfinite=0, bad_matrix=0)
    for p in glob.glob(V.OUT + "jobs/*.json"):
        r = json.load(open(p)); j = exp.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p) or any(r.get(k) != v for k, v in j.items()):
            rej["unexpected"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != V.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if not np.isfinite(r.get("mse_ho", np.nan)):
            rej["nonfinite"] += 1; continue
        R.setdefault((j["cond"], j["tag"]), {})[j["model"]] = (j, r)
    SC = np.load(V.OUT + "inp/SC.npy"); m = Lib.support(SC); out = []
    for (c, tag), d in R.items():
        if tag in DEMO or "signed" not in d or "coop" not in d:
            continue
        j, r = d["signed"]; C = np.load(j["file"].replace(".json", "_C.npy"))
        if C.shape != (100, 100) or not np.all(np.isfinite(C)) or V.sha_arr(C) != r["C_sha"]:
            rej["bad_matrix"] += 1; continue
        nf, gs = inp_measures(np.load(V.job_files(j)[2]))
        out.append(dict(unit=tag, group=c, fp=float((C[m] < 0).mean()), p2=float(r["mse_ho"] - d["coop"][1]["mse_ho"]), nf=nf, gs=gs))
    return out, dict(freeze=fsha, rejected=rej, conditions=list(V.CONDS))


def rows_v3():
    """Known truths: frozen Simulation 1 and Simulation 2 jobs (input measures recomputed from the frozen target inputs)."""
    import luppi_r4e_e10c as E10, luppi_r4e_e11 as E11, luppi_r4e_lib as Lib
    out = []; info = {}
    for mod, key in ((E10, "dose"), (E11, "cond")):
        fsha, fr = mod.verify_freeze(); info[mod.__name__] = fsha; exp = {j["key"]: j for j in mod.job_list()}; R = {}
        for p in glob.glob(mod.OUT + "jobs/*.json"):
            r = json.load(open(p)); j = exp.get(r.get("key"))
            if j is not None and r.get("state") == "done" and r.get("prov") == mod.job_prov(j, fsha, fr):
                R.setdefault((j["ttype"], j[key], j["tag"]), {})[j["model"]] = (j, r)
        for (tt, q, tag), d in R.items():
            if "signed" in d and "coop" in d:
                j, r = d["signed"]; s = int(tag.split("_")[1]); _, _, k, _ = mod.truth_c(s, tt); SC = Lib.demo_subject(k)[1]; m = Lib.support(SC)
                C = np.load(j["file"].replace(".json", "_C.npy")); nf, gs = inp_measures(Lib.bandpass(np.load(mod.job_files(j)[2])))
                out.append(dict(unit=f"{mod.__name__}:{tag}", group=f"{mod.__name__}:{tt}:{q}", ttype=tt, fp=float((C[m] < 0).mean()),
                                p2=float(r["mse_ho"] - d["coop"][1]["mse_ho"]), nf=nf, gs=gs))
    return out, info


def evaluate(rows, Gd, labelled, conditions=None):
    import e12a_g_freeze as GF
    rng = np.random.default_rng(134); tr = Gd["train_range"]
    sup = [r for r in rows if tr["nf"][0] <= r["nf"] <= tr["nf"][1] and tr["gs"][0] <= r["gs"] <= tr["gs"][1]]
    res = dict(n_rows=len(rows), n_supported=len(sup), excluded=len(rows) - len(sup),
               supported_by_group={g: sum(r["group"] == g for r in sup) for g in sorted({r["group"] for r in rows})})
    if labelled and (conditions is None or any(res["supported_by_group"].get(c, 0) < 90 for c in conditions)):
        res["label"] = "not analysable (completeness / support)"; labelled = False
    if len(sup) < 20:
        res["label"] = "not analysable"; return res
    nf = [r["nf"] for r in sup]; gs = [r["gs"] for r in sup]; units = sorted({r["unit"] for r in sup}); idx = {u: [i for i, r in enumerate(sup) if r["unit"] == u] for u in units}
    boots = [np.concatenate([idx[u] for u in rng.choice(units, len(units))]) for _ in range(2000)]
    for model in GF.MODELS:
        res[model] = {}
        for key in ("fp", "p2", "neg"):
            y = np.array([r[key] if key != "neg" else float(r["p2"] < 0) for r in sup]); e = y - GF.predict(Gd, model, key, nf, gs)
            B = np.array([(np.mean(np.abs(e[ii])), np.median(e[ii])) for ii in boots])
            rec = dict(mae=float(np.mean(np.abs(e))), mae_ci95=[float(np.quantile(B[:, 0], 0.025)), float(np.quantile(B[:, 0], 0.975))],
                       median_residual=float(np.median(e)), median_ci95=[float(np.quantile(B[:, 1], 0.025)), float(np.quantile(B[:, 1], 0.975))],
                       r2=float(1 - np.mean(e ** 2) / np.var(y)) if np.var(y) > 0 else np.nan,
                       by_group={gname: dict(n=int(sum(r["group"] == gname for r in sup)), median_residual=float(np.median(e[[i for i, r in enumerate(sup) if r["group"] == gname]])))
                                 for gname in sorted({r["group"] for r in sup})})
            if "ttype" in sup[0]:
                rec["by_type"] = {t: float(np.median(e[[i for i, r in enumerate(sup) if r["ttype"] == t]])) for t in sorted({r["ttype"] for r in sup})}
            res[model][key] = rec; res[model].setdefault("_abs_err", {})[key] = np.abs(e)
    for key in ("fp", "p2", "neg"):
        d = res["g"]["_abs_err"][key] - res["g_nf"]["_abs_err"][key]; B = [float(np.mean(d[ii])) for ii in boots]
        res[f"mae_g_minus_gnf_{key}"] = dict(diff=float(np.mean(d)), ci95=[float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))])
    for model in GF.MODELS:
        res[model].pop("_abs_err")
    if labelled:
        f = res["g"]["fp"]; k3 = all(abs(v["median_residual"]) <= MARGIN for v in f["by_group"].values())
        crit = dict(K1=bool(f["mae_ci95"][1] <= MARGIN), K2=bool(-MARGIN <= f["median_ci95"][0] and f["median_ci95"][1] <= MARGIN), K3=bool(k3))
        res["criteria"] = crit
        res["label"] = "readout law validated on v1 (transport across runs within the fixed eight-operator panel)" if all(crit.values()) else "not validated"
    return res


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("e12a_validate.py differs from the pre-registered version")
    Gd = json.load(open(G))
    if Gd["source_sha"] != sha(Gd["source"]) or sha(G) != pre["g_sha256"]:
        raise SystemExit("frozen predictors or their source changed")
    which = sys.argv[1]
    if which == "v1":
        rows, info = rows_v1(); res = evaluate(rows, Gd, True, info["conditions"])
    elif which == "v3":
        rows, info = rows_v3(); res = evaluate(rows, Gd, False)
    else:
        raise SystemExit("set must be v1 or v3")
    out = dict(set=which, prereg=pre, g_sha=sha(G), info=info, margin=MARGIN, result=res)
    json.dump(out, open(sys.argv[2], "x"), indent=1, default=float)
    print(which, {k: v for k, v in res.items() if k not in ("g", "g_nf")}, flush=True)
    for model in ("g", "g_nf"):
        for k in ("fp", "p2", "neg"):
            if model in res:
                r = res[model][k]
                print(" ", model, k, "MAE %.4f [%.4f, %.4f]" % (r["mae"], *r["mae_ci95"]), "median res %.4f [%.4f, %.4f]" % (r["median_residual"], *r["median_ci95"]),
                      "R2 %.3f" % r["r2"], "| by group", {gg: round(v["median_residual"], 4) for gg, v in r["by_group"].items()}, "| by type", r.get("by_type"), flush=True)


if __name__ == "__main__":
    main()
