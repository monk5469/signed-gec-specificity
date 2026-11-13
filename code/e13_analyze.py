"""Hierarchy analysis (Sections 2.7.2, 3.6; Supplementary Note 9): the hierarchy-gap coefficient E_h and its descriptive variants
for real and simulated fits, with the prespecified hypotheses H1 and H3.
Usage: python e13_analyze.py manifest (writes the fit manifest once, before registration)
       python e13_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

P33 = DATA_ROOT + "results/p33/"; HIER = P33 + "e13_hier.json"; PREREG = P33 + "e13_analyze_prereg.json"; FITMAN = P33 + "e13_fit_manifest.json"
DEMO = {"100307", "100408", "101107"}; MIN_N = 15; MIN_STRATUM = 5; MIN_CNT = 10; KEYS = ("E_h", "E_h_nd", "E_h_fc", "E_h_inp", "AUC_h", "fp")
REAL = ("acc", "acc1", "acc3", "gsr", "mot", "vol0", "null10")
SIMS = (("e10c_COOP_d136", "luppi_r4e_e10c", "dose", "COOP", "d136", None), ("e11_COOP_N49d136", "luppi_r4e_e11", "cond", "COOP", "N49-d136", None),
        ("e12c_COOP_N49d136_S1", "e12c_run", "cond", "COOP", "N49-d136", "S1"), ("e10c_CMPN_d136", "luppi_r4e_e10c", "dose", "CMPN", "d136", None),
        ("e11_CMPN_N49d136", "luppi_r4e_e11", "cond", "CMPN", "N49-d136", None), ("e11_CMPH_N49d136", "luppi_r4e_e11", "cond", "CMPH", "N49-d136", None))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def irls(y, X, j=1, ridge=1e-6, iters=100):
    """(coefficient, status); status 'ok', 'insufficient' (< MIN_CNT of either outcome) or 'nonconverged'."""
    beta = np.zeros(X.shape[1]); yf = y.astype(float)
    if yf.sum() < MIN_CNT or len(yf) - yf.sum() < MIN_CNT:
        return np.nan, "insufficient"
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))); Wd = p * (1 - p) + 1e-12
        H = X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1]); step = np.linalg.solve(H, X.T @ (yf - p) - ridge * beta); beta += step
        if np.max(np.abs(step)) < 1e-8:
            return (float(beta[j]), "ok") if np.all(np.isfinite(beta)) else (np.nan, "nonconverged")
    return np.nan, "nonconverged"


def quart(x):
    q = np.searchsorted(np.quantile(x, [0.25, 0.5, 0.75]), x); return [(q == k).astype(float) for k in (1, 2, 3)]


def metrics(C, SC, inp, h, dist):
    import luppi_hopf as L, luppi_r4e_lib as Lib
    from scipy.stats import rankdata
    m = Lib.support(SC); r = rankdata(h); D = np.abs(r[:, None] - r[None, :]) / 99.0; d = D[m]; z = (d - d.mean()) / d.std()
    f = np.corrcoef(L.prep_emp(inp))[m]; one = np.ones(m.sum())
    Xnd = np.column_stack([one, z] + quart(SC[m])); X = np.column_stack([Xnd] + quart(dist[m])); Xf = np.column_stack([X] + quart(f))
    y = C[m] < 0; yi = f < 0; a, b = d[y], d[~y]; out = {}; st = {}
    for k, (yy, XX) in dict(E_h=(y, X), E_h_nd=(y, Xnd), E_h_fc=(y, Xf), E_h_inp=(yi, X)).items():
        out[k], st[k] = irls(yy, XX)
    if a.size < MIN_CNT or b.size < MIN_CNT:
        out["AUC_h"], st["AUC_h"] = np.nan, "insufficient"
    else:
        out["AUC_h"] = float((rankdata(np.r_[a, b])[:a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size)); st["AUC_h"] = "ok"
    out["fp"], st["fp"] = float(y.mean()), "ok"
    return dict(val=out, status=st)


# ---------------------------------------------------------------- set definitions (expected jobs, admission context)
def set_context(name):
    """(expected jobs {key: job}, provenance function, records directory, input loader, SC loader, stratum function)."""
    import importlib, luppi_r4e_lib as Lib
    if name.startswith("real_"):
        import luppi_r4e_ppc as W
        X = W.X; W.point(name[5:]); fsha, fr = X.verify_freeze("e5"); out = X.OUT
        exp = {j["key"]: j for j in X.job_list("e5") if j["model"] == "signed" and j["tag"] not in DEMO}
        SC = np.load(out + "e5/SC.npy")
        return dict(exp=exp, prov=lambda j: X.job_prov(j, fsha, fr), dir=out + "jobs_e5/", freeze=fsha,
                    inp=lambda j: np.load(out + f"e5/{j['tag']}_REST1_input.npy"), sc=lambda j: SC, stratum=lambda j: None,
                    all_keys={j["key"] for j in X.job_list("e5")})
    _, modname, key, tt, q, st = [s for s in SIMS if s[0] == name][0]
    mod = importlib.import_module(modname); fsha, fr = mod.verify_freeze(); allj = mod.job_list()
    exp = {j["key"]: j for j in allj if j["model"] == "signed" and j.get("ttype") == tt and j.get(key) == q and (st is None or j.get("setting") == st)}
    SCg = np.load(mod.OUT + "inp/SC_group.npy") if st == "S1" else None
    return dict(exp=exp, prov=lambda j: mod.job_prov(j, fsha, fr), dir=mod.OUT + "jobs/", freeze=fsha,
                inp=lambda j: Lib.bandpass(np.load(mod.job_files(j)[0] if st else mod.job_files(j)[2])),
                sc=lambda j: SCg if st == "S1" else Lib.demo_subject((int(j["tag"].split("_")[1]) - 1) % 3)[1],
                stratum=lambda j: (int(j["tag"].split("_")[1]) - 1) % 3, all_keys={j["key"] for j in allj})


def set_names():
    return [f"real_{c}" for c in REAL] + [s[0] for s in SIMS]


def stage_manifest():
    """Create-once: sha256 of every expected job record and fitted matrix of every set, before the registration record."""
    if os.path.exists(FITMAN):
        raise SystemExit("fit manifest already exists")
    json.dump(build_manifest(), open(FITMAN, "x"), indent=1)


def build_manifest():
    man = {}
    for name in set_names():
        ctx = set_context(name); ent = {}
        for k, j in ctx["exp"].items():
            cp = j["file"].replace(".json", "_C.npy")
            ent[k] = dict(record=j["file"], record_sha=sha(j["file"]) if os.path.exists(j["file"]) else None, C=cp, C_sha=sha(cp) if os.path.exists(cp) else None)
        man[name] = dict(freeze=ctx["freeze"], expected=len(ent), jobs=ent)
        print(name, "expected", len(ent), "records", sum(e["record_sha"] is not None for e in ent.values()), flush=True)
    return dict(sets=man)


def load_set(name, H, dist, fitman):
    ctx = set_context(name); exp = ctx["exp"]; mf = fitman["sets"][name]["jobs"]
    rej = dict(malformed=0, unexpected=0, field_mismatch=0, not_done=0, wrong_prov=0, duplicate=0, bad_matrix=0, manifest_mismatch=0)
    seen = set(); out = {}; found = 0
    for p in sorted(glob.glob(ctx["dir"] + "*.json")):
        try:
            r = json.load(open(p)); assert isinstance(r, dict)
        except Exception:
            rej["malformed"] += 1; continue
        if r.get("key") in ctx["all_keys"] and r.get("key") not in exp:
            continue                                                   # a job of another set / model: not part of this set
        j = exp.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        found += 1
        if any(r.get(k) != v for k, v in j.items()):
            rej["field_mismatch"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != ctx["prov"](j):
            rej["wrong_prov"] += 1; continue
        if j["key"] in seen:
            rej["duplicate"] += 1; continue
        seen.add(j["key"]); cp = j["file"].replace(".json", "_C.npy")
        try:
            C = np.load(cp)
        except Exception:
            C = None
        if C is None or C.shape != (100, 100) or not np.all(np.isfinite(C)) or sha_arr(C) != r.get("C_sha"):
            rej["bad_matrix"] += 1; continue
        e = mf.get(j["key"])
        if e is None or e["record_sha"] != sha(j["file"]) or e["C_sha"] != sha(cp):
            rej["manifest_mismatch"] += 1; continue
        SC = ctx["sc"](j); inp = ctx["inp"](j)
        out[j["tag"]] = dict(stratum=ctx["stratum"](j), **{k: metrics(C, SC, inp, H[k], dist) for k in H})
    den = dict(expected=len(exp), records_found=found, missing=len(exp) - found, rejected=rej, admitted=len(out), freeze=ctx["freeze"])
    return out, den


def summarise(rows, rng, stratified):
    res = {}
    for mapk in ("A2", "A1"):
        res[mapk] = {}
        for key in KEYS:
            sts = [r[mapk]["status"][key] for r in rows.values()]
            fin = [(r["stratum"], r[mapk]["val"][key]) for r in rows.values() if r[mapk]["status"][key] == "ok"]
            v = np.array([x for _, x in fin]); cnt = dict(admissible=len(v), insufficient=sts.count("insufficient"), nonconverged=sts.count("nonconverged"))
            groups = {s: np.array([x for t, x in fin if t == s]) for s in (0, 1, 2)} if stratified else None
            if len(v) < MIN_N or (stratified and min(len(g) for g in groups.values()) < MIN_STRATUM):
                res[mapk][key] = dict(label="not analysable", **cnt); continue
            if stratified:
                B = [np.median(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups.values()])) for _ in range(2000)]
            else:
                B = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
            res[mapk][key] = dict(**cnt, median=float(np.median(v)), ci95=[float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))])
            if stratified:
                res[mapk][key]["per_stratum"] = {str(s): dict(n=len(g), median=float(np.median(g))) for s, g in groups.items()}
    return res


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"] or sha(HIER) != pre["hier_sha256"] or sha(FITMAN) != pre["fit_manifest_sha256"]:
        raise SystemExit("e13_analyze.py, the hierarchy file or the fit manifest differs from the pre-registration")
    hier = json.load(open(HIER)); H = dict(A2=np.array(hier["h_A2"]), A1=np.array(hier["h_A1"])); dist = np.array(hier["dist_mm"]); fitman = json.load(open(FITMAN))
    rng = np.random.default_rng(13); sets = {}; den = {}; summ = {}
    for name in set_names():
        rows, d = load_set(name, H, dist, fitman); sets[name] = rows; den[name] = d; summ[name] = summarise(rows, rng, not name.startswith("real_"))
    acc = sets["real_acc"]; s = summ["real_acc"]["A2"]
    H1 = "not analysable" if "median" not in s["E_h"] else bool(s["E_h"]["ci95"][0] > 0)
    t = [k for k in acc if acc[k]["A2"]["status"]["E_h"] == "ok" and acc[k]["A2"]["status"]["E_h_fc"] == "ok"]
    if len(t) >= MIN_N:
        dd = np.array([acc[k]["A2"]["val"]["E_h"] - acc[k]["A2"]["val"]["E_h_fc"] for k in t]); B = [np.median(dd[rng.integers(0, len(t), len(t))]) for _ in range(2000)]
        H2 = dict(descriptive=True, E_h_fc=s["E_h_fc"], paired_diff=dict(n=len(t), median=float(np.median(dd)), ci95=[float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))]))
    else:
        H2 = dict(descriptive=True, label="not analysable", n=len(t))
    h3 = {}
    for k in ("e10c_COOP_d136", "e11_COOP_N49d136"):
        x = summ[k]["A2"]["E_h"]
        h3[k] = "not analysable" if "median" not in x else bool(x["ci95"][0] > 0 and all(v["median"] > 0 for v in x["per_stratum"].values()))
    H3 = "not analysable" if any(v == "not analysable" for v in h3.values()) else dict(cells=h3, label=all(h3.values()))
    ok = H1 is True and isinstance(H3, dict) and H3["label"]
    reading = ("conditional on the covariates, atlas and SCs, a positive hierarchy-gap coefficient of negative fitted weights arises both in the real "
               "fits and in cooperative constructions without negative coupling; it is therefore not, by itself, specific evidence of competitive "
               "interactions" if ok else "no joint reading; report H1-H3 as observed")
    res = dict(prereg=pre, hier_sha=sha(HIER), denominators=den, summary=summ, H1=H1, H2=H2, H3=H3, reading=reading)
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float)
    for k, v in summ.items():
        print(k, den[k]["expected"], den[k]["admitted"], {m: {kk: (round(x["median"], 3), [round(c, 3) for c in x["ci95"]]) if "median" in x else x for kk, x in v[m].items()} for m in v}, flush=True)
    print("H1", H1, "| H2", H2, "| H3", H3, "| READING", reading, flush=True)


if __name__ == "__main__":
    stage_manifest() if sys.argv[1] == "manifest" else main()
