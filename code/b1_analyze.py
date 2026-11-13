"""Participant-mask analysis (accS; Sections 2.4.1, 3.2; Supplementary Note 4): admits the accS fits of the 95 non-example
participants, evaluates FP(accS) - FP(acc) and FP(accS) - FP(mot) against the prespecified criteria and applies the frozen readout
law without refitting.
Usage: python b1_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

P33 = DATA_ROOT + "results/p33/"; PREREG = P33 + "b1_analyze_prereg.json"; EQ = 0.03; EQ2 = 0.015; MIN_PAIRS = 80; B = 2000
DEMO = {"100307", "100408", "101107"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def admit(records_dir, exp, prov_fn, skip=frozenset()):
    """{(tag, model): (record, C)} for admitted fits; rejection counts by reason."""
    rej = dict(malformed=0, unexpected=0, field=0, state=0, prov=0, dup=0, matrix=0); seen = set(); out = {}
    for p in sorted(glob.glob(records_dir + "*.json")):
        try:
            r = json.load(open(p)); assert isinstance(r, dict)
        except Exception:
            rej["malformed"] += 1; continue
        if r.get("key") in skip:
            continue                                                   # a job outside this comparison (e.g. example participants)
        j = exp.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p):
            rej["unexpected"] += 1; continue
        if any(r.get(k) != v for k, v in j.items()):
            rej["field"] += 1; continue
        if r.get("state") != "done":
            rej["state"] += 1; continue
        if r.get("prov") != prov_fn(j):
            rej["prov"] += 1; continue
        if j["key"] in seen:
            rej["dup"] += 1; continue
        seen.add(j["key"])
        try:
            C = np.load(j["file"].replace(".json", "_C.npy"))
        except Exception:
            C = None
        if C is None or C.shape != (100, 100) or not np.all(np.isfinite(C)) or sha_arr(C) != r.get("C_sha"):
            rej["matrix"] += 1; continue
        out[(j["tag"], j["model"])] = (r, C)
    return out, dict(expected=len(exp), admitted=len(out), rejected=rej)


def fpm(C, SC):
    m = (SC != 0) & ~np.eye(100, dtype=bool); w = C[m]; return float(np.mean(w < 0)), float(np.mean(w < -0.05 * np.abs(w).max()))


def side_rows(fits, SC, inp_fn):
    import e12a_validate as V
    rows = {}
    for (tag, model), (r, C) in fits.items():
        if model != "signed" or tag in DEMO:
            continue
        fp, fp5 = fpm(C, SC); nf, gs = V.inp_measures(inp_fn(tag)); x = dict(FP=fp, FP5=fp5, nf=nf, gs=gs, P2=np.nan)
        if (tag, "coop") in fits:
            x["P2"] = r["mse_ho"] - fits[(tag, "coop")][0]["mse_ho"]
        rows[tag] = x
    return rows


def ppc_side(cond):
    import luppi_r4e_ppc as W
    X = W.X; W.point(cond); csha, cfr = X.verify_freeze("e5"); out = X.OUT
    cexp = {j["key"]: j for j in X.job_list("e5") if j["tag"] not in DEMO}
    fits, den = admit(out + "jobs_e5/", cexp, lambda j: X.job_prov(j, csha, cfr), skip={j["key"] for j in X.job_list("e5") if j["tag"] in DEMO})
    return side_rows(fits, np.load(out + "e5/SC.npy"), lambda t: np.load(out + f"e5/{t}_REST1_input.npy")), den


def load_all():
    import b1_run as R1, luppi_r4e_ppc as W
    fsha, fr = R1.verify_freeze(); exp = {j["key"]: j for j in R1.job_list()}
    b1fits, b1den = admit(R1.OUT + "jobs/", exp, lambda j: R1.job_prov(j, fsha, fr)); SC = np.load(R1.OUT + "e5/SC.npy")
    acc = side_rows(b1fits, SC, lambda t: np.load(R1.OUT + f"e5/{t}_REST1_input.npy")); sides = dict(accS=acc); dens = dict(accS=b1den)
    for cond in ("acc", "mot"):
        sides[cond], dens[cond] = ppc_side(cond)
    sim = {}
    for t in set(sides["accS"]) & set(sides["acc"]):
        a = np.load(R1.OUT + f"e5/{t}_REST1_input.npy"); b = np.load(P33 + f"r4e_ppc_acc/e5/{t}_REST1_input.npy")
        sim[t] = float(np.median([np.corrcoef(a[i], b[i])[0, 1] for i in range(100)]))
    return sides, dens, sim, fsha


def e_status(ci, bound):
    L, U = ci
    if -bound < L and U < bound:
        return "equivalent"
    if L > bound or U < -bound:
        return "outside"
    return "not established"


def boot(v, rng):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < MIN_PAIRS:
        return dict(label="not analysable", n=int(len(v)))
    bs = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(B)]
    return dict(n=int(len(v)), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


def compute(sides, sim, rng, G=None):
    s, a, m = sides["accS"], sides["acc"], sides["mot"]; t = sorted(set(s) & set(a) & set(m))
    res = dict(n_pairs=len(t))
    res["FP_accS"] = boot([s[k]["FP"] for k in t], rng); res["FP_acc"] = boot([a[k]["FP"] for k in t], rng)
    res["dFP_acc"] = boot([s[k]["FP"] - a[k]["FP"] for k in t], rng); res["dFP_mot"] = boot([s[k]["FP"] - m[k]["FP"] for k in t], rng)
    for key in ("FP5", "P2", "nf", "gs"):
        res[f"{key}_accS"] = boot([s[k][key] for k in t], rng); res[f"d{key}_acc"] = boot([s[k][key] - a[k][key] for k in t], rng)
    res["P2_share_neg_accS"] = float(np.mean([s[k]["P2"] < 0 for k in t if np.isfinite(s[k]["P2"])])) if t else np.nan
    res["P2_share_neg_acc"] = float(np.mean([a[k]["P2"] < 0 for k in t if np.isfinite(a[k]["P2"])])) if t else np.nan
    res["input_similarity"] = boot([sim[k] for k in t if k in sim], rng)
    if G is not None:
        import e12a_g_freeze as GF
        tr = G["train_range"]; sup = [k for k in t if tr["nf"][0] <= s[k]["nf"] <= tr["nf"][1] and tr["gs"][0] <= s[k]["gs"] <= tr["gs"][1]]
        if sup:
            e = np.array([s[k]["FP"] for k in sup]) - GF.predict(G, "g", "fp", [s[k]["nf"] for k in sup], [s[k]["gs"] for k in sup])
            res["readout_g"] = dict(n_supported=len(sup), mae=float(np.mean(np.abs(e))), median_err=float(np.median(e)))
    E = "not analysable" if "median" not in res["dFP_acc"] else e_status(res["dFP_acc"]["ci95"], EQ)
    E2 = "not analysable" if "median" not in res["dFP_acc"] else e_status(res["dFP_acc"]["ci95"], EQ2)
    T = ("not analysable" if "median" not in res["FP_accS"] or "median" not in res["dFP_mot"] else
         bool(res["FP_accS"]["ci95"][0] > 0.15 and res["dFP_mot"]["ci95"][0] > 0.10))
    reading = (("with subject-specific anatomical tissue-component regression (accS), the median paired FP differs from the historical acc by less than "
                "0.03 and the high-FP operator effect is reproduced (whole-pipeline change; the historical acc keeps its qualifier)") if E == "equivalent" and T is True else
               "equivalence within 0.03 was not established; the prespecified high-FP operator-effect criterion was met" if E == "not established" and T is True else
               "the median paired FP differs from acc by more than 0.03; the prespecified high-FP operator-effect criterion was met" if E == "outside" and T is True else
               "report as observed")
    res.update(B1_E=E, B1_E_0015=E2, B1_T=T, reading=reading)
    return res


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("b1_analyze.py differs from the pre-registration")
    sides, dens, sim, fsha = load_all(); G = json.load(open(P33 + "e12a_g.json"))
    import b1_run as R1
    res = compute(sides, sim, np.random.default_rng(150), G); res.update(prereg=pre, freeze=fsha, denominators=dens)
    fq = json.load(open(R1.MAN))["frequency_vs_acc"]; res["frequency_vs_acc"] = {k: v for k, v in fq.items() if k != "per_region_diff_hz"}
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float); show(res)


def show(res):
    for k, v in res.items():
        if isinstance(v, dict) and "median" in v:
            print(k, round(v["median"], 4), [round(c, 4) for c in v["ci95"]], "n", v["n"], flush=True)
    print("readout", res.get("readout_g"), "| P2 share<0 accS", res.get("P2_share_neg_accS"), "acc", res.get("P2_share_neg_acc"), flush=True)
    print("B1-E", res["B1_E"], "| B1-E(0.015)", res["B1_E_0015"], "| B1-T", res["B1_T"], "| READING", res["reading"], flush=True)


if __name__ == "__main__":
    main()
