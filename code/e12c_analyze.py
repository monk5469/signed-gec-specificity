"""Simulation 3 analysis (Sections 2.5.2, 3.3): residuals of the human-trained law g for cooperative and CMPH truths within the
frozen joint support, evaluated against the prespecified controlled-detection criteria.
Usage: python e12c_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

P33 = DATA_ROOT + "results/p33/"; PREREG = P33 + "e12c_analyze_prereg.json"; M = 0.023; MIN_N = 15


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    import e12c_run as R
    fsha, fr = R.verify_freeze(); exp = {j["key"]: j for j in R.job_list()}; out = {}; rej = dict(unexpected=0, not_done=0, wrong_prov=0, nonfinite=0, bad_matrix=0)
    for p in glob.glob(R.OUT + "jobs/*.json"):
        r = json.load(open(p)); j = exp.get(r.get("key"))
        if j is None or os.path.basename(j["file"]) != os.path.basename(p) or any(r.get(k) != v for k, v in j.items()):
            rej["unexpected"] += 1; continue
        if r.get("state") != "done":
            rej["not_done"] += 1; continue
        if r.get("prov") != R.job_prov(j, fsha, fr):
            rej["wrong_prov"] += 1; continue
        if not np.isfinite(r.get("mse_ho", np.nan)):
            rej["nonfinite"] += 1; continue
        if j["model"] == "signed":
            C = np.load(j["file"].replace(".json", "_C.npy"))
            if C.shape != (100, 100) or not np.all(np.isfinite(C)) or R.sha_arr(C) != r["C_sha"]:
                rej["bad_matrix"] += 1; continue
            r["_C"] = C
        out.setdefault((j["cell"], j["setting"], j["tag"]), {})[j["model"]] = (j, r)
    return out, rej, fsha


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("e12c_analyze.py differs from the pre-registered version")
    import e12c_run as R
    D, rej, fsha = load(); res = compute(D, rej, fsha, pre, json.load(open(R.PANEL)))
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float); report(res)


def compute(D, rej, fsha, pre, P):
    import e12c_run as R, e12a_g_freeze as GF, e12a_validate as V, luppi_r4e_lib as Lib
    Gd = json.load(open(R.G)); SCg = np.load(R.OUT + "inp/SC_group.npy")
    rows = {}
    for (cell, st, tag), d in D.items():
        if "signed" not in d:
            continue
        j, r = d["signed"]; C = r["_C"]; SC = SCg if st == "S1" else Lib.demo_subject(j["stratum"])[1]; m = Lib.support(SC)
        nf, gs = V.inp_measures(Lib.bandpass(np.load(R.job_files(j)[0]))); fp = float((C[m] < 0).mean())
        rows.setdefault((cell, st), {})[tag] = dict(stratum=j["stratum"], fp=fp, nf=nf, gs=gs, res=fp - float(GF.predict(Gd, "g", "fp", [nf], [gs])[0]),
                                                    res_nf=fp - float(GF.predict(Gd, "g_nf", "fp", [nf], [gs])[0]),
                                                    p2=float(r["mse_ho"] - d["coop"][1]["mse_ho"]) if "coop" in d else np.nan)
    rng = np.random.default_rng(1213); out = {}
    def boot_med(vals_by_tag, tags, key):
        v = np.array([vals_by_tag[t][key] for t in tags]); B = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
        return dict(n=len(v), median=float(np.median(v)), ci95=[float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))])
    for name, c in P["cells"].items():
        rec = dict(role=c["role"], eligible=c["eligible"])
        for st in ("S1", "S0"):
            rr = rows.get((name, st), {}); tags = sorted(t for t in c["supported"] if t in rr)
            if not c["eligible"] or len(tags) < MIN_N:
                rec[st] = dict(label="not analysable", n=len(tags)); continue
            rec[st] = dict(res=boot_med(rr, tags, "res"), res_nf=boot_med(rr, tags, "res_nf"), fp_median=float(np.median([rr[t]["fp"] for t in tags])),
                           nf_median=float(np.median([rr[t]["nf"] for t in tags])), gs_median=float(np.median([rr[t]["gs"] for t in tags])))
            if "coop" in c["models"]:
                rec[st]["p2"] = boot_med(rr, [t for t in tags if np.isfinite(rr[t]["p2"])], "p2")
        a, b = rows.get((name, "S1"), {}), rows.get((name, "S0"), {}); tags = sorted(t for t in c["supported"] if t in a and t in b)
        if c["eligible"] and len(tags) >= MIN_N:
            va = np.array([a[t]["res"] for t in tags]); vb = np.array([b[t]["res"] for t in tags]); B = []
            for _ in range(2000):
                ii = rng.integers(0, len(tags), len(tags)); B.append(np.median(va[ii]) - np.median(vb[ii]))
            ci = [float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))]
            rec["T3"] = dict(n=len(tags), diff=float(np.median(va) - np.median(vb)), ci95=ci, label=bool(-M <= ci[0] and ci[1] <= M))
        else:
            rec["T3"] = dict(label="not analysable")
        out[name] = rec
    within = lambda r: r.get("res") is not None and -M <= r["res"]["ci95"][0] and r["res"]["ci95"][1] <= M
    T1 = {n: ("bounded median residual bias" if within(r["S1"]) else "not bounded") for n, r in out.items() if r["role"] == "T1" and "res" in r["S1"]}
    if not T1:
        T1 = "not analysable"
    t2cells = {n: r for n, r in out.items() if r["role"] in ("T2", "T2opt") and "res" in r["S1"]}
    if "res" not in out["CMPH_N0-raw"]["S1"]:
        T2 = "not analysable"
    else:
        T2 = dict(cells=sorted(t2cells), label=all(r["S1"]["res"]["ci95"][1] < -0.05 for r in t2cells.values()))
    ctrl = out["COOP_N49-d042"]["S1"]; CTRL = "not analysable" if "res" not in ctrl else bool(within(ctrl))
    dist = None
    if "res" in ctrl and "res" in out["CMPH_N0-raw"]["S1"]:
        dist = dict(ctrl=[ctrl["nf_median"], ctrl["gs_median"]], cmph=[out["CMPH_N0-raw"]["S1"]["nf_median"], out["CMPH_N0-raw"]["S1"]["gs_median"]])
    reading = ("the readout residual flags this engineered strong competition at nearby inputs (aggregate; one construction)"
               if isinstance(T2, dict) and T2["label"] and CTRL is True else "no controlled detection reading")
    return dict(prereg=pre, freeze=fsha, rejected=rej, cells=out, T1=T1, T2=T2, CTRL=CTRL, ctrl_vs_cmph_inputs=dist, reading=reading,
                T3={n: r["T3"].get("label") for n, r in out.items()}, primary_setting="S1")


def report(res):
    out, T1, T2, CTRL, reading = res["cells"], res["T1"], res["T2"], res["CTRL"], res["reading"]
    for n, r in out.items():
        print(n, r["role"], "| S1", {k: r["S1"][k] for k in ("res", "fp_median", "nf_median", "gs_median") if k in r["S1"]} or r["S1"],
              "| S0 res", r["S0"].get("res"), "| T3", r["T3"], flush=True)
    print("T1", T1, "| T2", T2, "| CTRL", CTRL, "| READING", reading, flush=True)


if __name__ == "__main__":
    main()
