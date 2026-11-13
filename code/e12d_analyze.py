"""Simulation 4 analysis (Sections 2.5.2, 3.3): residuals of the in-family law g_sim for cooperative and CMPH truths and their
paired contrasts, against the specified criteria.
Usage: python e12d_analyze.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np

P33 = DATA_ROOT + "results/p33/"; PREREG = P33 + "e12d_analyze_prereg.json"; M = 0.023; MIN_N = 15


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load():
    import e12d_run as R
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
        out.setdefault((j["cell"], j["tag"]), {})[j["model"]] = (j, r)
    return out, rej, fsha


def main():
    pre = json.load(open(PREREG))
    if sha(__file__) != pre["sha256"]:
        raise SystemExit("e12d_analyze.py differs from the pre-registered version")
    import e12d_run as R
    D, rej, fsha = load(); res = compute(D, rej, fsha, pre, json.load(open(R.PANEL)))
    json.dump(res, open(sys.argv[1], "x"), indent=1, default=float); report(res)


def compute(D, rej, fsha, pre, P):
    import e12d_run as R, e12d_gsim_freeze as GS, e12a_validate as V, luppi_r4e_lib as Lib
    Gs = json.load(open(R.GSIM)); mg = Lib.support(np.load(R.OUT + "inp/SC_group.npy")); rows = {}
    for (cell, tag), d in D.items():
        if "signed" not in d:
            continue
        j, r = d["signed"]; nf, gs = V.inp_measures(Lib.bandpass(np.load(R.job_files(j)[0]))); fp = float((r["_C"][mg] < 0).mean())
        rows.setdefault(cell, {})[tag] = dict(fp=fp, nf=nf, gs=gs, res=fp - float(GS.predict(Gs, [nf])[0]),
                                              p2=float(r["mse_ho"] - d["coop"][1]["mse_ho"]) if "coop" in d else np.nan)
    rng = np.random.default_rng(1214)
    def boot(v):
        v = np.asarray(v); B = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
        return dict(n=len(v), median=float(np.median(v)), ci95=[float(np.quantile(B, 0.025)), float(np.quantile(B, 0.975))])
    cells = {}
    for name, c in P["cells"].items():
        rr = rows.get(name, {}); tags = sorted(t for t in c["supported"] if t in rr)
        if not c["eligible"] or len(tags) < MIN_N:
            cells[name] = dict(role=c["role"], label="not analysable", n=len(tags)); continue
        cells[name] = dict(role=c["role"], res=boot([rr[t]["res"] for t in tags]), fp_median=float(np.median([rr[t]["fp"] for t in tags])),
                           nf_median=float(np.median([rr[t]["nf"] for t in tags])), gs_median=float(np.median([rr[t]["gs"] for t in tags])))
        if "coop" in c["models"]:
            cells[name]["p2"] = boot([rr[t]["p2"] for t in tags if np.isfinite(rr[t]["p2"])])
    within = lambda r: -M <= r["res"]["ci95"][0] and r["res"]["ci95"][1] <= M
    d1c = {n: r for n, r in cells.items() if r["role"] == "D1" and "res" in r}
    D1 = "not analysable" if (not d1c or "res" not in cells["COOP_N49-d042"]) else dict(cells={n: bool(within(r)) for n, r in d1c.items()}, label=all(within(r) for r in d1c.values()))
    d2c = {n: r for n, r in cells.items() if r["role"] in ("D2", "D2opt") and "res" in r}
    D2 = "not analysable" if "res" not in cells["CMPH_N0-raw"] else dict(cells=sorted(d2c), label=all(r["res"]["ci95"][1] < -0.05 for r in d2c.values()))
    a, b = rows.get("CMPH_N0-raw", {}), rows.get("COOP_N49-d042", {})
    tags = sorted(set(a) & set(b) & set(P["cells"]["CMPH_N0-raw"]["supported"]) & set(P["cells"]["COOP_N49-d042"]["supported"]))
    if len(tags) >= MIN_N:
        D3 = dict(contrast=boot([a[t]["res"] - b[t]["res"] for t in tags])); D3["label"] = D3["contrast"]["ci95"][1] < -0.05
    else:
        D3 = "not analysable"
    ok = all(isinstance(x, dict) and x["label"] is True for x in (D1, D2, D3))
    reading = ("within the simulated family, the readout residual calibrated on a cooperative reference detects this engineered strong competition "
               "(aggregate; one construction; in principle; not a statement about real data)" if ok else "no in-family detection reading")
    return dict(prereg=pre, freeze=fsha, rejected=rej, cells=cells, D1=D1, D2=D2, D3=D3, reading=reading)


def report(res):
    for n, r in res["cells"].items():
        print(n, r, flush=True)
    print("D1", res["D1"], "| D2", res["D2"], "| D3", res["D3"], "| READING", res["reading"], flush=True)


if __name__ == "__main__":
    main()
