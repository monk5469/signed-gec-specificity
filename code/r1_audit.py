"""Exploratory audit of Simulation 6 without new fits (Section 3.4.3; Supplementary Note 10): baseline negative weights stratified
by true weight, and stopping reasons."""
import json, collections
import numpy as np
import r1_run as R, e14_analyze as A

SC = R.group_sc(); m = A.supp(SC)
print("SC sha", R.sha_arr(SC)[:12], "| support entries", int(m.sum()))
truth = {k: np.load(R.OUT + f"truth/k{k}_COOP_C.npy") for k in range(1, 31)}
outside = sum(int(((truth[k] != 0) & ~m & ~np.eye(100, dtype=bool)).sum()) for k in truth); diag = sum(int((np.diag(truth[k]) != 0).sum()) for k in truth)
print("support audit | truth nonzero off-diagonal entries outside SC support (all truths):", outside, "| nonzero diagonal entries:", diag)
pooled = np.concatenate([truth[k][m][truth[k][m] > 0] for k in truth]); edges = np.quantile(pooled, [0.25, 0.5, 0.75])
print("zero-truth share of support (median over truths):", round(float(np.median([np.mean(truth[k][m] == 0) for k in truth])), 3), "| positive-weight quartile edges", np.round(edges, 4).tolist())
J = [j for j in R.job_list() if j["sg"] == "s01" and j["g"] == 0 and j["seed"] == 11 and j["model"] == "signed" and j["arm"] in ("A0", "A1", "A2")]
strata = ("zero", "Q1", "Q2", "Q3", "Q4"); res = {}; stops = collections.defaultdict(collections.Counter); iters = collections.defaultdict(list)
for j in J:
    C = np.load(j["file"].replace(".json", "_C.npy")); rec = json.load(open(j["file"])); t = truth[j["k"]][m]; w = C[m]
    stops[j["arm"]][rec.get("stop")] += 1; iters[j["arm"]].append(rec.get("iterations"))
    lab = np.where(t == 0, 0, 1 + np.searchsorted(edges, t))
    r = res.setdefault(j["arm"], {s: dict(within=[], contrib=[]) for s in strata})
    for i, s in enumerate(strata):
        sel = lab == i
        if sel.any():
            r[s]["within"].append(float(np.mean(w[sel] < -2e-4))); r[s]["contrib"].append(float(np.sum(w[sel] < -2e-4) / w.size))
for a in ("A0", "A1", "A2"):
    print(a, "| within-stratum share C<-2a (median):", {s: round(float(np.median(v["within"])), 4) for s, v in res[a].items()},
          "| contribution to FP2a:", {s: round(float(np.median(v["contrib"])), 4) for s, v in res[a].items()})
    print("   stop reasons", dict(stops[a]), "| iterations median", float(np.median(iters[a])), "range", min(iters[a]), max(iters[a]))
JP = [j for j in R.job_list() if j["sg"] == "s01" and j["g"] == 0 and j["seed"] == 11 and j["model"] == "signed" and j["arm"] == "Pplus"]
sp = collections.Counter(json.load(open(j["file"])).get("stop") for j in JP); it = [json.load(open(j["file"])).get("iterations") for j in JP]
print("Pplus stop reasons", dict(sp), "| iterations median", float(np.median(it)))
