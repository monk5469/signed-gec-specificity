"""Exploratory analysis of Simulation 6 (Section 3.4.3): Spearman correlation across truths between baseline FP2alpha and the
negative-FC fraction of the simulated input."""
import json
import numpy as np
from scipy.stats import spearmanr
import r1_run as R, e14_analyze as A

SC = R.group_sc(); m = A.supp(SC)
man = json.load(open(R.MAN))["calibration"]
nf = {c["k"]: c["input"]["A0"]["neg_fc"] for c in man if c.get("valid") and c["sg"] == "s01" and c["g"] == 0}
nfx = {c["k"]: c["input"]["A0x"]["neg_fc"] for c in man if c.get("valid") and c["sg"] == "s01" and c["g"] == 0}
C = {}
for j in R.job_list():
    if j["sg"] == "s01" and j["g"] == 0 and j["seed"] == 11 and j["model"] == "signed" and j["arm"] in ("A0", "A0x", "A1"):
        C.setdefault(j["arm"], {})[j["k"]] = np.load(j["file"].replace(".json", "_C.npy"))
fp = lambda X: float(np.mean(X[m] < -2e-4)); f0 = lambda X: float(np.mean(X[m] < 0))
ks = sorted(C["A0"]); a = [fp(C["A0"][k]) for k in ks]
print("A0 FP2a range", round(min(a), 3), round(max(a), 3), "| truths with FP2a = 0:", sum(x == 0 for x in a), "| FP0 median", round(float(np.median([f0(C["A0"][k]) for k in ks])), 3))
print("Spearman(A0 FP2a, A0 input nf):", [round(float(x), 4) for x in spearmanr(a, [nf[k] for k in ks])])
ax = [fp(C["A0x"][k]) for k in ks]
print("Spearman(A0x FP2a, A0x input nf):", [round(float(x), 4) for x in spearmanr(ax, [nfx[k] for k in ks])])
print("A1 FP2a median", round(float(np.median([fp(C["A1"][k]) for k in ks])), 3))
print("per truth (A0 input nf, A0 FP2a):", [(round(nf[k], 2), round(fp(C["A0"][k]), 3)) for k in sorted(ks, key=lambda k: nf[k])])
