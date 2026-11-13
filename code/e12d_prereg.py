"""Registration record of Simulation 4: writes the SHA-256 of e12d_analyze.py and of the frozen g_sim before any fit.
Usage: python e12d_prereg.py"""
from paths import DATA_ROOT
import os, glob, json, time, hashlib
P33 = DATA_ROOT + "results/p33/"; P = P33 + "e12d_analyze_prereg.json"; sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if glob.glob(P33 + "e12d/jobs/*.json"):
    raise SystemExit("E12d jobs exist: pre-registration refused")
json.dump(dict(script="e12d_analyze.py", sha256=sha("e12d_analyze.py"), gsim_sha256=sha(P33 + "e12d_gsim.json"),
               created=time.strftime("%Y-%m-%d %H:%M:%S"), note="registration record written before any Simulation 4 fit; analysis tested on synthetic fits"), open(P, "x"), indent=1)
print("pre-registered", json.load(open(P)), flush=True)
