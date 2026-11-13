"""Registration record of Simulation 3: writes the SHA-256 of e12c_analyze.py, the frozen joint support and the frozen readout law
before any fit.
Usage: python e12c_prereg.py"""
from paths import DATA_ROOT
import os, glob, json, time, hashlib
P33 = DATA_ROOT + "results/p33/"; P = P33 + "e12c_analyze_prereg.json"; sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if glob.glob(P33 + "e12c/jobs/*.json"):
    raise SystemExit("E12c jobs exist: pre-registration refused")
json.dump(dict(script="e12c_analyze.py", sha256=sha("e12c_analyze.py"), support_sha256=sha(P33 + "e12c_support.json"), g_sha256=sha(P33 + "e12a_g.json"),
               created=time.strftime("%Y-%m-%d %H:%M:%S"), note="registration record written before any Simulation 3 fit; analysis tested on synthetic fits"), open(P, "x"), indent=1)
print("pre-registered", json.load(open(P)), flush=True)
