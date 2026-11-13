"""Registration record of Simulation 1: writes the SHA-256 of luppi_r4e_e10c_analyze.py before any fit; refuses if a Simulation 1
job file exists.
Usage: python e10c_prereg.py"""
from paths import DATA_ROOT
import os, glob, json, time, hashlib
OUT = DATA_ROOT + "results/p33/r4e_e10c/"; P = OUT + "r4e_e10c_analysis_prereg.json"; S = "luppi_r4e_e10c_analyze.py"
if glob.glob(OUT + "jobs/*.json"):
    raise SystemExit("E10c job files exist: pre-registration refused")
os.makedirs(OUT, exist_ok=True)
json.dump(dict(script=S, sha256=hashlib.sha256(open(S, "rb").read()).hexdigest(), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               note="registration record written before any Simulation 1 fit (input-level checks only)"),
          open(P, "x"), indent=1)
print("pre-registered", json.load(open(P))["sha256"], flush=True)
