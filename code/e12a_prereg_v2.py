"""Registration record of validation V2: writes the SHA-256 of e12a_validate_v2.py and of the frozen readout law before any V2 fit.
Usage: python e12a_prereg_v2.py"""
from paths import DATA_ROOT
import os, glob, json, time, hashlib
P33 = DATA_ROOT + "results/p33/"; P = P33 + "e12a_validate_v2_prereg.json"; sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if glob.glob(P33 + "e12a_v2/jobs/*.json"):
    raise SystemExit("E12a-V2 job files exist: pre-registration refused")
json.dump(dict(script="e12a_validate_v2.py", sha256=sha("e12a_validate_v2.py"), g_sha256=sha(P33 + "e12a_g.json"), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               note="registration record written before any V2 fit"), open(P, "x"), indent=1)
print("pre-registered", json.load(open(P)), flush=True)
