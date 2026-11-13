"""Selection of the 60 V2 participants (Section 2.1): screening of HCP participants outside the main cohort in ascending ID order,
with the motion criterion and file availability.
Usage: python e12a_v2_select.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, hashlib
D = DATA_ROOT + "raw/p33_hcp_new/"; N = 60; THR = 0.20
c = json.load(open(D + "screen_candidates.json")); sel, log = [], []
for s in c["candidates"]:
    fs = [D + f"screen/{s}_REST1_{r}_RelativeRMS_mean.txt" for r in ("LR", "RL")]
    if not c["files_available"].get(s):
        log.append(dict(subject=s, status="missing REST1 files")); continue
    if not all(os.path.exists(f) for f in fs):
        log.append(dict(subject=s, status="motion file not retrieved")); continue
    mot = [float(open(f).read().split()[0]) for f in fs]
    ok = max(mot) <= THR; log.append(dict(subject=s, status="eligible" if ok else "motion", motion=mot, sha=[hashlib.sha256(open(f, "rb").read()).hexdigest() for f in fs]))
    if ok:
        sel.append(s)
        if len(sel) == N:
            break
if len(sel) < N:
    raise SystemExit(f"only {len(sel)} eligible among {len(c['candidates'])} candidates: extend the ordered screening")
json.dump(dict(rule="first 60 eligible in ascending ID order; eligible = all REST1 files + max RelativeRMS mean (LR, RL) <= 0.20 mm", threshold=THR,
               selected=sel, last_examined=log[-1]["subject"], log=log, candidates_sha=hashlib.sha256(open(D + "screen_candidates.json", "rb").read()).hexdigest()),
          open(sys.argv[1], "x"), indent=1)
print("selected", len(sel), "| examined", len(log), "| statuses", {k: sum(x["status"] == k for x in log) for k in ("eligible", "motion", "missing REST1 files", "motion file not retrieved")}, flush=True)
