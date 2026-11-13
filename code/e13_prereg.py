"""Registration record of the hierarchy analysis: writes, once, the SHA-256 of e13_analyze.py, the hierarchy file, the fit
manifest and the pre-run test before any alignment statistic.
Usage: python e13_prereg.py"""
from paths import DATA_ROOT
import json, hashlib, time
P33 = DATA_ROOT + "results/p33/"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script="e13_analyze.py", sha256=sha("e13_analyze.py"), hier_sha256=sha(P33 + "e13_hier.json"),
           fit_manifest_sha256=sha(P33 + "e13_fit_manifest.json"), fixture_sha256=sha("e13_fixture.py"), version="v1.1",
           spec="see the paper, Methods")
json.dump(rec, open(P33 + "e13_analyze_prereg.json", "x"), indent=1); print("prereg", rec["sha256"][:12], "hier", rec["hier_sha256"][:12], "fitman", rec["fit_manifest_sha256"][:12])
