"""Registration record of Simulation 5: writes, once, the SHA-256 of e14b_analyze.py, its helper e14_analyze.py, the runner and
the pre-run test before any fit.
Usage: python e14b_prereg.py"""
from paths import DATA_ROOT
import json, hashlib, time
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script="e14b_analyze.py", sha256=sha("e14b_analyze.py"), helper_sha256=sha("e14_analyze.py"),
           runner_sha256=sha("e14b_run.py"), fixture_sha256=sha("e14b_fixture.py"), config="luppi_hopf.RELEASED", primary="FP2a (C < -2e-4)",
           spec="see the paper, Methods")
json.dump(rec, open(DATA_ROOT + "results/p33/e14b_analyze_prereg.json", "x"), indent=1); print("prereg", rec["sha256"][:12])
