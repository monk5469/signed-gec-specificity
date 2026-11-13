"""Registration record of the participant-mask analysis: writes, once, the SHA-256 of b1_analyze.py, b1_run.py and their pre-run
test before any fit.
Usage: python b1_prereg.py"""
from paths import DATA_ROOT
import json, hashlib, time
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), script="b1_analyze.py", sha256=sha("b1_analyze.py"), runner_sha256=sha("b1_run.py"),
           fixture_sha256=sha("b1_fixture.py"), version="v0.2", mask="MNINonLinear/ROIs/wmparc.2.nii.gz (subject-specific)",
           spec="see the paper, Methods")
json.dump(rec, open(DATA_ROOT + "results/p33/b1_analyze_prereg.json", "x"), indent=1); print("prereg", rec["sha256"][:12])
