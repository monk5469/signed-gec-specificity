"""Surface ICA-FIX condition (fix; descriptive; Sections 2.3, 3.2): the PPC fitting protocol applied unchanged to the HCP ICA-FIX
data.
Usage: python luppi_r4e_e5fix.py e5_prepare | freeze e5 | run e5 [--procs 56]"""
from paths import DATA_ROOT
import sys
import luppi_r4e_ext as X

X.OUT = DATA_ROOT + "results/p33/r4e_e5fix/"
X.HCP = DATA_ROOT + "work/p33_ts_fix/"
X.CODE = X.CODE + ["luppi_r4e_e5fix.py", "luppi_r4e_e5fix_analyze.py", "luppi_r4e_e8_analyze.py", "luppi_r4e_run.py"]   # + analysis deps
X.RAW = X.RAW + [DATA_ROOT + "results/p33/fix_parcellation_manifest.json"]      # dtseries -> parcellated lineage

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("group", nargs="?"); ap.add_argument("--procs", type=int, default=56)
    a = ap.parse_args(); os.makedirs(X.OUT, exist_ok=True)
    if a.stage == "e5_prepare":
        X.stage_e5_prepare()
    elif a.stage == "freeze":
        assert a.group == "e5"; X.stage_freeze("e5")
    elif a.stage == "run":
        assert a.group == "e5"; X.stage_run("e5", a.procs)
