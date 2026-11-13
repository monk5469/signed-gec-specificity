"""Pre-run test of e14b_analyze.py: runs the analysis on the fits of the initial exploratory L1 run in place of the Simulation 5
fits, and checks the zero-denominator branches of the ratios.
Usage: python e14b_fixture.py"""
from paths import DATA_ROOT
import sys, traceback
import numpy as np
bad = 0
try:
    import e14b_analyze as B, e14_analyze as A, e14_run as R14
    rows, den, _ = A.load(R14); print("E14 rows", den, flush=True)
    res = B.compute(rows, DATA_ROOT + "results/p33/r4e_e11/", np.random.default_rng(141)); B.show(res)
    assert res["P"] is True and res["Q1_L1_numeric"] is True
    pr = res["R"]["FP2a"]["coop_over_cmph_paired"]; assert "median" in pr and pr["n"] == 30, pr
    assert res["cmph_counts"]["enrichment_n"] == 30 and res["cmph_counts"]["no_prediction"] == 0
    z = dict(median=0.0, _boot=[0.0] * 10); nz = dict(median=0.1, _boot=[0.1] * 10); part = dict(median=0.1, _boot=[0.0] + [0.1] * 9)
    assert B.ratio(nz, z).startswith("undefined") and B.ratio(nz, part)["ci95"] == "unbounded"
    print("per-fit rows", len(res["per_fit"]), "| fixture ok", flush=True)
except Exception:
    traceback.print_exc(); bad = 1
sys.exit(bad)
