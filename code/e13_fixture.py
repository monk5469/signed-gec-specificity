"""Pre-run test of e13_analyze.py: builds the fit manifest in memory and runs every set with random hierarchy vectors and a random
distance matrix, so that no real alignment statistic is computed before registration; also checks that a tampered manifest
entry is rejected.
Usage: python e13_fixture.py"""
import sys, copy, traceback
import numpy as np
import e13_analyze as A

rng = np.random.default_rng(99); H = dict(A2=rng.permutation(100).astype(float), A1=rng.permutation(100).astype(float))
R = rng.random((100, 100)); dist = (R + R.T) / 2; np.fill_diagonal(dist, 0); bad = 0
try:
    man = A.build_manifest()
    for name in A.set_names():
        rows, den = A.load_set(name, H, dist, man); s = A.summarise(rows, np.random.default_rng(0), not name.startswith("real_"))
        e = s["A2"]["E_h"]
        print(name, "| expected", den["expected"], "found", den["records_found"], "missing", den["missing"], "admitted", den["admitted"],
              "| rejected", {k: v for k, v in den["rejected"].items() if v}, "| E_h admissible", e["admissible"], "insufficient", e["insufficient"],
              "nonconverged", e["nonconverged"], "| strata", "per_stratum" in e, flush=True)
    t = copy.deepcopy(man); k0 = next(iter(t["sets"]["real_acc"]["jobs"])); t["sets"]["real_acc"]["jobs"][k0]["C_sha"] = "0" * 64
    rows, den = A.load_set("real_acc", H, dist, t)
    assert den["rejected"]["manifest_mismatch"] == 1 and den["admitted"] == 94, den
    print("tamper check ok", flush=True)
except Exception:
    traceback.print_exc(); bad = 1
sys.exit(bad)
