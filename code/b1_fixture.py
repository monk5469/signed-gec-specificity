"""Pre-run test of b1_analyze.py: runs the analysis on frozen PPC acc and mot fits in place of the accS fits, and on synthetic
accS samples covering each classification of the equivalence rule.
Usage: python b1_fixture.py"""
import sys, json, copy, traceback
import numpy as np
bad = 0
try:
    import b1_analyze as A
    acc, dacc = A.ppc_side("acc"); mot, dmot = A.ppc_side("mot"); print("acc", dacc, "| mot", dmot, flush=True)
    G = json.load(open(A.P33 + "e12a_g.json")); sim = {t: 1.0 for t in acc}
    res = A.compute(dict(accS=copy.deepcopy(acc), acc=acc, mot=mot), sim, np.random.default_rng(150), G); A.show(res)
    assert res["B1_E"] == "equivalent" and abs(res["dFP_acc"]["median"]) < 1e-12
    rng = np.random.default_rng(7)
    for name, mu, sd in (("inside", 0.0, 0.02), ("cross_pos", 0.03, 0.03), ("cross_neg", -0.03, 0.03), ("outside", 0.08, 0.02)):
        sh = copy.deepcopy(acc); e = rng.normal(0, sd, len(sh)); e = e - np.median(e) + mu      # median shift exactly mu
        for v, x in zip(sh.values(), e):
            v["FP"] += x
        r = A.compute(dict(accS=sh, acc=acc, mot=mot), sim, np.random.default_rng(150), None); L, U = r["dFP_acc"]["ci95"]
        print(name, "CI [%.4f, %.4f]" % (L, U), "| E", r["B1_E"], "| E(0.015)", r["B1_E_0015"], "| T", r["B1_T"], "|", r["reading"][:70], flush=True)
        assert r["B1_E"] == A.e_status([L, U], 0.03)
        if name == "inside":
            assert r["B1_E"] == "equivalent"
        if name in ("cross_pos", "cross_neg"):
            assert r["B1_E"] == "not established" and (L < 0.03 < U or L < -0.03 < U), (L, U)
            assert r["reading"].startswith("equivalence within 0.03 was not established") or r["B1_T"] is not True
        if name == "outside":
            assert r["B1_E"] == "outside"
    for ci, want in (([-0.03, 0.01], "not established"), ([-0.01, 0.03], "not established"), ([0.03, 0.05], "not established"),
                     ([0.0301, 0.05], "outside"), ([-0.05, -0.0301], "outside"), ([-0.0299, 0.0299], "equivalent")):
        assert A.e_status(ci, 0.03) == want, (ci, A.e_status(ci, 0.03), want)
    print("boundary cases ok | fixture ok", flush=True)
except Exception:
    traceback.print_exc(); bad = 1
sys.exit(bad)
