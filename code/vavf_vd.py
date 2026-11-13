"""Original hierarchy estimand O_h (Sections 2.7.1, 3.6; Supplementary Note 9): co-occurrence of negative weights with negative
entries of the hierarchy outer product, the sign-swap null and the group-level spin test.
Usage: python vavf_vd.py OUT_JSON [--fixture]"""
import sys, json, argparse
import numpy as np

SETS = ("real_mot", "real_acc", "real_gsr", "e10c_COOP_d136", "e11_COOP_N49d136", "e12c_COOP_N49d136_S1",
        "e10c_CMPN_d136", "e11_CMPN_N49d136", "e11_CMPH_N49d136")


def co(neg, S, Hn):
    s = S & ~np.eye(len(S), dtype=bool); M = int(s.sum()); k = int((neg & s).sum()); b = float((Hn & s).sum() / M)
    if k == 0:
        return dict(k=0, O=None, b=b, excess=None, z=None)
    O = float((neg & s & Hn).sum() / k); sd = np.sqrt(b * (1 - b) * (M - k) / (k * (M - 1))) if k < M else np.nan
    return dict(k=k, O=O, b=b, excess=O - b, z=float((O - b) / sd) if sd > 0 else None)


def fc_of(inp):
    inp = np.asarray(inp, float)
    return inp if inp.shape[0] == inp.shape[1] else np.corrcoef(inp)


def boot(v, rng, B=2000):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    bs = [np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(B)]
    return dict(n=int(v.size), median=float(np.median(v)), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--fixture", action="store_true"); a = ap.parse_args()
    import r4_spin as R4
    if not a.fixture:
        import vavf_rules
        vavf_rules.check("vd", ("vavf_vd.py",))
    data, h, dist = R4.load_raw(SETS); z = (h - h.mean()) / h.std(); Hn = np.outer(z, z) < 0; rng = np.random.default_rng(20261027); out = {}
    c, hemi = R4.centroids(); perms = R4.spin_perms(c, hemi, 50 if a.fixture else 10000, 20261016)
    for n in SETS:
        rows = [v["raw"] for v in data[n][0].values()]; rows = rows[:5] if a.fixture else rows
        ind = [co(np.asarray(r["C"]) < 0, np.asarray(r["SC"]) != 0, Hn) for r in rows]
        inp = [co(fc_of(r["inp"]) < 0, np.asarray(r["SC"]) != 0, Hn) for r in rows]
        o = dict(n_fits=len(rows), individual=dict(excess=boot([x["excess"] for x in ind], rng), z=boot([x["z"] for x in ind], rng),
                                                    share_z_gt_1_96=float(np.mean([x["z"] is not None and x["z"] > 1.96 for x in ind]))),
                 input=dict(excess=boot([x["excess"] for x in inp], rng), z=boot([x["z"] for x in inp], rng)))
        if n.startswith("real"):
            S = np.asarray(rows[0]["SC"]) != 0; Cm = np.mean([np.asarray(r["C"]) for r in rows], 0); g = co(Cm < 0, S, Hn)
            Fm = np.mean([fc_of(r["inp"]) for r in rows], 0); gi = co(Fm < 0, S, Hn)
            o["group"] = dict(fit=g, input=gi, spin_p=None, spin_null_q95=None, note="no negative entry in the group-mean C" if g["excess"] is None else "")
            if g["excess"] is not None:
                null = np.array([co(Cm < 0, S, np.outer(z[p], z[p]) < 0)["excess"] for p in perms], float)
                o["group"].update(spin_p=float((1 + np.sum(null >= g["excess"])) / (1 + null.size)), spin_null_q95=float(np.quantile(null, 0.95)))
        out[n] = o
        print("V-D", n, "| individual excess", o["individual"]["excess"], "| z", o["individual"]["z"].get("median"), "| input excess",
              o["input"]["excess"].get("median"), "| group", {k: o["group"][k] for k in ("spin_p",)} if "group" in o else "", flush=True)
    json.dump(dict(fixture=a.fixture, sets=out), open(a.out, "w" if a.fixture else "x"), indent=1, default=float)
