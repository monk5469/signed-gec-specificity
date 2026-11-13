"""Calibration targets of the known-truth simulations (Section 2.6): removal-dose targets (median residual global shares of the
human PPC inputs) and the structural and input-realism references.
Usage: python e10c_targets.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, glob, json, hashlib
import numpy as np
import luppi_hopf as L, luppi_r4e_lib as Lib
P33 = DATA_ROOT + "results/p33/"; W = DATA_ROOT + "work/"; PPC = P33 + "ppc_analysis.json"; EF = P33 + "ppc_experiment_freeze.json"; PR = P33 + "ppc_analysis_prereg.json"
NETS = P33 + "r4e_e8/networks.json"; DEMO = {"100307", "100408", "101107"}; DROP = 11
COND = dict(raw="mot", d284="accero", d189="acc1", d164="acc3", d136="acc", gsr="gsr")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
a = json.load(open(PPC))
if a["experiment_freeze"] != sha(EF) or a["prereg"]["sha256"] != json.load(open(PR))["sha256"]:
    raise SystemExit("PPC snapshot does not verify against its experiment freeze / pre-registration")
pc = a["secondary"]["per_condition"]
nets = np.asarray(json.load(open(NETS))["networks"]); iu = np.triu_indices(100, 1); same = (nets[:, None] == nets[None, :])[iu]


def struct(x):
    F = np.corrcoef(L.prep_emp(x))[iu]
    return dict(neg_fc=float((F < 0).mean()), sd=float(F.std()), mod=float(F[same].mean() - F[~same].mean()), mean_fc=float(F.mean()),
                gs=float(np.var(x.mean(0)) / np.mean(x.var(1))))


subs = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(W + "p33_ts_gsr/*_REST1_LR_s100.npy")} - DEMO)
assert len(subs) == 95, len(subs)
hashes, ref_in = {}, {}
for d, c in COND.items():
    R = []
    for s in subs:
        p = W + f"p33_ts_{c}/{s}_REST1_LR_s100.npy"; hashes[p] = sha(p)
        R.append(struct(Lib.bandpass(np.load(p).T.astype(float)[:, DROP:])))
    ref_in[d] = dict(condition=c, **{k: float(np.median([r[k] for r in R])) for k in R[0]}, mod_q25_q75=[float(np.quantile([r["mod"] for r in R], q)) for q in (0.25, 0.75)])
tg = dict(raw_gs=float(pc["mot"]["gs"]["median"]), d284=float(pc["accero"]["gs"]["median"]), d189=float(pc["acc1"]["gs"]["median"]),
          d136=float(pc["acc"]["gs"]["median"]), gsr_mod=ref_in["gsr"]["mod"])
ref = {d: dict(condition=c, fp=pc[c]["fp"], p2_negative_share=float(pc[c]["p2_negative_share"]), E_bn_fc=pc[c]["E_bn_fc"]) for d, c in COND.items() if d != "d164"}
json.dump(dict(source=PPC, source_sha=sha(PPC), experiment_freeze=sha(EF), n=95, subjects=subs, series_sha=hashes, targets=tg, input_reference=ref_in, reference=ref,
               note="targets: input-level only (gs medians, GSR modularity); input_reference and reference are never used for calibration"),
          open(sys.argv[1], "x"), indent=1)
print("targets", {k: round(v, 4) for k, v in tg.items()}, flush=True)
print("input reference", {d: {k: round(v, 3) for k, v in r.items() if k in ("gs", "neg_fc", "sd", "mod")} for d, r in ref_in.items()}, flush=True)
