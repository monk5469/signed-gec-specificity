"""Hierarchy map (Section 2.7): parcellates the T1w/T2w myelin maps into the surface Schaefer-100 atlas and freezes the hierarchy
vectors A2 (participant mean, primary) and A1 (HCP group average).
Usage: python e13_hier.py OUT_JSON"""
from paths import DATA_ROOT
import sys, os, json, glob, hashlib
import numpy as np
import nibabel as nib

D = DATA_ROOT + "raw/p33_myelin/"; ATLAS = DATA_ROOT + "raw/p33_hcp/Schaefer2018_100Parcels_7Networks_order.dlabel.nii"
SUBJ = DATA_ROOT + "results/p33/p33_hcp_s200_v13.json"; A1 = "Q1-Q6_R440.MyelinMap_BC.32k_fs_LR.dscalar.nii"
VOLATLAS = DATA_ROOT + "raw/p33_hcp_vol/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def vertex_map(img):
    """{(structure, vertex): column index} for the cortical brain models of a CIFTI image."""
    ax = img.header.get_axis(1); out = {}
    for name, sl, bm in ax.iter_structures():
        if "CORTEX" in name:
            for k, v in enumerate(bm.vertex):
                out[(name, int(v))] = sl.start + k
    return out


if __name__ == "__main__":
    v = json.load(open(D + "myelin_etag_verify.json"))
    if not v.get("all_ok"):
        raise SystemExit("download verification missing or failed")
    lab = nib.load(ATLAS); L = np.asarray(lab.get_fdata())[0].astype(int); lm = vertex_map(lab)
    subs = [s["subject"] for s in json.load(open(SUBJ))["subjects"]]
    def load(p):
        img = nib.load(p); x = np.asarray(img.get_fdata())[0]; vm = vertex_map(img); return {k: x[c] for k, c in vm.items()}
    maps = [load(D + f"{s}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii") for s in subs]
    keys = sorted(set(lm) & set.intersection(*[set(m) for m in maps]))
    a2 = {k: float(np.mean([m[k] for m in maps])) for k in keys}; a1 = load(D + A1)
    def parcel(vals):
        h = np.full(100, np.nan)
        for p in range(1, 101):
            ks = [k for k in keys if L[lm[k]] == p and k in vals]
            h[p - 1] = float(np.mean([vals[k] for k in ks])) if ks else np.nan
        return h
    hA2, hA1 = parcel(a2), parcel(a1)
    if np.any(~np.isfinite(hA2)) or np.any(~np.isfinite(hA1)):
        raise SystemExit("a parcel has no vertices")
    va = nib.load(VOLATLAS); vl = np.asarray(va.dataobj).astype(int); cen = np.zeros((100, 3))
    for p in range(1, 101):
        ijk = np.argwhere(vl == p)
        if len(ijk) == 0:
            raise SystemExit("a volumetric parcel is empty")
        cen[p - 1] = (ijk @ va.affine[:3, :3].T + va.affine[:3, 3]).mean(0)
    dist = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    out = dict(h_A2=hA2.tolist(), h_A1=hA1.tolist(), corr_A2_A1=float(np.corrcoef(hA2, hA1)[0, 1]), n_subjects_A2=len(maps), n_vertices=len(keys),
               centroids_mm=cen.tolist(), dist_mm=dist.tolist(), vol_atlas=VOLATLAS, vol_atlas_sha=sha(VOLATLAS),
               atlas=ATLAS, atlas_sha=sha(ATLAS), inputs={os.path.basename(p): sha(p) for p in sorted(glob.glob(D + "*.dscalar.nii"))},
               script_sha=sha(__file__), note="h = parcel mean of the bias-corrected T1w/T2w myelin map; higher = more myelinated (lower hierarchy)")
    json.dump(out, open(sys.argv[1], "x"), indent=1)
    print("hierarchy frozen: A2 from", len(maps), "subjects,", len(keys), "vertices; corr(A2, A1) = %.3f" % out["corr_A2_A1"], "| A2 range %.3f-%.3f" % (hA2.min(), hA2.max()), flush=True)
