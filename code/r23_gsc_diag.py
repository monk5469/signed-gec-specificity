"""Descriptive check of the SC calibration (Supplementary Note 12): participant identity and region order of the reconstructed
SCs against the authors' example SCs."""
import json
import numpy as np, scipy.io as sio, nibabel as nib
from scipy.stats import spearmanr
import r23_sc as S

iu = np.triu_indices(100, 1)


def union_sp(a, b):
    x, y = a[iu], b[iu]; u = (x > 0) | (y > 0); return float(spearmanr(x[u], y[u])[0])


def jac(a, b):
    x, y = a[iu] > 0, b[iu] > 0; return float((x & y).sum() / (x | y).sum())


if __name__ == "__main__":
    ref = [np.asarray(x, float) for x in sio.loadmat(S.DEMO_SC)["DTI_fibers_VolNorm_HCP"].ravel()]
    ours = [np.load(S.SCDIR + f"{s}.npz")["sc100_ce1"].astype(float) for s in S.DEMO]
    A = nib.load(S.A100); L = np.asarray(A.dataobj).astype(int); ijk = np.argwhere(L > 0); xyz = nib.affines.apply_affine(A.affine, ijk)
    cen = np.array([xyz[L[tuple(ijk.T)] == k].mean(0) for k in range(1, 101)]); D = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    out = dict(ref_props=[dict(shape=r.shape, symmetric=bool(np.allclose(r, r.T)), diag_max=float(np.abs(np.diag(r)).max()), density=float((r[iu] > 0).mean()),
                               min_pos=float(r[r > 0].min()), max=float(r.max())) for r in ref])
    out["cross_union_spearman_ours_x_ref"] = [[round(union_sp(o, r), 3) for r in ref] for o in ours]
    out["ref_x_ref"] = [[round(union_sp(a, b), 3) for b in ref] for a in ref]; out["ours_x_ours"] = [[round(union_sp(a, b), 3) for b in ours] for a in ours]
    out["jaccard_ref_x_ref"] = [[round(jac(a, b), 3) for b in ref] for a in ref]; out["jaccard_ours_x_ours"] = [[round(jac(a, b), 3) for b in ours] for a in ours]
    d = D[iu]; inter = np.zeros((100, 100), bool); inter[:50, 50:] = True; inter[50:, :50] = True; it = inter[iu]
    swap = np.r_[50:100, 0:50]
    rows = []
    for s, o, r in zip(S.DEMO, ours, ref):
        x, y = o[iu], r[iu]; m = (x > 0) & (y > 0)
        rows.append(dict(id=s,
            dist_spearman_nonzero=dict(ours=float(spearmanr(x[x > 0], d[x > 0])[0]), ref=float(spearmanr(y[y > 0], d[y > 0])[0])),
            presence_vs_distance_auc_like=dict(ours=float(spearmanr(x > 0, d)[0]), ref=float(spearmanr(y > 0, d)[0])),
            interhemispheric_share_of_edges=dict(ours=float(it[x > 0].mean()), ref=float(it[y > 0].mean())),
            homotopic_present=dict(ours=int(sum(o[i, i + 50] > 0 for i in range(50))), ref=int(sum(r[i, i + 50] > 0 for i in range(50)))),
            ref_hemisphere_swapped_union_spearman=union_sp(o, r[np.ix_(swap, swap)]),
            shared_support_spearman=float(spearmanr(x[m], y[m])[0]), shared_support_pearson_log=float(np.corrcoef(np.log(x[m]), np.log(y[m]))[0, 1]),
            n_shared=int(m.sum()), n_ours_only=int(((x > 0) & ~(y > 0)).sum()), n_ref_only=int((~(x > 0) & (y > 0)).sum()),
            mean_dist_mm=dict(shared=float(d[m].mean()), ours_only=float(d[(x > 0) & ~(y > 0)].mean()), ref_only=float(d[~(x > 0) & (y > 0)].mean())),
            ref_only_weight_pct_rank=float(np.mean([(y[m] < v).mean() for v in y[~(x > 0) & (y > 0)]])),
            ours_only_weight_pct_rank=float(np.mean([(x[m] < v).mean() for v in x[(x > 0) & ~(y > 0)]])),
            strength_spearman=float(spearmanr(o.sum(0), r.sum(0))[0]),
            top10pct_overlap=float(len(set(np.argsort(-x)[:int(0.1 * (y > 0).sum())]) & set(np.argsort(-y)[:int(0.1 * (y > 0).sum())])) / int(0.1 * (y > 0).sum()))))
    out["per_participant"] = rows
    json.dump(out, open(S.OUT + "r23_gsc_diag.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
