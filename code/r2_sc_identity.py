"""Reference-SC check (Section 2.1): compares the SC distributed with the authors' example code with the individual SCs of the
three example participants."""
import numpy as np, scipy.io as sio
from paths import DATA_ROOT

a = sio.loadmat(DATA_ROOT + "raw/luppi_demo/gec_3/SC.mat")["DTI_fibers_VolNorm_HCP"].ravel()
g = np.load(DATA_ROOT + "results/p33/r4e_ppc_acc/e5/SC.npy")
s = sio.loadmat(DATA_ROOT + "raw/luppi_demo/single/SC.mat")
print("single/SC.mat keys", [k for k in s if not k.startswith("__")])
for i, x in enumerate(a):
    x = np.asarray(x, float)
    print(i, "max|diff| vs e5/SC.npy", float(np.max(np.abs(x - g))), "| corr", round(float(np.corrcoef(x.ravel(), g.ravel())[0, 1]), 4))
m = ~np.eye(100, dtype=bool); avg = np.mean([np.asarray(x, float) for x in a], 0)
print("mean of the 3 vs e5/SC.npy corr", round(float(np.corrcoef(avg[m], g[m])[0, 1]), 4), "| density of mean", round(float((avg[m] > 0).mean()), 3))
