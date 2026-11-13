"""Cross-machine reproducibility check (Supplementary Note 11): refits three frozen PPC fits on a second workstation into a
temporary directory and compares the SHA-256 of the fitted matrices with the original records.
Usage: python e12a_xcheck.py OUT_JSON"""
import sys, os, json, time, tempfile
os.environ["PPC_COND"] = "acc"
import numpy as np
import luppi_r4e_ppc as P
X = P.X


if __name__ == "__main__":
    out = sys.argv[1]; J = [j for j in X.job_list("e5") if j["subject_index"] in (0, 1, 2) and j["model"] == "signed"]; res = []
    with tempfile.TemporaryDirectory() as td:
        for j in J:
            frozen = json.load(open(j["file"])); t0 = time.time()
            r = X.run_job(dict(j, file=os.path.join(td, "x.json")), time.time() + 6 * 3600)
            C_new = np.load(os.path.join(td, "x_C.npy")); C_old = np.load(j["file"].replace(".json", "_C.npy"))
            res.append(dict(subject=j["tag"], identical_sha=bool(r["C_sha"] == frozen["C_sha"]), max_abs_diff=float(np.max(np.abs(C_new - C_old))),
                            mse_new=r["mse_ho"], mse_frozen=frozen["mse_ho"], secs=round(time.time() - t0, 1)))
            print(res[-1], flush=True)
    ok = all(x["identical_sha"] for x in res); ok_tol = all(x["max_abs_diff"] < 1e-10 for x in res)
    json.dump(dict(host=os.environ.get("COMPUTERNAME"), results=res, identical=ok, within_1e10=ok_tol), open(out, "x"), indent=1)
    print("IDENTICAL" if ok else ("WITHIN 1e-10" if ok_tol else "DIFFERENT"), flush=True)
    sys.exit(0 if ok_tol else 1)
