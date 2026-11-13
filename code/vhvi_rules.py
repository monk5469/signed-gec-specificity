"""Code registration for the distributional implementation comparison and the frequency control (Supplementary Notes 13 and 15):
one create-once record per group holding the SHA-256 of the group's scripts, of this module and of the specification; check()
refuses to run if anything has changed.
Usage: python vhvi_rules.py GROUP file1 [file2 ...]"""
import sys, os, json, time, hashlib
from paths import DATA_ROOT

OUT = DATA_ROOT + "results/p33/vhvi/"; SPEC = DATA_ROOT + "docs_r1/179_vhvi_spec_v0.1.md"
SPEC_SHA = "b95699c0a7d02276d97e95de4ad3f83ec30420256363a120f1366d53011947f1"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hashes(files):
    h = {os.path.basename(f): sha(f) for f in files}; h["vhvi_rules.py"] = sha(os.path.abspath(__file__)); h["spec"] = sha(SPEC); return h


def check(group, files):
    if sha(SPEC) != SPEC_SHA:
        raise SystemExit("spec differs from its registration")
    r = json.load(open(OUT + f"rules_{group}.json"))["hashes"]; now = hashes(files)
    if r != now:
        raise SystemExit(f"{group}: registered code changed: {[k for k in now if now[k] != r.get(k)]}")


if __name__ == "__main__":
    g, files = sys.argv[1], sys.argv[2:]; p = OUT + f"rules_{g}.json"
    if sha(SPEC) != SPEC_SHA:
        raise SystemExit("spec differs from its registration")
    if os.path.exists(p):
        raise SystemExit("already registered")
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), group=g, hashes=hashes(files)), open(p, "x"), indent=1); print(open(p).read())
