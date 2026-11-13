"""Code registration for the bifurcation-parameter checks (Supplementary Notes 13 and 14): one create-once record per group
holding the SHA-256 of the group's scripts, of this module and of the specification; check() refuses to run if anything has
changed.
Usage: python vj_rules.py GROUP file1 [file2 ...]"""
import sys, os, json, time, hashlib
from paths import DATA_ROOT

OUT = DATA_ROOT + "results/p33/vj/"; SPEC = DATA_ROOT + "docs_r1/183_vj_spec_v0.1.md"
SPEC_SHA = "f01f571884757da89e5859a29555543a5ff7f10c95b1a5ca1a84077283044459"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hashes(files):
    h = {os.path.basename(f): sha(f) for f in files}; h["vj_rules.py"] = sha(os.path.abspath(__file__)); h["spec"] = sha(SPEC); return h


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
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), group=g, hashes=hashes(files)), open(p, "x"), indent=1); print(open(p).read())
