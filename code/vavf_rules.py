"""Code registration for the released-configuration, tissue-pool, hierarchy-estimand and descriptive checks: one create-once
record per script group holding the SHA-256 of the group's scripts, of this module and of the specification; check() refuses to
run if anything has changed.
Usage: python vavf_rules.py GROUP script1.py [script2.py ...]"""
import sys, os, json, time, hashlib
from paths import DATA_ROOT

OUT = DATA_ROOT + "results/p33/vavf/"; SPEC = DATA_ROOT + "docs_r1/177_vavf_spec_v0.1.md"
GROUPS = {"desc": ("vavf_desc.py",)}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def hashes(files):
    h = {f: sha(f) for f in files}; h["vavf_rules.py"] = sha(os.path.abspath(__file__)); h["spec"] = sha(SPEC); return h


def check(group="desc", files=None):
    files = files or GROUPS[group]; r = json.load(open(OUT + f"rules_{group}.json"))["hashes"]; now = hashes(files)
    if r != now:
        raise SystemExit(f"{group}: registered code changed: {[k for k in now if now[k] != r.get(k)]}")


if __name__ == "__main__":
    g, files = sys.argv[1], sys.argv[2:]; p = OUT + f"rules_{g}.json"
    if os.path.exists(p):
        raise SystemExit("already registered")
    json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), group=g, hashes=hashes(files)), open(p, "x"), indent=1); print(open(p).read())
