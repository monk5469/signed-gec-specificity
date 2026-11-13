"""Download verification of the myelin maps (Section 2.7): compares every file with the S3 ETag and size recorded at download.
Usage: python myelin_etag.py"""
from paths import DATA_ROOT
import hashlib, json, os
D = DATA_ROOT + "raw/p33_myelin/"; PART = 8 * 1024 * 1024


def job(line):
    n, e, s = line.rstrip("\n").split("\t"); p = D + n
    if not os.path.exists(p):
        return dict(name=n, ok=False, reason="missing")
    nparts = int(e.split("-")[1]) if "-" in e else 1
    if nparts == 1:
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
    else:
        md5s = []
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(PART), b""):
                md5s.append(hashlib.md5(b).digest())
        h = hashlib.md5(b"".join(md5s)).hexdigest() + f"-{len(md5s)}"
    return dict(name=n, ok=bool(h == e and os.path.getsize(p) == int(s)), etag_s3=e, etag_local=h, size=os.path.getsize(p))


if __name__ == "__main__":
    lines = [l for l in open(D + "myelin_etags.txt") if l.strip()]
    res = [job(l) for l in lines]
    bad = [r for r in res if not r["ok"]]
    json.dump(dict(n=len(res), n_ok=len(res) - len(bad), all_ok=not bad, bad=bad, part_size=PART, etags_sha=hashlib.sha256(open(D + "myelin_etags.txt", "rb").read()).hexdigest()),
              open(D + "myelin_etag_verify.json", "w"), indent=1)
    print("files", len(res), "ok", len(res) - len(bad), "bad", [b["name"] for b in bad][:10], flush=True)
