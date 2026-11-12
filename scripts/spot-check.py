"""Record a ticked spot check (tuna review file) in its question set's manifest.

  .venv/bin/python scripts/spot-check.py REVIEW_MD --repo REPO --set SET

Each question block in the review file has three boxes: good / too vague / gives the answer away. A
question counts as rejected when "too vague" or "gives the answer away" is ticked; blocks with no tick are
counted as unreviewed. Writes manifest["spot_check"] = {file, reviewed, good, vague, gives_away,
unreviewed, reject_share}.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", default=common.default_out())
    args = ap.parse_args()
    blocks = re.split(r"\n## \d+\. ", open(args.review).read())[1:]
    good = vague = give = unrev = 0
    for b in blocks:
        g = re.search(r"- \[[xX]\] good", b)
        v = re.search(r"- \[[xX]\] too vague", b)
        a = re.search(r"- \[[xX]\] gives the answer away", b)
        if not (g or v or a):
            unrev += 1
        elif v or a:
            vague += bool(v)
            give += bool(a)
        else:
            good += 1
    reviewed = len(blocks) - unrev
    rej = sum(1 for b in blocks if re.search(r"- \[[xX]\] (too vague|gives the answer away)", b))
    sc = {"file": os.path.basename(args.review), "reviewed": reviewed, "good": good, "vague": vague,
          "gives_away": give, "unreviewed": unrev, "reject_share": round(rej / reviewed, 4) if reviewed else None}
    mp = os.path.join(args.out, args.repo, "questions", f"{args.set}.manifest.json")
    m = json.load(open(mp))
    m["spot_check"] = sc
    with open(mp, "w") as fh:
        json.dump(m, fh, indent=1)
    print(json.dumps(sc))


if __name__ == "__main__":
    main()
