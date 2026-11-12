"""The gold profile (tables/gold-profile.json) as a plain-text table, for reading.

  python3 -I scripts/gold-profile-render.py [--out $TUNA_OUT/checks/render-gold-profile.txt]

One row per repo and set, columns as stored: n, shares of classes / functions / methods, median lines, shares
with a docstring / a leading underscore / first in their file, mean relative position in the file. The set
"eligible" field says which baseline a set is drawn from (held-out files or training files).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = [("n", "n"), ("share_class", "class"), ("share_function", "func"), ("share_method", "method"),
        ("median_lines", "med.lines"), ("share_docstring", "docstr"), ("share_underscore", "_name"),
        ("share_first_in_file", "first"), ("mean_relative_position", "rel.pos")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(common.default_out(), "checks", "render-gold-profile.txt"))
    ap.add_argument("--data", default=os.path.join(common.default_out(),
                                                    "tables", "gold-profile.json"))
    args = ap.parse_args()
    rows = json.load(open(args.data))
    w = max(len(r["set"]) for r in rows)
    head = f"{'repo':8s} {'set':{w}s} {'from':9s} " + " ".join(f"{h:>9s}" for _, h in COLS)
    lines = ["GOLD PROFILE ($TUNA_OUT/tables/gold-profile.json): the gold symbols of each set against all symbols of the "
             "files it is drawn from", "", head, "-" * len(head)]
    last = None
    for r in rows:
        if last and r["repo"] != last:
            lines.append("")
        last = r["repo"]
        cells = " ".join(f"{(str(r[k]) if k in r else '-'):>9s}" for k, _ in COLS)
        lines.append(f"{r['repo']:8s} {r['set']:{w}s} {r.get('eligible', '-'):9s} {cells}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
