"""Render a paired-bootstrap JSON file as the text table scripts/paired-bootstrap.py prints, without recomputing.

  python3 -I scripts/bootstrap-txt.py IN.json OUT.txt
"""
import json
import sys


def render(d):
    label = "MRR@10" if d.get("metric") == "mrr10" else "fhit@10"
    out = [f"{'repo':20s} {'set':17s} {'a':26s} {'b':20s} {'n':>4s} {label + ' a':>9s} {'b':>7s} {'diff':>7s}  95% CI"]
    for r in d["rows"]:
        out.append(f"{r['repo']:20s} {r['set']:17s} {r['a']:26s} {r['b']:20s} {r['n']:4d} {r['mrr10_a']:9.3f} {r['mrr10_b']:7.3f} "
                   f"{r['diff']:+7.3f}  [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]{'  sig' if r['sig'] else ''}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
    open(sys.argv[2], "w").write(render(json.load(open(sys.argv[1]))))
