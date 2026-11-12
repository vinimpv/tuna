"""Gold profile: what kind of symbols each question set picks as gold, against all symbols of its eligible files.

  .venv/bin/python scripts/gold-profile.py [--json OUT]

Per repo and set (qwen-train1, claude-m2 and claude-m3-new (the two claude-m3 batches), lfm3-t400, issues-v1) and
per baseline (all Function/Method/Class symbols in the eligible files: the training files for qwen-train1, the
held-out files for the test sets): share of classes, functions and methods; median length in lines; share with a
docstring; share whose name starts with an underscore; share that are their file's first symbol; mean relative
position in the file (0 = first symbol, 1 = last). n counts gold symbols (a question with several golds counts each),
or for a baseline all symbols. claude-m3-new is labelled "claude-m3 (300 questions not in claude-m2)".
"""
import argparse
import json

SET_LABELS = {"claude-m3-new": "claude-m3 (300 questions not in claude-m2)"}
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

OUT = common.default_out()
KINDS = ("Class", "Function", "Method")
DOC = re.compile(r'^\s*[rRuUbB]{0,2}("""|\'\'\'|"|\')')


def has_doc(body):
    lines = [l for l in body.splitlines() if not l.startswith("# ")]  # drop the FQN/File header
    i = 0
    while i < len(lines) and not lines[i].rstrip().endswith(":"):  # signature (may span lines)
        i += 1
    for l in lines[i + 1:]:
        if l.strip():
            return bool(DOC.match(l))
    return False


def profile(syms, pos):
    n = len(syms)
    if not n:
        return None
    return {"n": n, **{f"share_{k.lower()}": round(sum(s["kind"] == k for s in syms) / n, 3) for k in KINDS},
            "median_lines": statistics.median(s["line_end"] - s["line_start"] + 1 for s in syms),
            "share_docstring": round(sum(has_doc(s["body"]) for s in syms) / n, 3),
            "share_underscore": round(sum(s["fqn"].rsplit(".", 1)[-1].startswith("_") for s in syms) / n, 3),
            "share_first_in_file": round(sum(pos[s["fqn"]][0] == 0 for s in syms) / n, 3),
            "mean_relative_position": round(statistics.mean(pos[s["fqn"]][1] for s in syms), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=os.path.join(OUT, "tables", "gold-profile.json"))
    args = ap.parse_args()
    rows = []
    for repo in ["saleor", "zulip", "posthog"]:
        syms = [s for s in (json.loads(l) for l in open(os.path.join(OUT, repo, "symbols.jsonl"))) if s["kind"] in KINDS]
        by_fqn = {s["fqn"]: s for s in syms}
        by_file = {}
        for s in syms:
            by_file.setdefault(s["file_path"], []).append(s)
        pos = {}
        for f, ss in by_file.items():
            ss = sorted(ss, key=lambda s: (s["line_start"], s["fqn"]))
            for i, s in enumerate(ss):
                pos[s["fqn"]] = (i, i / (len(ss) - 1) if len(ss) > 1 else 0.0)
        q = os.path.join(OUT, repo, "questions")
        held = {l.strip() for l in open(os.path.join(q, "heldout-files.txt")) if l.strip()}
        base = {"heldout": [s for s in syms if s["file_path"] in held], "training": [s for s in syms if s["file_path"] not in held]}
        for name, b in base.items():
            rows.append({"repo": repo, "set": f"baseline: all symbols of the {name} files", "n_counts": "symbols",
                         **profile(b, pos)})
        for st, elig in (("qwen-train1", "training"), ("claude-m2", "heldout"), ("claude-m3-new", "heldout"),
                         ("lfm3-t400", "heldout"), ("issues-v1", "heldout")):
            p = os.path.join(q, f"{st}.jsonl")
            if not os.path.exists(p):
                continue
            golds = []
            for l in open(p):
                x = json.loads(l)
                for g in x.get("gold_fqns") or [x["gold_fqn"]]:
                    if g in by_fqn:
                        golds.append(by_fqn[g])
            rows.append({"repo": repo, "set": SET_LABELS.get(st, st), "eligible": elig, "n_counts": "gold symbols",
                         **profile(golds, pos)})
    keys = ["n", "share_class", "share_function", "share_method", "median_lines", "share_docstring", "share_underscore",
            "share_first_in_file", "mean_relative_position"]
    print(f"{'repo':8s} {'set':44s} " + " ".join(f"{k[:10]:>10s}" for k in keys))
    for r in rows:
        print(f"{r['repo']:8s} {r['set'][:44]:44s} " + " ".join(f"{r[k]:>10}" for k in keys))
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
