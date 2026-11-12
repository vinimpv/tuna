"""Merge training question sets into one, keeping validation files out of training.

  .venv/bin/python scripts/merge-sets.py --repo saleor --sets qwen-train1 qwen-bug1 --name qwen-trainbug1

Reads <set>.train.jsonl and <set>.val.jsonl of each set (tuna questions
--val-fraction). A file that is in any set's validation split goes to the
merged validation split with all its questions, so no file's code is trained
on in one set and validated on in another. Writes <name>.train.jsonl,
<name>.val.jsonl, <name>.jsonl (every question, with "split" and "source_set")
and <name>.manifest.json, then a ledger line.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tuna import common, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=common.default_out())
    args = ap.parse_args()
    cfg = common.load_config(args.repo)
    qdir = os.path.join(common.workdir(args.out, cfg), "questions")
    with metrics.Stage("merge-sets", cfg["name"], None, vars(args)) as stage:
        rows, val_files = [], set()
        for s in args.sets:
            for split in ("train", "val"):
                for r in common.read_jsonl(os.path.join(qdir, f"{s}.{split}.jsonl")):
                    rows.append(dict(r, source_set=s, source_split=split))
                    if split == "val":
                        val_files.add(r["file"])
        seen, out = set(), []
        for r in rows:
            key = (r["gold_fqn"], r["question"])
            if key in seen:
                continue
            seen.add(key)
            r["split"] = "val" if r["file"] in val_files else "train"
            out.append(r)
        for split in ("train", "val"):
            common.write_jsonl(os.path.join(qdir, f"{args.name}.{split}.jsonl"), [r for r in out if r["split"] == split])
        common.write_jsonl(os.path.join(qdir, f"{args.name}.jsonl"), out)
        moved = sum(1 for r in out if r["split"] != r["source_split"])
        man = {"name": args.name, "sets": args.sets, "questions": len(out),
               "splits": {sp: {"questions": sum(r["split"] == sp for r in out),
                               "files": len({r["file"] for r in out if r["split"] == sp})} for sp in ("train", "val")},
               "moved_to_val_by_file": moved,
               "by_source": dict(collections.Counter(f"{r['source_set']}:{r['split']}" for r in out))}
        with open(os.path.join(qdir, f"{args.name}.manifest.json"), "w") as fh:
            json.dump(man, fh, indent=1)
        stage.counts(items=len(out), questions=len(out), moved_to_val=moved)
        stage.extra(**man)
        stage.outputs(*(os.path.join(qdir, f"{args.name}{x}") for x in (".train.jsonl", ".val.jsonl", ".jsonl")))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
