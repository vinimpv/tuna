#!/usr/bin/env python3
"""Concatenate question sets into one, ids renumbered in order (a larger test set that keeps a smaller one).

  scripts/merge-question-sets.py --repo saleor --name claude-m3 claude-m2 claude-m3-new

Rows keep every field; "id" becomes <repo>-<name>-NNN and the old id and set go to
"source_id" and "source_set". Writes <name>.jsonl, <name>.rejections.jsonl (each
source's rejections, tagged with source_set) and <name>.manifest.json (per-source
manifests, counts, gold uniqueness). Stops if a gold appears twice.
"""
import argparse
import collections
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402


def merged_cost(manifests):
    """The merged set's writer cost: the sum over its sources when every source recorded one; otherwise None, with
    the recorded part and the sources that recorded none."""
    missing = [n for n, m in manifests.items() if m.get("cost_usd") is None]
    recorded = round(sum(m.get("cost_usd") or 0 for m in manifests.values()), 4)
    if not missing:
        return {"cost_usd": recorded}
    return {"cost_usd": None, "cost_usd_recorded": recorded,
            "cost_note": f"no writer cost recorded for {', '.join(missing)} (written before tuna questions recorded "
                         f"it); cost_usd_recorded covers the other sources only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args()
    cfg = common.load_config(args.repo)
    qdir = os.path.join(common.workdir(common.default_out(), cfg), "questions")
    rows, rejections, manifests = [], [], {}
    for src in args.sources:
        for r in common.read_jsonl(os.path.join(qdir, f"{src}.jsonl")):
            rows.append(dict(r, source_id=r["id"], source_set=src))
        rp = os.path.join(qdir, f"{src}.rejections.jsonl")
        if os.path.exists(rp):
            rejections += [dict(r, source_set=src) for r in common.read_jsonl(rp)]
        with open(os.path.join(qdir, f"{src}.manifest.json")) as fh:
            m = json.load(fh)
        manifests[src] = {k: v for k, v in m.items() if k != "surplus_accepted"}
    dup = [g for g, c in collections.Counter(r["gold_fqn"] for r in rows).items() if c > 1]
    if dup:
        sys.exit(f"{len(dup)} golds appear twice, e.g. {dup[:3]}")
    width = max(3, len(str(len(rows))))
    for k, r in enumerate(rows, 1):
        r["id"] = f"{cfg['name']}-{args.name}-{k:0{width}d}"
    out = os.path.join(qdir, f"{args.name}.jsonl")
    common.write_jsonl(out, rows)
    common.write_jsonl(os.path.join(qdir, f"{args.name}.rejections.jsonl"), rejections)
    manifest = {"repo": cfg["name"], "name": args.name, "merged_from": args.sources,
                "accepted": len(rows), "per_source": dict(collections.Counter(r["source_set"] for r in rows)),
                "files_with_questions": len({r["file"] for r in rows}), "unique_golds": len({r["gold_fqn"] for r in rows}),
                "rejections": dict(collections.Counter(r["reason"] for r in rejections).most_common()),
                **merged_cost(manifests),
                "sources": manifests, "sha256": common.file_sha256(out),
                "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(qdir, f"{args.name}.manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[merge] {cfg['name']}: {args.name}: {len(rows)} questions from {manifest['per_source']} -> {out}")


if __name__ == "__main__":
    main()
