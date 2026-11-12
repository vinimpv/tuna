"""Held-out files: everything any test set touches, kept out of training.

  tuna heldout --repo saleor                                   # build and save the list
  tuna heldout --repo saleor --check qwen-train1               # verify a training set against it

A file is held out when it is the target file of a question in any test set,
or holds a gold symbol of one (every gold, for sets with several golds per
question). Training questions may not target a held-out file and no training
gold may live in one; tuna questions --exclude-files enforces this when
writing, and --check verifies a finished set again, against the saved list
and against the test sets themselves (gold FQNs and question texts too).

The list is written to <out>/<repo>/questions/heldout-files.txt (one path per
line, sorted) with heldout-files.json beside it (the sets used, their sha256,
per-set file counts).
"""
import json
import os
import sys

from . import common

DEFAULT_TEST_SETS = ("qwen-m1", "qwen-m2", "claude-m2", "claude-m3", "qwen-m4-400", "issues-v1", "issues-v1-noleak")


def golds(row):
    g = row.get("gold_fqns") or ([row["gold_fqn"]] if row.get("gold_fqn") else [])
    return list(g)


def heldout_files(test_rows, file_of):
    """Files touched by test rows. file_of maps FQN -> repo-relative file. Every gold
    must resolve: an unknown gold could hide a held-out file, so it is an error."""
    files, missing = set(), []
    for row in test_rows:
        if row.get("file"):
            files.add(row["file"])
        for g in golds(row):
            if g in file_of:
                files.add(file_of[g])
            else:
                missing.append(g)
    if missing:
        raise ValueError(f"{len(missing)} test golds are not in symbols.jsonl, e.g. {missing[:3]}")
    return files


def check_training(train_rows, held, test_rows, file_of):
    """Problems (empty list if none) of a training set against held-out files and test sets."""
    problems = []
    test_golds = {g for r in test_rows for g in golds(r)}
    test_questions = {r["question"].strip().lower() for r in test_rows}
    for r in train_rows:
        if r.get("file") in held:
            problems.append(f"{r['id']}: targets held-out file {r['file']}")
        for g in golds(r):
            if file_of.get(g) in held:
                problems.append(f"{r['id']}: gold {g} lives in held-out file {file_of[g]}")
            if g in test_golds:
                problems.append(f"{r['id']}: gold {g} is a test gold")
        if r["question"].strip().lower() in test_questions:
            problems.append(f"{r['id']}: question text equals a test question")
    return problems


def load_list(path):
    with open(path, encoding="utf-8") as fh:
        return {l.rstrip("\n") for l in fh if l.strip()}


def _paths(work):
    q = os.path.join(work, "questions")
    return os.path.join(q, "heldout-files.txt"), os.path.join(q, "heldout-files.json")


def main(args):
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    file_of = {s["fqn"]: s["file_path"] for s in common.load_symbols(work)}
    qdir = os.path.join(work, "questions")
    sets, rows_by_set = [], {}
    for name in args.sets:
        path = os.path.join(qdir, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"[heldout] {cfg['name']}: no {name}, skipped", flush=True)
            continue
        rows_by_set[name] = common.read_jsonl(path)
        sets.append({"name": name, "path": path, "rows": len(rows_by_set[name]), "sha256": common.file_sha256(path)})
    test_rows = [r for rows in rows_by_set.values() for r in rows]
    txt, meta = _paths(work)

    if args.check:
        held = load_list(txt)
        train_path = os.path.join(qdir, f"{args.check}.jsonl")
        train = common.read_jsonl(train_path)
        problems = check_training(train, held, test_rows, file_of)
        # the saved list must still cover the current test sets
        stale = heldout_files(test_rows, file_of) - held
        problems += [f"held-out list misses {f} (test sets changed; rebuild it)" for f in sorted(stale)]
        for p in problems[:20]:
            print(f"[heldout] LEAK {p}", flush=True)
        print(f"[heldout] {cfg['name']}: {args.check}: {len(train)} questions checked against {len(held)} "
              f"held-out files and {len(test_rows)} test questions: {len(problems)} problems", flush=True)
        sys.exit(1 if problems else 0)

    held = heldout_files(test_rows, file_of)
    for s in sets:
        s["files"] = len(heldout_files(rows_by_set[s["name"]], file_of))
    all_files = {s["file_path"] for s in common.load_symbols(work)}
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("".join(f + "\n" for f in sorted(held)))
    with open(meta, "w") as fh:
        json.dump({"repo": cfg["name"], "sets": sets, "heldout_files": len(held),
                   "repo_files_with_symbols": len(all_files), "symbols_sha256": common.file_sha256(
                       os.path.join(work, "symbols.jsonl"))}, fh, indent=1)
    print(f"[heldout] {cfg['name']}: {len(held)} of {len(all_files)} files held out "
          f"({', '.join(f'{s['name']} {s['files']}' for s in sets)}) -> {txt}", flush=True)


def add_parser(sub):
    p = sub.add_parser("heldout", help="list the files test sets touch (kept out of training), or --check a set")
    p.add_argument("--sets", nargs="+", default=list(DEFAULT_TEST_SETS), help="test sets under questions/")
    p.add_argument("--check", metavar="TRAIN_SET", help="verify this training set instead of building the list")
    p.set_defaults(func=main)
    return p
