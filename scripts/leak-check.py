"""Held-out assertions over the final files: test golds outside, training golds and hard negatives inside.

  python3 -I scripts/leak-check.py [--out $TUNA_OUT] [--report $TUNA_OUT/checks/leak-check.txt]
                                   [--json $TUNA_OUT/checks/leak-check.json]

Per repo (Saleor, Zulip, PostHog), against questions/heldout-files.txt and symbols.jsonl:
  test sets      every gold symbol's file, and the question's own target file, must be in the held-out
                 list. claude-m3 is checked per batch (source_set claude-m2 and claude-m3-new), plus
                 qwen-m4-400, lfm3-t400 and the other test sets present.
  training sets  no gold symbol's file and no target file may be in the held-out list: qwen-train1
                 (train and val splits), qwen-bug1 and qwen-trainbug1 where they exist.
  negatives      every hard negative recorded in models/*.mined.jsonl (the negatives the run trained
                 on, by symbol FQN) must lie outside the held-out list. Re-mining rounds (rows with
                 -remine) draw from the same pool under the same mask but are not saved per question,
                 so only their first round is checked here.
All counts should be zero; exit 1 otherwise. Dispatch and Home Assistant are test-only (nothing
trains on them), so they have no held-out list and are not checked. Trained rows whose mined
negatives are not on this machine (rows trained on rented machines) are listed as "unchecked", not counted.
The mined file holds each question's whole candidate pool (mine depth 30, after the skips); the
negatives a step trains on are sampled from it, so the pool check covers them.
"""
import argparse
import glob
import json
import re
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPOS = ["saleor", "zulip", "posthog"]
# Mining commits of the rows trained on rented machines, whose mined-negatives files were not kept. The mining code
# at these commits is identical to that of the checked rows. (dev-NN: commits of the unpublished development history.)
RENTED_MINING_COMMITS = {"lfm25-ft-lora-s1": "dev-13", "lfm25-ft-lora-s2": "dev-13",
                         "jina-0.5b-ft-lora-s1": "dev-25", "jina-0.5b-ft-lora-s2": "dev-25",
                         "jina-0.5b-ft-lora-4090": "dev-25"}
TEST = [("claude-m3 batch 1 (claude-m2)", "claude-m3", "claude-m2"),
        ("claude-m3 batch 2 (claude-m3-new)", "claude-m3", "claude-m3-new"),
        ("qwen-m4-400", "qwen-m4-400", None), ("lfm3-t400", "lfm3-t400", None),
        ("qwen-m1", "qwen-m1", None), ("qwen-m2", "qwen-m2", None),
        ("issues-v1", "issues-v1", None), ("issues-v1-noleak", "issues-v1-noleak", None)]
TRAIN = [("qwen-train1 train", "qwen-train1.train"), ("qwen-train1 val", "qwen-train1.val"),
         ("qwen-bug1", "qwen-bug1"), ("qwen-trainbug1", "qwen-trainbug1")]


def released_row(w, row):
    """A trained row counts when its training set is one of TRAIN's (qwen-train1, its subsets, qwen-trainbug1); rows
    trained on any other set are not checked or counted."""
    p = os.path.join(w, "models", row, "tuna-model.json")
    if not os.path.exists(p):
        return True
    st = re.sub(r"-n\d+$", "", json.load(open(p)).get("questions") or "")
    return st in {"qwen-train1", "qwen-trainbug1"}


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def golds(q):
    return q.get("gold_fqns") or ([q["gold_fqn"]] if q.get("gold_fqn") else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--report", default=os.path.join(common.default_out(), "checks", "leak-check.txt"))
    ap.add_argument("--json", default=os.path.join(common.default_out(), "checks", "leak-check.json"))
    args = ap.parse_args()
    result, lines, bad = [], [], 0
    for repo in REPOS:
        w = os.path.join(args.out, repo)
        held = {l.strip() for l in open(os.path.join(w, "questions", "heldout-files.txt")) if l.strip()}
        file_of = {}
        for l in open(os.path.join(w, "symbols.jsonl")):
            s = json.loads(l)
            file_of[s["fqn"]] = s["file_path"]
        lines.append(f"== {repo}: {len(held)} held-out files")
        # timeline: first test sets written -> list in use -> training sets started (manifest times)
        import hashlib, datetime
        sha = hashlib.sha256(open(os.path.join(w, "questions", "heldout-files.txt"), "rb").read()).hexdigest()
        hp = os.path.join(w, "questions", "heldout-files.json")   # not in the release: its checks report None
        hmeta = json.load(open(hp)) if os.path.exists(hp) else None
        def mstart(st):
            mp = os.path.join(w, "questions", f"{st}.manifest.json")
            if not os.path.exists(mp):
                return None, None
            m = json.load(open(mp))
            return ((m.get("runs") or [{}])[0].get("started") or m.get("created") or m.get("updated")), m
        first = ["qwen-m1", "qwen-m2", "claude-m2", "issues-v1"]   # issues-v1-noleak is a subset of issues-v1
        recomputed = set()
        for st in first + ["issues-v1-noleak"]:
            for q in rows(os.path.join(w, "questions", f"{st}.jsonl")) or []:
                if q.get("file"):
                    recomputed.add(q["file"])
                recomputed |= {file_of[g] for g in golds(q) if g in file_of}
        t_first = max(mstart(st)[0] for st in first)
        users = []
        for st in ("claude-m3-new", "lfm3-t400", "qwen-train1", "qwen-bug1"):
            t, m = mstart(st)
            if m and sha in (m.get("only_files_sha256"), m.get("exclude_files_sha256")):
                users.append((t, st))
        t_list = min(users)[0] if users else None
        trains = [(mstart(st)[0], st) for st in ("qwen-train1", "qwen-bug1") if mstart(st)[1]]
        t_train = min(trains)[0]
        rebuilt = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(w, "questions", "heldout-files.txt")),
                                                  datetime.timezone.utc).isoformat(timespec="minutes")
        order_ok = t_first[:16] <= t_list[:16] <= t_train[:16] and recomputed == held
        bad += 0 if order_ok else 1
        result.append({"repo": repo, "kind": "timeline", "first_test_sets_written_by": t_first,
                       "list_in_use_by": t_list, "list_first_user": min(users)[1], "training_started": t_train,
                       "first_training_set": min(trains)[1], "list_file_last_written": rebuilt,
                       "list_equals_files_of_first_test_sets": recomputed == held,
                       "sets_recorded_in_list_metadata": [x["name"] for x in hmeta["sets"]] if hmeta else None, "order_ok": order_ok})
        lines.append(f"   timeline  first test sets ({', '.join(first)}) written by {t_first[:16]}; list in use by "
                     f"{t_list[:16]} ({min(users)[1]} records its hash); training started {t_train[:16]} "
                     f"({min(trains)[1]}); list file last rewritten {rebuilt[:16]} (same content); list == files of "
                     f"the first test sets: {recomputed == held}{'' if order_ok else '  <-- ORDER PROBLEM'}")

        def record(kind, name, n_q, n_g, leaks, unknown, source):
            nonlocal bad
            bad += leaks + unknown
            result.append({"repo": repo, "kind": kind, "set": name, "questions": n_q, "golds_or_negatives": n_g,
                           "leaks": leaks, "unresolved": unknown, "source": source})
            lines.append(f"   {kind:9s} {name:34s} {n_q:6d} questions {n_g:7d} checked  leaks {leaks}"
                         + (f"  unresolved {unknown}" if unknown else ""))

        for label, st, src in TEST:
            qs = rows(os.path.join(w, "questions", f"{st}.jsonl"))
            if qs is None:
                continue
            if src:
                qs = [q for q in qs if q.get("source_set") == src]
            gs = [g for q in qs for g in golds(q)]
            unknown = sum(1 for g in gs if g not in file_of)
            out = sum(1 for g in gs if g in file_of and file_of[g] not in held)
            out += sum(1 for q in qs if q.get("file") and q["file"] not in held)
            record("test", label, len(qs), len(gs), out, unknown, f"{repo}/questions/{st}.jsonl")
        for label, st in TRAIN:
            qs = rows(os.path.join(w, "questions", f"{st}.jsonl"))
            if qs is None:
                continue
            gs = [g for q in qs for g in golds(q)]
            unknown = sum(1 for g in gs if g not in file_of)
            inside = sum(1 for g in gs if file_of.get(g) in held)
            inside += sum(1 for q in qs if q.get("file") in held)
            record("training", label, len(qs), len(gs), inside, unknown, f"{repo}/questions/{st}.jsonl")
        for p in sorted(glob.glob(os.path.join(w, "models", "*.mined.jsonl"))):
            row = os.path.basename(p)[:-len(".mined.jsonl")]
            if not released_row(w, row):
                continue
            mined = rows(p)
            negs = [n for m in mined for n in m.get("negatives", [])]
            unknown = sum(1 for n in negs if n not in file_of)
            inside = sum(1 for n in negs if file_of.get(n) in held)
            record("negatives", row, len(mined), len(negs), inside, unknown, f"{repo}/models/{row}.mined.jsonl")
        # trained rows whose negatives were not saved on this machine (trained elsewhere): listed, not checked
        mined_rows = {os.path.basename(p)[:-len(".mined.jsonl")] for p in glob.glob(os.path.join(w, "models", "*.mined.jsonl"))}
        dirs = glob.glob(os.path.join(w, "models", "*", "tuna-model.json")) + \
            [p for p in glob.glob(os.path.join(args.out, "pooled", "models", "*", "tuna-model.json"))]
        for d in sorted(dirs):
            row = os.path.basename(os.path.dirname(d))
            if row not in mined_rows and released_row(os.path.dirname(os.path.dirname(os.path.dirname(d))), row):
                meta = json.load(open(d))
                where = meta.get("machine") or ("local" if not meta.get("gpu") else meta.get("gpu"))
                commit = RENTED_MINING_COMMITS.get(row)
                result.append({"repo": repo, "kind": "unchecked", "set": row, "reason": "mined file lost with the machine"
                               if commit else "no mined.jsonl here", "trained_on": where, "mining_commit": commit})
                lines.append(f"   unchecked {row:34s} no saved negatives on this machine (trained on: {where}; "
                             f"mining commit {commit or 'unknown'})")
    lines.append(f"RESULT: {'PASS (all zero)' if not bad else f'FAIL ({bad} leaks or unresolved symbols)'}")
    text = (f"leak-check report (scripts/leak-check.py), held-out lists: <repo>/questions/heldout-files.txt\n"
            + "\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    open(args.report, "w").write(text)
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=1)
    print(text, end="")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
