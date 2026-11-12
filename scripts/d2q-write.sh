#!/usr/bin/env bash
# Doc2query control, writing side: one Qwen question per symbol of REPO's index (every symbol,
# held-out files included), the gold-targeted twin of qwen-train1's prompt (question-v2-gold, same v2
# filter, 3 attempts, 90 s), written with tuna questions --gold-list. Index-side output only:
# the sets are named index-* and `tuna train` refuses them.
#
# --gold-list stops a file at its first slot that fails all attempts, so the run goes in passes:
# pass 1 asks for every symbol (largest files first, so the longest per-file sequences start early);
# each later pass asks only for the symbols no earlier pass reached. Symbols that failed all attempts
# are not retried. The accepted questions of all passes are merged into questions/index-d2q.jsonl.
#
#   scripts/d2q-write.sh REPO          (needs the question-writer server running; see scripts/pi-ask)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=${1:?usage: d2q-write.sh REPO [SHARD NSHARDS | --merge]}
# Shards: with SHARD k of NSHARDS, only symbols with int(sha256(fqn), 16) % NSHARDS == k are asked, as their
# own sets (index-d2q-s<k>of<n>-p<pass>), so each shard has one writer engine. --merge only merges.
SHARD=${2:-} NSHARDS=${3:-}
MERGE_ONLY=0; [ "$SHARD" = --merge ] && { MERGE_ONLY=1; SHARD=; }
PREFIX=index-d2q${SHARD:+-s${SHARD}of${NSHARDS}}
OUT=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}
Q=$OUT/$REPO/questions
QWORKERS=${QWORKERS:-8}   # parallel writers; match your server's concurrency
PY=$PWD/.venv/bin/python
log() { echo "=== d2q $REPO $* $(date -u +%H:%M:%S)"; }
for pass in 1 2 3 4 5 6 7 8; do
  [ "$MERGE_ONLY" = 1 ] && break
  name=$PREFIX-p$pass
  list=$Q/$name.golds.jsonl
  n=$(python3 -I - "$OUT/$REPO/symbols.jsonl" "$Q" "$pass" "$list" "$PREFIX" "$SHARD" "$NSHARDS" <<'PY'
import collections, hashlib, json, os, sys
syms_path, qdir, npass, out, prefix = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
shard, nshards = (int(sys.argv[6]), int(sys.argv[7])) if sys.argv[6] else (None, None)
syms = [json.loads(l) for l in open(syms_path)]
by_file = collections.defaultdict(list)
for s in syms:
    if s["kind"] in ("Function", "Method", "Class"):
        if shard is not None and int(hashlib.sha256(s["fqn"].encode()).hexdigest(), 16) % nshards != shard:
            continue
        by_file[s["file_path"]].append(s["fqn"])
reached = set()
for k in range(1, npass):
    p = os.path.join(qdir, f"{prefix}-p{k}.progress.jsonl")
    for l in open(p):
        r = json.loads(l)
        if "gold_slots" not in r:
            continue
        slots = {a["slot"] for a in r.get("attempts", [])}
        reached |= {g for i, g in enumerate(r["gold_slots"]) if i + 1 in slots}
files = sorted(by_file, key=lambda f: (-len(by_file[f]), f))
n = 0
if os.path.exists(out) and npass > 1:
    pass  # rebuilt each time from the earlier passes, below
with open(out, "w") as fh:
    for f in files:
        for g in by_file[f]:
            if g not in reached:
                fh.write(json.dumps({"file": f, "gold_fqn": g}) + "\n")
                n += 1
print(n)
PY
)
  if [ "$n" = 0 ]; then log "pass $pass: every symbol reached"; break; fi
  log "pass $pass: $n symbols"
  PYTHONPATH=src "$PY" -m tuna questions --repo "$REPO" --name "$name" \
    --prompt question-v2-gold --gold-list "$list" --n "$n" --writer pi --workers "$QWORKERS" \
    --timeout 90 --max-tool-calls 40 ${ASK:+--ask "$ASK"}   # ASK: a stand-in writer, smoke tests only
done
python3 -I - "$OUT/$REPO/symbols.jsonl" "$Q" <<'PY'
import glob, json, os, re, sys
syms_path, qdir = sys.argv[1:3]
n_syms = sum(1 for _ in open(syms_path))
rows, seen, passes = [], set(), []
PASS = re.compile(r"index-d2q-(?:s\d+of\d+-)?p(\d+)\.jsonl$")
files = [p for p in glob.glob(os.path.join(qdir, "index-d2q-*.jsonl")) if PASS.search(os.path.basename(p))]
for p in sorted(files, key=lambda x: (int(PASS.search(os.path.basename(x)).group(1)), x)):
    k = 0
    for l in open(p):
        q = json.loads(l)
        if q["gold_fqn"] in seen:
            continue
        seen.add(q["gold_fqn"])
        q["id"] = f"index-d2q-{len(rows) + 1:05d}"
        q["index_side"] = True
        rows.append(q)
        k += 1
    passes.append({"set": os.path.basename(p)[:-6], "accepted": k})
with open(os.path.join(qdir, "index-d2q.jsonl"), "w") as fh:
    for q in rows:
        fh.write(json.dumps(q) + "\n")
m = {"name": "index-d2q", "index_side": True, "note": "doc2query control: index-side questions, never training data",
     "symbols": n_syms, "questions": len(rows), "fallback_symbols": n_syms - len(rows), "passes": passes}
json.dump(m, open(os.path.join(qdir, "index-d2q.manifest.json"), "w"), indent=1)
print(json.dumps(m))
PY
log "done"
