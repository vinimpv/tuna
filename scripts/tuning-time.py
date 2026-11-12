"""Tuning time per model and repo, local card only, from the run ledger and the checkpoints.

  python3 -I scripts/tuning-time.py [--out $TUNA_OUT] [--ledger FILE] [--json OUT] [--extract-times FILE]

For each locked row (the LoRA rows) on each repo, from the local machine's ledger lines (tuna.metrics.is_local:
the local RTX 5090; lines from rented machines are skipped):
  mine      the row's "mine" stage: hard-negative mining with its own stock vectors, and the
            judge's false-negative pass inside it ("judge-encode" lines, cached per repo and set)
  setup     from the end of mining to the start of the "train" stage: loading the model and the
            stock validation pass (epoch 0); not a ledger stage of its own, so read off timestamps
  train     the "train" stage (= tuna-model.json train_seconds): every epoch's training steps plus
            the validation pass after each epoch (re-encoding the repo, scoring the val split);
            split into the two with the checkpoint's per-epoch history
  run       start of mining to end of training: one tuning run as a person would wait for it
Also, per repo, for the from-scratch sentence: qwen-train1 question writing (the manifest's run
wall seconds, local writer) and per row the re-indexing "embed" stage.

Cold start, every step in the order a user runs them (all local, wall seconds):
  clone         not measured (a network fetch; not in the ledger)
  extract       tuna extract; not in the ledger, so timed by --measure-extract into a scratch output
                directory (its symbols.jsonl must equal the real one) and kept in --extract-times
                (default $TUNA_OUT/tables/extract-time.json; not in this release)
  stock index   the model's own stock vectors (mining needs them): the "-remeasure" embed lines
                (stock vectors re-encoded on the local card, 2026-11-12)
  judge index   stock Qwen3-Embedding-0.6B vectors (the false-negative judge needs them, whichever
                model is tuned); for Qwen3 this is its own stock index, counted once
  questions, run (mining + judge pass + training with per-epoch validation), re-index: as above
Model loading inside a step is part of that step's wall time; model loading before mining is in no
ledger line and is not counted.
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common, metrics  # noqa: E402

REPOS = ["saleor", "zulip", "posthog"]
ROWS = {"LFM2.5-350M": ("lfm25-ft-lora", "LiquidAI/LFM2.5-Embedding-350M"),
        "jina-0.5B": ("jina-0.5b-ft-lora", "jinaai/jina-code-embeddings-0.5b"),
        "Qwen3-0.6B": ("qwen3-0.6b-ft-lora", "Qwen/Qwen3-Embedding-0.6B")}


def ts(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def local(l):
    return metrics.is_local(l) and l.get("status") == "ok"


def remeasure(L, repo, tag):
    """Wall seconds of the local stock re-measure embed line for tag (stock vectors re-encoded on the local card,
    2026-11-12), or None."""
    hits = [l for l in L if l["stage"] == "embed" and l.get("repo") == repo and local(l)
            and (l.get("config") or {}).get("tag") == f"{tag}-remeasure"]
    return round(hits[-1]["wall_seconds"], 1) if hits else None


def measure_extract(args):
    import subprocess
    import tempfile
    import time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    res = json.load(open(args.extract_times)) if os.path.exists(args.extract_times) else {}
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, TUNA_OUT=os.path.join(tmp, "out"), TUNA_METRICS=os.path.join(tmp, "metrics"),
                   PYTHONPATH=os.path.join(root, "src"))
        for repo in REPOS:
            t0 = time.time()
            subprocess.run([sys.executable, "-m", "tuna", "extract", "--repo", repo], env=env, check=True,
                           stdout=subprocess.DEVNULL)
            sec = time.time() - t0
            same = open(os.path.join(tmp, "out", repo, "symbols.jsonl"), "rb").read() == \
                open(os.path.join(args.out, repo, "symbols.jsonl"), "rb").read()
            if not same:
                sys.exit(f"tuning-time: re-extracted {repo} differs from the real symbols.jsonl; not recorded")
            res[repo] = {"seconds": round(sec, 1), "measured_utc": datetime.datetime.now(datetime.timezone.utc)
                         .isoformat(timespec="seconds"), "how": "tuna extract into a scratch TUNA_OUT, output identical"}
    os.makedirs(os.path.dirname(os.path.abspath(args.extract_times)), exist_ok=True)
    with open(args.extract_times, "w") as fh:
        json.dump(res, fh, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--ledger", default=os.path.join(common.metrics_root(),
        "ledger.jsonl"))
    ap.add_argument("--json")
    ap.add_argument("--extract-times", default=os.path.join(common.default_out(), "tables", "extract-time.json"))
    ap.add_argument("--measure-extract", action="store_true",
                    help="time tuna extract per repo into a scratch directory and update --extract-times")
    args = ap.parse_args()
    if args.measure_extract:
        measure_extract(args)
    ext = json.load(open(args.extract_times)) if os.path.exists(args.extract_times) else {}
    L = [json.loads(l) for l in open(args.ledger)]
    out = []
    for repo in REPOS:
        man = json.load(open(os.path.join(args.out, repo, "questions", "qwen-train1.manifest.json")))
        q_s = sum(r["wall_seconds"] for r in man.get("runs") or [])
        for model, (row, base) in ROWS.items():
            meta = json.load(open(os.path.join(args.out, repo, "models", row, "tuna-model.json")))
            tr = [l for l in L if l["stage"] == "train" and l.get("repo") == repo and local(l)
                  and (l.get("config") or {}).get("name") == row]
            # the checkpoint's own run: the train line whose wall time is its train_seconds
            tr = [l for l in tr if abs(l["wall_seconds"] - meta["train_seconds"]) < 1.0]
            if len(tr) != 1:
                out.append({"repo": repo, "model": model, "row": row, "error": f"{len(tr)} matching local train lines"})
                continue
            t = tr[0]
            t_start = ts(t["start_utc"])
            mines = [l for l in L if l["stage"] == "mine" and l.get("repo") == repo and local(l) and l.get("model") == base
                     and (l.get("config") or {}).get("questions") == "qwen-train1" and ts(l["start_utc"]) <= t_start]
            m = max(mines, key=lambda l: l["start_utc"])
            m_start, m_end = ts(m["start_utc"]), ts(m["end_utc"])
            hist = meta["history"][1:]
            ep_train = sum(h.get("train_seconds", 0) for h in hist)
            ep_val = sum(h.get("val_seconds", 0) for h in hist)
            emb = [l for l in L if l["stage"] == "embed" and l.get("repo") == repo and local(l)
                   and (l.get("config") or {}).get("tag") == row]
            judge = [l for l in L if l["stage"] == "judge-encode" and l.get("repo") == repo and local(l)
                     and (l.get("config") or {}).get("questions") == "qwen-train1"]
            stock_tag = {"LFM2.5-350M": "lfm25-emb", "jina-0.5B": "jina-0.5b", "Qwen3-0.6B": "qwen3-0.6b"}[model]
            stock = remeasure(L, repo, stock_tag)
            judge_idx = remeasure(L, repo, "qwen3-0.6b")
            out.append({
                "repo": repo, "model": model, "row": row, "gpu": meta.get("gpu"),
                "extract_s": ext.get(repo, {}).get("seconds"),
                "stock_index_s": stock, "judge_index_s": None if stock_tag == "qwen3-0.6b" else judge_idx,
                "mine_s": round(m["wall_seconds"], 1),
                "judge_s_once_per_repo": round(judge[0]["wall_seconds"], 1) if judge else None,
                "setup_s": round((t_start - m_end).total_seconds(), 1),
                "train_stage_s": round(t["wall_seconds"], 1),
                "epoch_training_s": round(ep_train, 1), "epoch_validation_s": round(ep_val, 1),
                "epochs_run": len(hist), "best_epoch": meta["best_epoch"],
                "run_s": round((ts(t["end_utc"]) - m_start).total_seconds(), 1),
                "index_s": round(emb[-1]["wall_seconds"], 1) if emb else None,
                "questions_s_per_repo": round(q_s, 1),
                "sources": {"train_line": t["start_utc"], "mine_line": m["start_utc"],
                            "checkpoint": f"{repo}/models/{row}/tuna-model.json",
                            "questions": f"{repo}/questions/qwen-train1.manifest.json runs"}})
    print(f"{'repo':8s} {'model':12s} {'run min':>7s} {'mine s':>6s} {'setup s':>7s} {'train min':>9s} "
          f"{'(steps':>7s} {'val)':>6s} {'epochs':>6s} {'index min':>9s} {'questions min':>13s}")
    for r in out:
        if "error" in r:
            print(r["repo"], r["model"], r["error"])
            continue
        print(f"{r['repo']:8s} {r['model']:12s} {r['run_s'] / 60:7.1f} {r['mine_s']:6.1f} {r['setup_s']:7.1f} "
              f"{r['train_stage_s'] / 60:9.1f} {r['epoch_training_s'] / 60:7.1f} {r['epoch_validation_s'] / 60:6.1f} "
              f"{r['epochs_run']:6d} {(r['index_s'] or 0) / 60:9.2f} {r['questions_s_per_repo'] / 60:13.1f}")
    print()
    print(f"{'repo':8s} {'model':12s} {'extract s':>9s} {'stock idx s':>11s} {'judge idx s':>11s} {'cold min':>8s}")
    for r in out:
        if "error" in r:
            continue
        parts = [r["extract_s"], r["stock_index_s"], r["judge_index_s"] or 0, r["questions_s_per_repo"],
                 r["run_s"], r["index_s"]]
        r["cold_start_s"] = round(sum(p for p in parts if p is not None), 1) if None not in parts[:2] else None
        print(f"{r['repo']:8s} {r['model']:12s} {r['extract_s'] or 0:9.1f} {r['stock_index_s'] or 0:11.1f} "
              f"{r['judge_index_s'] or 0:11.1f} {(r['cold_start_s'] or 0) / 60:8.1f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
