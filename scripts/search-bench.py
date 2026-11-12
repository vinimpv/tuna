"""Search cost of one embedding index: size on disk, query latency, CPU rebuild cost.

  .venv/bin/python scripts/search-bench.py --repo saleor --tag lfm25-ft-full --questions qwen-m2 --device cuda
  .venv/bin/python scripts/search-bench.py --repo saleor --tag lfm25-ft-full --questions qwen-m2 --device cpu

Portable: needs only this checkout's venv and the repo's out/ directory (vectors,
symbols.jsonl, question set; copy them as they are to another machine and set
TUNA_DATA, and --weights if the trained model lives elsewhere). Measures:

  index_bytes       the vector matrix, its owner index and metadata on disk
  query latency     per test question, one at a time as a user would search:
                    encode the question, score every chunk, best chunk per
                    symbol, top 10. p50 / p95 / mean in ms, encode and search
                    reported apart. After --warmup unmeasured queries.
  rebuild (cpu)     embedding a seeded random sample of --rebuild-sample symbols
                    (split like the index) on the CPU, extrapolated to the whole
                    index: documents/s and estimated seconds for a full rebuild.

Prints one JSON object and appends it to the run ledger (stage search-bench).
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tuna import chunking, common, encoders, metrics  # noqa: E402


def pick_device(name):
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True, help="vector tag under vectors/ (tuna embed --tag)")
    ap.add_argument("--questions", required=True, help="question set under questions/")
    ap.add_argument("--device", default="auto", help="cuda, cpu, mps or auto")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--rebuild-sample", type=int, default=64, help="symbols embedded on CPU to time a rebuild (0 = skip)")
    ap.add_argument("--weights", help="override the trained-model path stored with the vectors")
    ap.add_argument("--out", default=common.default_out())
    args = ap.parse_args()

    import numpy as np
    import torch
    device = pick_device(args.device)
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    vdir = os.path.join(work, "vectors")
    with open(os.path.join(vdir, f"{args.tag}.json")) as fh:
        meta = json.load(fh)
    if args.weights:
        meta["weights"], meta["weights_sha256"] = args.weights, None
    files = [os.path.join(vdir, f"{args.tag}{x}") for x in (".npy", "_owner.npy", ".json")]
    M = np.load(files[0])
    owner = np.load(files[1])
    syms = common.load_symbols(work)
    qs = common.read_jsonl(os.path.join(work, "questions", f"{args.questions}.jsonl"))
    out = {"repo": cfg["name"], "tag": args.tag, "model": meta["model"], "weights": metrics.portable(meta.get("weights")),
           "device": device, "machine": metrics.machine(), "questions_set": args.questions, "questions": len(qs),
           "documents": int(len(M)), "symbols": len(syms), "dim": int(M.shape[1]),
           "index_bytes": sum(os.path.getsize(f) for f in files)}

    with metrics.Stage("search-bench", cfg["name"], meta["model"], {**vars(args), "device": device}) as stage:
        enc = encoders.from_meta(meta, device=device)
        Mt = M  # numpy search: the same code path on every machine
        enc_ms, search_ms, total_ms = [], [], []
        for i, q in enumerate(qs[:args.warmup] + qs):
            sync(device)
            t0 = time.perf_counter()
            qv = enc.encode_queries([q["question"]])[0]
            sync(device)
            t1 = time.perf_counter()
            s = chunking.max_per_symbol(Mt @ qv, owner, len(syms))
            top = np.argpartition(-s, 10)[:10]
            top = top[np.argsort(-s[top])]
            t2 = time.perf_counter()
            if i >= args.warmup:
                enc_ms.append((t1 - t0) * 1e3)
                search_ms.append((t2 - t1) * 1e3)
                total_ms.append((t2 - t0) * 1e3)
        for name, xs in (("query_ms", total_ms), ("encode_ms", enc_ms), ("search_ms", search_ms)):
            out[name] = {"p50": round(pct(xs, 0.5), 2), "p95": round(pct(xs, 0.95), 2),
                         "mean": round(statistics.mean(xs), 2)}
        del enc
        if args.rebuild_sample:
            cpu = encoders.from_meta(meta, device="cpu")
            sample = random.Random(0).sample(range(len(syms)), min(args.rebuild_sample, len(syms)))
            texts, _, st = chunking.split_symbols([common.symbol_doc(syms[i]) for i in sample], cpu.count_tokens,
                                                  cpu.limit)
            t0 = time.perf_counter()
            cpu.encode_documents(texts)
            secs = time.perf_counter() - t0
            dps = len(texts) / secs
            out["cpu_rebuild"] = {"sample_symbols": len(sample), "sample_documents": len(texts),
                                  "sample_seconds": round(secs, 2), "documents_per_s": round(dps, 2),
                                  "estimated_full_rebuild_s": round(len(M) / dps, 1),
                                  "cpu": metrics.environment()["cpu"], "threads": torch.get_num_threads()}
        stage.counts(items=len(qs), questions=len(qs), documents=out["documents"], symbols=len(syms))
        stage.extra(**{k: v for k, v in out.items() if k not in ("repo", "model")})
        stage.outputs(*files)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
