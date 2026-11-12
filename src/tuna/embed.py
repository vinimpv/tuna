"""Embed every symbol with a single-vector encoder (see encoders.py).

Each symbol is read at the model's own maximum length; a symbol over it is
split into header-prefixed chunks on line boundaries (chunking.py), and the
split is counted and logged, never silent. Writes, under <work>/vectors/:

  <tag>.npy        one L2-normalised row per chunk
  <tag>_owner.npy  the symbols.jsonl row each chunk belongs to
  <tag>.json       encoder settings, chunk counts, symbols_sha256 (a stale
                   matrix is refused later)
"""
import json
import os
import time

from . import chunking, common, encoders, metrics


def _main(args, stage):
    import numpy as np
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    syms = common.load_symbols(work)
    gpu = _gpu_start()
    enc = encoders.from_args(args)
    texts, owner, stats = chunking.split_symbols([common.symbol_doc(s) for s in syms], enc.count_tokens, enc.limit)
    print(chunking.describe(stats, cfg["name"], args.model), flush=True)
    t0 = time.time()
    M = enc.encode_documents(texts)
    embed_seconds = round(time.time() - t0, 1)
    vdir = os.path.join(work, "vectors")
    os.makedirs(vdir, exist_ok=True)
    np.save(os.path.join(vdir, f"{args.tag}.npy"), M)
    np.save(os.path.join(vdir, f"{args.tag}_owner.npy"), np.asarray(owner, dtype=np.int32))
    meta = {"tag": args.tag, **enc.meta(), "chunking": stats._asdict(), "symbols": len(syms),
            "documents": len(texts), "dim": int(M.shape[1]), "embed_seconds": embed_seconds, **_gpu_peak(gpu),
            "symbols_sha256": common.file_sha256(os.path.join(work, "symbols.jsonl"))}
    with open(os.path.join(vdir, f"{args.tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    stage.counts(items=len(texts), symbols=len(syms), chunks=len(texts), split_symbols=stats.split)
    stage.extra(embed_seconds=embed_seconds, limit=stats.limit, weights=meta.get("weights"))
    stage.outputs(*(os.path.join(vdir, f"{args.tag}{x}") for x in (".npy", "_owner.npy", ".json")))
    print(f"[embed] wrote vectors/{args.tag}.npy {M.shape}; {embed_seconds}s, "
          f"peak {meta.get('peak_vram_gib')} GiB on {meta.get('gpu')}", flush=True)


def _gpu_start():
    """Reset the CUDA peak-memory counter when the GPU is in use (local backend)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except ImportError:
        pass
    return False


def _gpu_peak(gpu):
    if not gpu:
        return {}
    import torch
    return {"gpu": torch.cuda.get_device_name(0),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def main(args):
    config = metrics.config_of(args)
    if getattr(args, "backend", None) != "served":   # the URL is only used (and recorded) for a served model
        config.pop("url", None)
    with metrics.Stage('embed', args.repo, getattr(args, "model", None), config) as stage:
        return _main(args, stage)


def add_parser(sub):
    p = sub.add_parser("embed", help="embed symbols (long ones split at the model's limit)")
    encoders.add_arguments(p)
    p.add_argument("--tag", required=True, help="vector file name, e.g. jina05b-stock")
    p.set_defaults(func=main)
    return p
