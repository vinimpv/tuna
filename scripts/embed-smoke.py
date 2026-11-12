"""GPU smoke test: load each stock embedder at a pinned revision and embed a
handful of symbols from symbols.jsonl plus one query.

  .venv/bin/python scripts/embed-smoke.py --repo saleor [--n 10] [--models qwen3-0.6b jina-0.5b]

Models download into $HF_HOME (default $TUNA_DATA/hf, $TUNA_DATA defaulting to ~/tuna-data). Only models that need
no remote code are listed here; the Liquid models, which do, are smoke-tested
by scripts/liquid-smoke.py through tuna.pinned.
"""
import argparse
import os
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser(os.environ.get("TUNA_DATA", "~/tuna-data")), "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# name -> (HF repo, pinned commit, query prompt). Revisions also in the README.
MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                   "Instruct: Given a developer's question about a codebase, retrieve the code that answers it\nQuery:"),
    "jina-0.5b": ("jinaai/jina-code-embeddings-0.5b", "4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
                  "Find the most relevant code snippet given the following query:\n"),
    "jina-1.5b": ("jinaai/jina-code-embeddings-1.5b", "39aeb4fb9b60f930934c78ae5d749a46287c248a",
                  "Find the most relevant code snippet given the following query:\n"),
}
PASSAGE_PREFIX = {"jina-0.5b": "Candidate code snippet:\n", "jina-1.5b": "Candidate code snippet:\n"}


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    from tuna import common, pinned

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="saleor")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--query", default="How long can an anonymous shopping cart sit untouched "
                                        "before a background job removes it?")
    args = ap.parse_args()

    cfg = common.load_config(args.repo)
    syms = common.load_symbols(common.workdir(common.default_out(), cfg))
    # the first n symbols plus the gold symbol for the default query
    picked = syms[:args.n - 1] + [s for s in syms if s["fqn"] == "saleor.checkout.tasks.delete_expired_checkouts"][:1]
    docs = [common.symbol_doc(s, 4000) for s in picked]
    print(f"{len(docs)} symbols from {cfg['name']}; GPU: {torch.cuda.get_device_name(0)}")
    for name in args.models:
        repo, rev, qprompt = MODELS[name]
        pinned.refuse_unpinned(repo)
        t0 = time.time()
        model = SentenceTransformer(repo, revision=rev, device="cuda", trust_remote_code=False,
                                    model_kwargs={"dtype": torch.bfloat16})
        load = time.time() - t0
        D = model.encode([PASSAGE_PREFIX.get(name, "") + d for d in docs], normalize_embeddings=True,
                         convert_to_tensor=True)
        Q = model.encode([qprompt + args.query], normalize_embeddings=True, convert_to_tensor=True)
        sims = (Q @ D.T)[0].float()
        best = int(sims.argmax())
        print(f"{name:10s} {repo}@{rev[:10]} load {load:.1f}s dim {D.shape[1]} device {D.device} "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB top1 {picked[best]['fqn']} "
              f"({sims[best]:.3f})")
        del model, D, Q
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
