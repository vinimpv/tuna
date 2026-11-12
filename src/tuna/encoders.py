"""Single-vector encoders for `tuna embed` / `tuna dense`.

Each encoder knows its model's input limit and counts tokens exactly as the
model will see a document (prompt and special tokens included), so
chunking.split_symbols can split long symbols instead of the model truncating
them.

  served  an OpenAI-compatible /v1/embeddings endpoint (vLLM, TEI, ...). Token
          counts come from a local copy of the served model's tokenizer
          (--tokenizer, pinned with --tokenizer-revision) and the limit is
          --max-tokens, the server's maximum input length.
  local   sentence-transformers on this machine; the limit is the model's own
          max_seq_length. Liquid models load through tuna.pinned.
"""
import os
import sys
import time

from . import chunking, common, pinned


class ServedEncoder:
    backend = "served"

    def __init__(self, model, url, tokenizer, tokenizer_revision, max_tokens, prefix, query_prefix,
                 api_key_env=None, batch=32):
        from transformers import AutoTokenizer
        if not max_tokens:
            raise SystemExit("--backend served needs --max-tokens (the server's maximum input length)")
        if not tokenizer:
            raise SystemExit("--backend served needs --tokenizer (the served model's HF id or path) to count tokens")
        pinned.refuse_unpinned(tokenizer)
        self.model, self.url, self.batch = model, url, batch
        self.prefix, self.query_prefix, self.limit = prefix, query_prefix, max_tokens
        self.api_key_env = api_key_env
        self.tokenizer_id, self.tokenizer_revision = tokenizer, tokenizer_revision
        self._tok = AutoTokenizer.from_pretrained(tokenizer, revision=tokenizer_revision)
        self.count_tokens = chunking.token_counter(self._tok, prefix)

    def _post(self, texts, label):
        import numpy as np
        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # similar lengths per batch
        out, t0 = None, time.time()
        for s in range(0, len(order), self.batch):
            part = order[s:s + self.batch]
            data = common.post_json(self.url, {"model": self.model, "input": [texts[i] for i in part]},
                                    api_key=api_key)["data"]
            data.sort(key=lambda e: e.get("index", 0))
            if out is None:
                out = np.zeros((len(texts), len(data[0]["embedding"])), dtype=np.float32)
            for i, e in zip(part, data):
                out[i] = e["embedding"]
            if (s // self.batch) % 100 == 0:
                print(f"[{label}] {s + len(part)}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
        return _normalize(out)

    def encode_documents(self, texts):
        return self._post([self.prefix + t for t in texts], "embed")

    def encode_queries(self, texts):
        return self._post([self.query_prefix + t for t in texts], "query")

    def meta(self):
        return {"backend": self.backend, "model": self.model, "url": self.url, "tokenizer": self.tokenizer_id,
                "tokenizer_revision": self.tokenizer_revision, "max_tokens": self.limit, "prefix": self.prefix,
                "query_prefix": self.query_prefix, "api_key_env": self.api_key_env, "batch": self.batch}


class LocalEncoder:
    backend = "local"

    def __init__(self, model, revision=None, device="cuda", prefix=None, query_prefix=None, batch=32,
                 token_budget=65536, weights=None, max_tokens=None):
        self.weights, self.weights_sha256 = weights, None
        if model in pinned.pins():
            self._st = pinned.load(model, device=device)
            revision = pinned.pin_for(model).sha
            if weights:  # a tuna-trained checkpoint on top of the pinned base
                from . import train
                trained = train.load_weights(self._st, weights)
                if trained["base"] != model:
                    raise SystemExit(f"{weights} was trained from {trained['base']}, not {model}")
                self.weights_sha256 = common.file_sha256(os.path.join(weights, "weights.safetensors"))
        else:
            import torch
            from sentence_transformers import SentenceTransformer
            pinned.refuse_unpinned(model)
            if weights and (not revision or len(revision) != 40):
                raise SystemExit("--weights on a model outside pins.toml needs its full --revision")
            self._st = SentenceTransformer(model, revision=revision, device=device, trust_remote_code=False,
                                           model_kwargs={"dtype": torch.bfloat16})
            if weights:  # a tuna-trained checkpoint on top of the exact base revision
                from . import train
                trained = train.load_weights(self._st, weights, revision)
                if trained["base"] != model:
                    raise SystemExit(f"{weights} was trained from {trained['base']}, not {model}")
                self.weights_sha256 = common.file_sha256(os.path.join(weights, "weights.safetensors"))
        prompts = getattr(self._st, "prompts", None) or {}
        # The model's own document/query prompts win unless overridden.
        self.prefix = prefix if prefix is not None else prompts.get("document", common.PASSAGE_PREFIX)
        self.query_prefix = query_prefix if query_prefix is not None else prompts.get("query", common.QUERY_PREFIX)
        self.model, self.revision, self.device, self.batch = model, revision, device, batch
        self.token_budget = token_budget
        if max_tokens:  # read at most this many tokens (longer symbols are split), e.g. LFM2.5's 512
            self._st.max_seq_length = int(max_tokens)
        self.limit = int(self._st.max_seq_length)
        self.max_tokens_override = int(max_tokens) if max_tokens else None
        self.count_tokens = chunking.token_counter(self._st.tokenizer, self.prefix)

    def _encode(self, texts):
        return self._st.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=len(texts) > 1000).astype("float32")

    def encode_documents(self, texts):
        """Batches hold at most `batch` documents and `token_budget` padded tokens, longest
        first, so a few 32k-token symbols cannot exhaust GPU memory."""
        import numpy as np
        lengths = [self.count_tokens(t) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: -lengths[i])
        out, i, t0, next_report = None, 0, time.time(), 0.1
        while i < len(order):
            n = max(1, min(self.batch, self.token_budget // max(lengths[order[i]], 1)))
            part = order[i:i + n]
            emb = self._st.encode([self.prefix + texts[j] for j in part], batch_size=len(part),
                                  normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            if out is None:
                out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
            out[part] = emb
            i += n
            if i / len(order) >= next_report:
                print(f"[embed] {min(i, len(order))}/{len(order)} documents ({time.time() - t0:.0f}s)", flush=True,
                      file=sys.stderr)
                next_report += 0.1
        return out

    def encode_queries(self, texts):
        return self._encode([self.query_prefix + t for t in texts])

    def meta(self):
        return {"backend": self.backend, "model": self.model, "revision": self.revision, "device": self.device,
                "max_tokens": self.limit, "prefix": self.prefix, "query_prefix": self.query_prefix,
                "batch": self.batch, "token_budget": self.token_budget, "weights": self.weights,
                "weights_sha256": self.weights_sha256, "max_tokens_override": self.max_tokens_override}


def _normalize(M):
    import numpy as np
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n == 0, 1, n)


def from_args(args):
    if args.backend == "served":
        return ServedEncoder(args.model, args.url, args.tokenizer, args.tokenizer_revision, args.max_tokens,
                             common.PASSAGE_PREFIX if args.prefix is None else args.prefix,
                             common.QUERY_PREFIX if args.query_prefix is None else args.query_prefix,
                             args.api_key_env, args.batch)
    return LocalEncoder(args.model, args.revision, args.device, args.prefix, args.query_prefix, args.batch,
                        args.token_budget, args.weights, args.max_tokens)


def from_meta(meta, device=None):
    """Rebuild the encoder a vector file was made with, to encode queries the same way."""
    if meta["backend"] == "served":
        return ServedEncoder(meta["model"], meta["url"], meta["tokenizer"], meta["tokenizer_revision"],
                             meta["max_tokens"], meta["prefix"], meta["query_prefix"], meta.get("api_key_env"),
                             meta.get("batch", 32))
    enc = LocalEncoder(meta["model"], meta.get("revision"), device or meta.get("device", "cuda"),
                       meta["prefix"], meta["query_prefix"], meta.get("batch", 32), meta.get("token_budget", 65536),
                       meta.get("weights"), meta.get("max_tokens_override"))
    if meta.get("weights_sha256") and enc.weights_sha256 != meta["weights_sha256"]:
        raise SystemExit(f"{meta['weights']} changed since the vectors were built; re-embed")
    return enc


def add_arguments(p):
    p.add_argument("--backend", choices=["served", "local"], default="served")
    p.add_argument("--model", required=True, help="served model name, or HF id for --backend local")
    p.add_argument("--revision", help="--backend local: HF commit to load (Liquid models use pins.toml)")
    p.add_argument("--device", default="cuda", help="--backend local: torch device")
    p.add_argument("--url", default=os.environ.get("TUNA_EMB_URL", "http://127.0.0.1:8000/v1/embeddings"))
    p.add_argument("--tokenizer", help="--backend served: HF id or path of the served model's tokenizer")
    p.add_argument("--tokenizer-revision", help="--backend served: HF commit of --tokenizer")
    p.add_argument("--max-tokens", type=int, help="served: the server's maximum input length; local: read at most "
                                                     "this many tokens (longer symbols are split)")
    p.add_argument("--prefix", help="document prefix (default: the model's document prompt, else jina nl2code)")
    p.add_argument("--query-prefix", help="query prefix (default: the model's query prompt, else jina nl2code)")
    p.add_argument("--batch", type=int, default=32, help="documents per batch at most")
    p.add_argument("--weights", help="--backend local: a tuna train output dir applied to the pinned base model")
    p.add_argument("--token-budget", type=int, default=65536,
                   help="--backend local: padded tokens per batch at most (longest documents first)")
    p.add_argument("--api-key-env", help="env var holding a bearer token, if the endpoint needs one")
