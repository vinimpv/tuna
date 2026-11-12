"""`tuna mcp`: a stdio MCP server with one tool, `search`, over a saved embedding index.

  tuna mcp --repo saleor --tag lfm25-emb          # stock LFM2.5 index
  tuna mcp --repo saleor --tag lfm25-ft-lora      # tuned (the LoRA adapter is loaded for queries)
  tuna mcp --repo saleor --tag qwen3-0.6b-ft-lora --fuse-bm25   # BM25 + tuned Qwen3, RRF
  tuna mcp --repo saleor --tag bm25                              # BM25 alone

The corpus vectors are read from <out>/<repo>/vectors/<tag>.npy (+ _owner.npy,
.json); the server only encodes the query, on the CPU, with the encoder the
vectors were built with (encoders.from_meta). It never embeds a corpus and
never touches the GPU or the network (CUDA_VISIBLE_DEVICES and HF_HUB_OFFLINE
are set before torch loads).

Protocol: MCP over stdio, newline-delimited JSON-RPC 2.0, hand-rolled
(initialize, notifications/initialized, ping, tools/list, tools/call). Only
protocol messages go to stdout; everything else goes to stderr. The model
loads in a background thread so `initialize` answers at once; a search waits
for it.

--log FILE appends one JSON line per search (query, k, returned FQNs, latency).
"""
import json
import os
import sys
import threading
import time

from . import __version__

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_NAME = "tuna"
TOOL = {
    "name": "search",
    "description": ("Semantic code search over this repository's Python classes, functions and methods. "
                    "Give a natural-language description of the code you are looking for; returns the "
                    "k best-matching symbols, each with its fully qualified name, file path, line range "
                    "and a short snippet."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what the code does, in plain words"},
            "k": {"type": "integer", "description": "number of results (default 10, max 50)",
                  "minimum": 1, "maximum": 50, "default": 10},
        },
        "required": ["query"],
    },
}
MAX_K = 50
BM25_TAG = "bm25"  # --tag bm25: BM25 alone, no vectors
SNIPPET_LINES = 6
SNIPPET_CHARS = 400


def snippet(body, lines=SNIPPET_LINES, chars=SNIPPET_CHARS):
    """The symbol's first source lines, without extract's "# FQN:" / "# File:" header."""
    src = [l for l in body.splitlines() if not (l.startswith("# FQN: ") or l.startswith("# File: "))]
    src = [l for l in src if l.strip()][:lines]
    text = "\n".join(src)
    return text if len(text) <= chars else text[:chars].rstrip() + " ..."


def format_results(hits):
    """Plain text, one block per hit: rank, FQN, path:start-end, then the indented snippet."""
    if not hits:
        return "No results."
    out = []
    for h in hits:
        out.append(f"{h['rank']}. {h['fqn']}  ({h['file']}:{h['line_start']}-{h['line_end']})")
        out += ["   " + l for l in h["snippet"].splitlines()]
    return "\n".join(out)


class Index:
    """Vectors + symbols in memory; `encode` is any callable str -> unit vector.

    With `bm25` (a tuna.bm25.BM25 over the same symbols) the ranking is BM25 fused with
    the dense ranking by reciprocal rank fusion, exactly as scripts/rrf.py fuses saved
    ranks: each list cut to RRF_DEPTH, score 1/(RRF_K + rank), BM25's order first on ties."""
    RRF_K = 60
    RRF_DEPTH = 200

    def __init__(self, M, owner, symbols, encode, bm25=None):
        from . import chunking
        self.M, self.owner, self.symbols, self.encode, self.bm25 = M, owner, symbols, encode, bm25
        self._max = chunking.max_per_symbol
        if M is None:  # BM25 alone (tag "bm25"): no vectors, no encoder
            if bm25 is None:
                raise ValueError("an index needs vectors or BM25")
        elif len(M) != len(owner):
            raise ValueError(f"{len(M)} vectors but {len(owner)} owner entries")

    def dense(self, query):
        return self._max(self.M @ self.encode(query), self.owner, len(self.symbols))

    def ranked(self, query, k):
        """[(symbol index, score)] best first, k long."""
        import numpy as np
        if self.M is None:
            return [(int(i), float(sc)) for i, sc in self.bm25.top(query, k)]
        s = self.dense(query)
        if self.bm25 is None:
            top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
            top = top[np.argsort(-s[top], kind="stable")]
            return [(int(i), float(s[i])) for i in top]
        depth = min(self.RRF_DEPTH, len(s))
        dense_top = np.argpartition(-s, depth - 1)[:depth] if depth < len(s) else np.arange(len(s))
        dense_top = dense_top[np.argsort(-s[dense_top], kind="stable")]
        sc = {}
        for lst in ([i for i, _ in self.bm25.top(query, self.RRF_DEPTH)], [int(i) for i in dense_top]):
            for r, i in enumerate(lst):
                sc[i] = sc.get(i, 0.0) + 1.0 / (self.RRF_K + r + 1)
        return [(i, sc[i]) for i in sorted(sc, key=lambda i: -sc[i])[:k]]

    def search(self, query, k=10):
        k = max(1, min(int(k), MAX_K, len(self.symbols)))
        hits = []
        for r, (i, score) in enumerate(self.ranked(query, k), 1):
            sym = self.symbols[i]
            hits.append({"rank": r, "fqn": sym["fqn"], "file": sym["file_path"], "line_start": sym["line_start"],
                         "line_end": sym["line_end"], "score": round(score, 4), "snippet": snippet(sym["body"])})
        return hits


def load_index(out, repo, tag, threads=4, fuse_bm25=False):
    """Read the saved index and build the CPU query encoder it was made with."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np
    import torch
    from . import common, encoders
    torch.set_num_threads(threads)
    cfg = common.load_config(repo)
    work = os.path.join(out, cfg["name"])
    if tag == BM25_TAG:  # BM25 alone, as `tuna bm25` ranks (k1 1.5, b 0.75, no cap)
        from . import bm25 as bm25_mod
        symbols = [{k: s[k] for k in ("fqn", "file_path", "line_start", "line_end", "body")}
                   for s in common.load_symbols(work)]
        return Index(None, None, symbols, None, bm25_mod.BM25([s["body"] for s in symbols])), {"tag": BM25_TAG}
    vdir = os.path.join(work, "vectors")
    with open(os.path.join(vdir, f"{tag}.json")) as fh:
        meta = json.load(fh)
    M = np.load(os.path.join(vdir, f"{tag}.npy"))
    owner = np.load(os.path.join(vdir, f"{tag}_owner.npy"))
    symbols = [{k: s[k] for k in ("fqn", "file_path", "line_start", "line_end", "body")}
               for s in common.load_symbols(work)]
    if meta.get("symbols") and meta["symbols"] != len(symbols):
        raise SystemExit(f"{tag}: built over {meta['symbols']} symbols, symbols.jsonl has {len(symbols)}")
    enc = encoders.from_meta(meta, device="cpu")
    bm25 = None
    if fuse_bm25:  # the same documents and settings as `tuna bm25` (k1 1.5, b 0.75, no cap)
        from . import bm25 as bm25_mod
        bm25 = bm25_mod.BM25([s["body"] for s in symbols])

    def encode(q):
        with torch.inference_mode():
            return enc.encode_queries([q])[0].astype(M.dtype)
    return Index(M, owner, symbols, encode, bm25), meta


class Server:
    """JSON-RPC dispatch. `index_factory` runs in a thread; searches wait for it."""

    def __init__(self, index_factory, write, log_path=None, label=""):
        self.write, self.log_path, self.label = write, log_path, label
        self.index, self.error = None, None
        self.ready = threading.Event()
        self.lock = threading.Lock()

        def load():
            try:
                self.index = index_factory()
            except BaseException as e:  # noqa: BLE001 - reported to the client on the next call
                self.error = f"{type(e).__name__}: {e}"
                print(f"[tuna mcp] index failed to load: {self.error}", file=sys.stderr, flush=True)
            finally:
                self.ready.set()
        threading.Thread(target=load, daemon=True).start()

    def reply(self, mid, result=None, error=None):
        msg = {"jsonrpc": "2.0", "id": mid}
        msg["error" if error else "result"] = error or result
        self.write(json.dumps(msg, ensure_ascii=False))

    def handle(self, msg):
        if not isinstance(msg, dict):
            return self.reply(None, error={"code": -32600, "message": "invalid request"})
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        is_request = "id" in msg
        if method is None:  # a response to something we never sent; ignore
            return None
        if method == "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return self.reply(mid, {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False}},
                                    "serverInfo": {"name": SERVER_NAME, "version": __version__}})
        if method == "ping":
            return self.reply(mid, {}) if is_request else None
        if method == "tools/list":
            return self.reply(mid, {"tools": [TOOL]})
        if method == "tools/call":
            return self.reply(mid, self.call(params))
        if not is_request:  # notifications/initialized, notifications/cancelled, ...
            return None
        return self.reply(mid, error={"code": -32601, "message": f"method not found: {method}"})

    def call(self, params):
        if params.get("name") != "search":
            return {"isError": True, "content": [{"type": "text", "text": f"unknown tool {params.get('name')!r}"}]}
        args = params.get("arguments") or {}
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"isError": True, "content": [{"type": "text", "text": "search needs a non-empty 'query'"}]}
        try:
            k = int(args.get("k", 10))
        except (TypeError, ValueError):
            return {"isError": True, "content": [{"type": "text", "text": "'k' must be an integer"}]}
        t0 = time.perf_counter()
        self.ready.wait()
        waited = time.perf_counter() - t0
        if self.index is None:
            return {"isError": True, "content": [{"type": "text", "text": f"search index unavailable: {self.error}"}]}
        t1 = time.perf_counter()
        with self.lock:
            hits = self.index.search(query, k)
        ms = (time.perf_counter() - t1) * 1e3
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": time.time(), "label": self.label, "query": query, "k": k,
                                     "ranked": [h["fqn"] for h in hits], "search_ms": round(ms, 1),
                                     "load_wait_s": round(waited, 2)}) + "\n")
        return {"content": [{"type": "text", "text": format_results(hits)}]}

    def serve(self, lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.reply(None, error={"code": -32700, "message": "parse error"})
                continue
            for m in (msg if isinstance(msg, list) else [msg]):
                self.handle(m)


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, set before anything initialises CUDA
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    out_fh = sys.stdout
    sys.stdout = sys.stderr  # library prints (progress bars, warnings) must not corrupt the protocol stream
    write_lock = threading.Lock()

    def write(s):
        with write_lock:
            out_fh.write(s + "\n")
            out_fh.flush()
    server = Server(lambda: load_index(args.out, args.repo, args.tag, args.threads, args.fuse_bm25)[0], write,
                    log_path=args.log, label=f"{args.repo}/{args.tag}" + ("+bm25" if args.fuse_bm25 else ""))
    server.serve(sys.stdin)


def add_parser(sub):
    p = sub.add_parser("mcp", help="stdio MCP server: `search` over a saved embedding index (CPU query encoding)")
    p.add_argument("--tag", required=True, help="vector tag under vectors/, e.g. lfm25-emb or lfm25-ft-lora; "
                                                "'bm25' for BM25 alone")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads for query encoding")
    p.add_argument("--log", help="append one JSON line per search to this file")
    p.add_argument("--fuse-bm25", action="store_true",
                   help="rank by BM25 + dense reciprocal rank fusion (k=60, top 200 each), as scripts/rrf.py")
    p.set_defaults(func=run)
    return p
