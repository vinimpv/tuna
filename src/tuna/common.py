"""Shared helpers: repo configs, scope globs, jsonl I/O, rank files, HTTP retry.

Stdlib only, so clone, extraction, BM25 and eval run without numpy or torch.
"""
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.request

# jina-code-embeddings retrieval prefixes (nl2code task), as in the model card.
QUERY_PREFIX = "Find the most relevant code snippet given the following query:\n"
PASSAGE_PREFIX = "Candidate code snippet:\n"
RANK_DEPTH = 200

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# Source checkout layout: <root>/src/tuna/common.py and <root>/repos/*.toml.
PROJECT_REPOS_DIR = os.path.join(os.path.dirname(os.path.dirname(PKG_DIR)), "repos")


def data_root():
    """Where clones and outputs live: $TUNA_DATA, default ~/tuna-data."""
    return os.path.expanduser(os.environ.get("TUNA_DATA", "~/tuna-data"))


def repos_dirs():
    """Config search path for bare names: $TUNA_REPOS_DIR, ./repos, the checkout's repos/."""
    dirs = [os.environ.get("TUNA_REPOS_DIR"), os.path.join(os.getcwd(), "repos"), PROJECT_REPOS_DIR]
    return [d for d in dirs if d]


def load_config(name_or_path):
    """`saleor` resolves to repos/saleor.toml (see repos_dirs); a path is read as is.

    The clone path is, in order: $TUNA_<NAME>_PATH, `path` in the config, or
    <data root>/repos/<name>.
    """
    path = name_or_path
    if not os.path.isfile(path):  # a bare name; a directory called `saleor` in the cwd is not a config
        tried = [os.path.join(d, f"{name_or_path}.toml") for d in repos_dirs()]
        path = next((t for t in tried if os.path.exists(t)), None)
        if path is None:
            sys.exit(f"no config for {name_or_path!r}; looked in: " + ", ".join(tried))
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    for key in ("name", "commit", "include"):
        if key not in cfg:
            sys.exit(f"{path}: missing '{key}'")
    cfg.setdefault("exclude", [])
    env_key = "TUNA_" + re.sub(r"[^A-Z0-9]", "_", cfg["name"].upper()) + "_PATH"
    default = cfg.get("path") or os.path.join(data_root(), "repos", cfg["name"])
    cfg["path"] = os.path.expanduser(os.environ.get(env_key, default))
    cfg["_file"] = path
    return cfg


def _match(path, patterns):
    # fnmatch's `*` crosses `/`; testing "/"+path too lets "*/tests/*" hit a top-level tests/.
    return any(fnmatch.fnmatchcase(path, p) or fnmatch.fnmatchcase("/" + path, p) for p in patterns)


def in_scope(path, cfg):
    return _match(path, cfg["include"]) and not _match(path, cfg["exclude"])


def check_commit(cfg, allow_mismatch=False):
    """The clone must sit at the pinned commit with no tracked modifications."""
    root = cfg["path"]
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    problems = []
    if head != cfg["commit"]:
        problems.append(f"HEAD {head} != pinned {cfg['commit']}")
    if dirty:
        problems.append(f"tracked files modified:\n{dirty[:500]}")
    if problems and not allow_mismatch:
        sys.exit(f"{cfg['name']}: " + "; ".join(problems))
    return head


def workdir(out_root, cfg):
    d = os.path.join(out_root, cfg["name"])
    os.makedirs(d, exist_ok=True)
    return d


def metrics_root():
    """The run ledger's directory: $TUNA_METRICS, default <data root>/metrics."""
    return os.path.expanduser(os.environ.get("TUNA_METRICS") or os.path.join(data_root(), "metrics"))


def default_out():
    """Output root: $TUNA_OUT, default <data root>/out. Kept out of the source tree
    because vectors and rank files are large."""
    return os.path.expanduser(os.environ.get("TUNA_OUT") or os.path.join(data_root(), "out"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_symbols(work):
    return read_jsonl(os.path.join(work, "symbols.jsonl"))


def load_questions(path):
    """Questions jsonl: {"id", "question", "gold_fqns": [...]}; id defaults to the line number."""
    qs = read_jsonl(path)
    for i, q in enumerate(qs):
        q.setdefault("id", str(i))
        if not q.get("question") or not q.get("gold_fqns"):
            sys.exit(f"{path}: row {i} needs 'question' and a non-empty 'gold_fqns'")
    return qs


def rank_row(q, fqns, scores):
    return {"id": q["id"], "question": q["question"], "ranked": list(fqns),
            "scores": [round(float(s), 5) for s in scores]}


def symbol_doc(sym, cap=0):
    """The document BM25, the embedders and the reranker see: header + source
    (see extract.symbol_text), optionally cut to `cap` characters."""
    doc = sym["body"]
    return doc[:cap] if cap else doc


def post_json(url, payload, timeout=600, api_key=None, tries=4):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as e:  # noqa: BLE001 - retry anything, report it
            print(f"[warn] {url} attempt {attempt + 1}/{tries}: {e}", file=sys.stderr, flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")
