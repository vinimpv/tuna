"""`tuna hint`: a UserPromptSubmit hook that puts a context hint card in front of the agent.

  tuna hint --repo saleor --tag lfm25-ft-lora --backend claude --model claude-haiku-5-5 [--log FILE]
  tuna hint ... --backend openai --url http://127.0.0.1:8000/v1 --model LFM3-1.7B

Claude Code runs it before the agent sees the prompt (hook input JSON on stdin). The agent
gets no search tool; search happens here instead:

  1. a small model reads the prompt and writes 0-4 search queries (0 = no card, the prompt
     passes through untouched)
  2. each query runs through tuna search (the MCP server's code, CPU query encoding, top 10)
  3. the small model reads the hits and picks up to 6 leads; the card lists each lead's file,
     FQN and line range (taken from the index, never from the model) and one line on why
  4. the card is returned as additionalContext

With --query-cache DIR the queries are written once per prompt and reused, so two indexes (stock
and tuned) are searched with identical queries; each card is still composed from its own hits.

Rules enforced in code, not only asked of the model:
  - at most 4 queries; a query naming an identifier (snake_case, camelCase, dotted) that is
    not in the prompt is dropped; at least one query is a literal span of the prompt (if the
    model gave none, the prompt's longest line, cut to 25 words, is added)
  - a lead must be one of this run's hits, verbatim (path, FQN and lines come from the index; a
    mechanical check after rendering logs every violation and drops it before injection); a "why" that concludes from absence ("does not
    exist", "nothing about", "never") is dropped, the lead kept
  - the card is at most CARD_MAX_CHARS (about 300 tokens); leads are cut from the end to fit
  - the card opens with a wrapper telling the agent these are unverified leads
  - fail open: any error or the 60 s cap gives a one-line "hint search failed" notice instead
  - recursion guard: with TUNA_HINT_INSIDE set the hook does nothing; the hook's own model
    calls set it, and run with no hooks, memory, CLAUDE.md, MCP, tools or settings

--log FILE appends one JSON record per prompt: queries, hits, card, model usage and cost per
call, latencies, status (card | passthrough | failed).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

from . import common

ENV_GUARD = "TUNA_HINT_INSIDE"
PROMPT_VERSION = "hint-v1"
MAX_QUERIES = 4
MAX_CANDIDATES = 6
HITS_PER_QUERY = 10
HITS_SHOWN = 24
CARD_MAX_CHARS = 1200
WHY_MAX_WORDS = 22
TIMEOUT_S = 60
WRAPPER = ("Suggested leads, auto-generated. Verify each one before relying on it, "
           "and keep searching if none fits.")
FAILED = "Hint search failed ({reason}); no suggested leads this turn. Work as usual."

QUERY_SYSTEM = """You write code-search queries for a semantic search index over one repository's Python classes, functions and methods.

Read the user's request and decide which searches would help find the code it is about. Reply with JSON only:
{"queries": ["...", "..."]}

Rules:
- 0 to 4 queries. Use 0 ({"queries": []}) when the request is not about finding code in this repository.
- The first query copies the user's own words: a short verbatim span (at most 20 words) from the request.
- The others describe the behaviour to find in plain words.
- Never invent identifiers. Use a function, class, variable or file name only if it appears in the request.
- No explanations, only the JSON."""

CARD_SYSTEM = """You pick leads for a developer from code-search results.

You get the developer's request and a numbered list of search hits (fully qualified name, file and lines, first lines of code). Choose up to 6 hits that are most likely where the request is handled. Reply with JSON only:
{"candidates": [{"symbol": "<fully qualified name copied exactly from the hits>", "why": "<one line>"}]}

Rules:
- Use only symbols from the hit list, copied exactly. Order them most promising first.
- "why": at most 20 words on what this code does that relates to the request.
- These are leads, not answers. Never conclude anything from absence: do not say something does not exist, is missing, is never recorded, or that nothing handles it. Do not claim certainty or diagnose the bug.
- If none of the hits looks related, reply {"candidates": []}.
- No explanations, only the JSON."""

ABSENCE = re.compile(r"\b(not exist|doesn'?t exist|does not exist|isn'?t (handled|implemented|recorded|stored)|"
                     r"is not (handled|implemented|recorded|stored)|never|nothing|no (code|handling|implementation|"
                     r"support|logic|check)|missing|absent|lacks?|without any)\b", re.I)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[A-Za-z0-9]*_[A-Za-z0-9_]+|"
                   r"[a-z]+[A-Z][A-Za-z0-9]*")


# ---------------------------------------------------------------- model backends

def _usage(cost=None, model=None, inp=0, cw=0, cr=0, out=0, seconds=0.0):
    return {"model": model, "cost_usd": cost, "input_tokens": inp, "cache_creation_tokens": cw,
            "cache_read_tokens": cr, "output_tokens": out, "seconds": round(seconds, 2)}


class ClaudeCLI:
    """`claude -p` headless with our own system prompt, no tools, and the bench agent's isolation:
    no settings, hooks, plugins, memory, CLAUDE.md, MCP or session; TUNA_HINT_INSIDE set."""
    name = "claude"
    ENV = {"CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
           "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
           "ENABLE_CLAUDEAI_MCP_SERVERS": "false", "DISABLE_AUTOUPDATER": "1"}
    SETTINGS = {"disableAllHooks": True, "autoMemoryEnabled": False,
                # built-in plugins of Claude Code 2.1.322, disabled for isolation
                "enabledPlugins": {p: False for p in ("cc-plugin-agents-md@builtin", "cc-plugin-telemetry@builtin",
                                                      "cc-plugin-privacy@builtin",
                                                      "cc-plugin-plugin-authoring@builtin")}}

    def __init__(self, model, workdir, debug_dir=None, binary="claude"):
        self.model, self.workdir, self.debug_dir, self.binary = model, workdir, debug_dir, binary
        self.calls = 0

    def complete(self, system, user, timeout):
        home = os.path.expanduser("~")
        # the binary is resolved on the caller's PATH, before the call's own PATH replaces it
        binary = shutil.which(self.binary) or self.binary
        env = {"HOME": home, "PATH": f"{home}/.local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "TERM": "dumb",
               ENV_GUARD: "1", **self.ENV}
        self.calls += 1
        cmd = [binary, "-p", "--output-format", "json", "--model", self.model, "--system-prompt", system,
               "--tools", "", "--restricted", "--setting-sources", "", "--settings", json.dumps(self.SETTINGS),
               "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"]
        if self.debug_dir:
            cmd += ["--debug-file", os.path.join(self.debug_dir, f"hint-model-{self.calls}.debug.log")]
        os.makedirs(self.workdir, exist_ok=True)
        t0 = time.perf_counter()
        p = subprocess.run(cmd + ["--", user], cwd=self.workdir, env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=timeout)
        secs = time.perf_counter() - t0
        try:
            e = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude exit {p.returncode}: {p.stderr[-300:] or p.stdout[-300:]}")
        if e.get("is_error") or e.get("subtype") != "success":
            raise RuntimeError(f"claude {e.get('subtype')} {e.get('api_error_status')}: {str(e.get('result'))[:300]}")
        u = e.get("usage") or {}
        models = sorted((e.get("modelUsage") or {}).keys())
        return e.get("result") or "", _usage(e.get("total_cost_usd"), ",".join(models) or self.model,
                                             u.get("input_tokens", 0), u.get("cache_creation_input_tokens", 0),
                                             u.get("cache_read_input_tokens", 0), u.get("output_tokens", 0), secs)


class OpenAICompat:
    """Any OpenAI-compatible /v1/chat/completions endpoint (vLLM, llama.cpp, ...): url + model id.
    Cost is not reported by such servers, so it is recorded as None."""
    name = "openai"

    def __init__(self, url, model, api_key_env=None):
        self.url = url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model, self.api_key_env = model, api_key_env

    def complete(self, system, user, timeout):
        body = {"model": self.model, "temperature": 0, "max_tokens": 600,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json"}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        t0 = time.perf_counter()
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            e = json.loads(r.read())
        u = e.get("usage") or {}
        return (e["choices"][0]["message"]["content"] or "",
                _usage(None, e.get("model") or self.model, u.get("prompt_tokens", 0), 0, 0,
                       u.get("completion_tokens", 0), time.perf_counter() - t0))


# ---------------------------------------------------------------- the card

def parse_json(text):
    """The first JSON object in the text (models sometimes wrap it in a fence)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def identifiers(text):
    return set(IDENT.findall(text))


def clean_queries(raw, prompt):
    """Apply the query rules. Returns (queries, notes)."""
    notes = {"model_queries": len(raw), "dropped_invented": [], "literal_fallback": False, "cut": 0}
    seen, out = set(), []
    for q in raw:
        if not isinstance(q, str) or not q.strip():
            continue
        q = " ".join(q.split())[:300]
        invented = [t for t in identifiers(q) if t not in prompt]
        if invented:
            notes["dropped_invented"].append({"query": q, "identifiers": invented})
            continue
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    if raw and out and not any(q.lower() in " ".join(prompt.split()).lower() for q in out):
        lines = [l.strip() for l in prompt.splitlines() if l.strip()]
        literal = " ".join(max(lines, key=len).split()[:25]) if lines else ""
        if literal:
            out.insert(0, literal)
            notes["literal_fallback"] = True
    if len(out) > MAX_QUERIES:
        notes["cut"] = len(out) - MAX_QUERIES
        out = out[:MAX_QUERIES]
    return out, notes


def merge_hits(results):
    """{query: [hit, ...]} -> hits by FQN, best rank first, at most HITS_SHOWN."""
    best = {}
    for q, hits in results.items():
        for h in hits:
            if h["fqn"] not in best or h["rank"] < best[h["fqn"]]["rank"]:
                best[h["fqn"]] = dict(h, query=q)
    return sorted(best.values(), key=lambda h: (h["rank"], -h["score"]))[:HITS_SHOWN]


def hits_prompt(prompt, hits):
    rows = []
    for i, h in enumerate(hits, 1):
        code = "\n".join("     " + l for l in h["snippet"].splitlines()[:3])
        rows.append(f"{i}. {h['fqn']}  ({h['file']}:{h['line_start']}-{h['line_end']})\n{code}")
    return f"Request:\n{prompt}\n\nSearch hits:\n" + "\n".join(rows)


def build_card(raw_candidates, hits):
    """Validate the model's picks against the hits and render the card text. Returns (card, leads, notes)."""
    by_fqn = {h["fqn"]: h for h in hits}
    leads, notes = [], {"model_candidates": len(raw_candidates), "dropped_unknown": [], "why_dropped": 0, "cut": 0}
    for c in raw_candidates:
        sym = (c or {}).get("symbol", "") if isinstance(c, dict) else ""
        sym = sym.strip().strip("`")
        if sym not in by_fqn:
            notes["dropped_unknown"].append(sym)
            continue
        if any(l["fqn"] == sym for l in leads):
            continue
        why = " ".join(str(c.get("why") or "").split()[:WHY_MAX_WORDS])
        if ABSENCE.search(why):
            why, notes["why_dropped"] = "", notes["why_dropped"] + 1
        h = by_fqn[sym]
        leads.append({"fqn": sym, "file": h["file"], "line_start": h["line_start"], "line_end": h["line_end"],
                      "why": why})
        if len(leads) == MAX_CANDIDATES:
            break
    if not leads:
        return None, [], notes

    def render(ls):
        return WRAPPER + "\n" + "\n".join(
            f"- {l['file']}:{l['line_start']}-{l['line_end']}  {l['fqn']}" + (f" — {l['why']}" if l["why"] else "")
            for l in ls)
    card = render(leads)
    while len(card) > CARD_MAX_CHARS and len(leads) > 1:
        leads.pop()
        notes["cut"] += 1
        card = render(leads)
    return card[:CARD_MAX_CHARS], leads, notes


def write_queries(prompt, backend, left):
    text, usage = backend.complete(QUERY_SYSTEM, prompt, left())
    queries, notes = clean_queries(parse_json(text).get("queries") or [], prompt)
    return queries, notes, dict(usage, step="queries")


def query_key(prompt, backend):
    import hashlib
    ident = json.dumps([PROMPT_VERSION, getattr(backend, "name", "?"), getattr(backend, "model", "?"), QUERY_SYSTEM,
                        prompt])
    return hashlib.sha256(ident.encode()).hexdigest()[:24]


def get_queries(prompt, backend, left, cache_dir=None):
    """The queries for this prompt. With cache_dir, written once per prompt and shared: every arm (the arm names
    are in scripts/cc-bench.py) that sees the same prompt (H-stock and H-tuned on one issue) searches with identical
    queries.
    A file lock makes a concurrent second arm wait for the first; its wait is returned. The usage
    of the call that wrote them is returned to every reader (each arm is charged for it, as a
    deployment would be). Returns (queries, notes, usage, source, lock_wait_s)."""
    if not cache_dir:
        q, n, u = write_queries(prompt, backend, left)
        return q, n, u, "model", 0.0
    import fcntl
    os.makedirs(cache_dir, exist_ok=True)
    key = query_key(prompt, backend)
    path = os.path.join(cache_dir, f"{key}.json")
    t0 = time.perf_counter()
    with open(os.path.join(cache_dir, f"{key}.lock"), "w") as lk:
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                left()  # raises at the time cap
                time.sleep(0.05)
        wait = round(time.perf_counter() - t0, 2)
        try:
            if os.path.exists(path):
                with open(path) as fh:
                    c = json.load(fh)
                return c["queries"], c["notes"], c["usage"], "cache", wait
            q, n, u = write_queries(prompt, backend, left)
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"queries": q, "notes": n, "usage": u, "prompt_sha256_24": key}, fh)
            os.replace(tmp, path)
            return q, n, u, "model", wait
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def check_card(card, leads, hits):
    """Mechanical check that every lead in the rendered card (path and FQN) is one of this run's
    hits. Returns the violations; build_card should make this empty, it is checked anyway."""
    pairs = {(h["file"], h["fqn"], h["line_start"], h["line_end"]) for h in hits}
    bad = [l for l in leads if (l["file"], l["fqn"], l["line_start"], l["line_end"]) not in pairs]
    lines = [l for l in (card or "").splitlines() if l.startswith("- ")]
    for line in lines:
        m = re.match(r"- (\S+):(\d+)-(\d+)  (\S+)", line)
        if not m or (m.group(1), m.group(4), int(m.group(2)), int(m.group(3))) not in pairs:
            bad.append({"line": line})
    return bad


def make_card(prompt, backend, search, timeout=TIMEOUT_S, query_cache=None):
    """The whole hint step. `search(query) -> [hit]` may block while the index loads.
    Returns the log record; record["card"] is the text to inject (None = pass through)."""
    deadline = time.monotonic() + timeout
    rec = {"version": PROMPT_VERSION, "backend": getattr(backend, "name", "?"), "status": None, "card": None,
           "queries": [], "calls": [], "seconds": {}}

    def left():
        s = deadline - time.monotonic()
        if s <= 1:
            raise TimeoutError("hint step over its time cap")
        return s
    t0 = time.perf_counter()
    queries, rec["query_notes"], usage, rec["query_source"], rec["query_lock_wait_s"] = get_queries(
        prompt, backend, left, query_cache)
    rec["calls"].append(dict(usage, step="queries", from_cache=rec["query_source"] == "cache"))
    rec["queries"] = queries
    rec["seconds"]["queries"] = round(time.perf_counter() - t0, 2)
    if not queries:
        rec["status"] = "passthrough"
        return rec
    t1 = time.perf_counter()
    results = {}
    for q in queries:
        left()
        results[q] = search(q)
    hits = merge_hits(results)
    rec["hits"] = [h["fqn"] for h in hits]
    seen = {}
    for hs in results.values():  # everything retrieved: the union of every query's top 10
        for h in hs:
            seen.setdefault(h["fqn"], h["file"])
    rec["retrieved"] = [{"fqn": f, "file": fl} for f, fl in seen.items()]
    rec["search_ranked"] = {q: [h["fqn"] for h in hs] for q, hs in results.items()}
    rec["seconds"]["search"] = round(time.perf_counter() - t1, 2)
    t2 = time.perf_counter()
    text, usage = backend.complete(CARD_SYSTEM, hits_prompt(prompt, hits), left())
    rec["calls"].append(dict(usage, step="card"))
    card, leads, rec["card_notes"] = build_card(parse_json(text).get("candidates") or [], hits)
    post = check_card(card, leads, hits)
    if post:  # should not happen: drop the offending leads and re-render from the rest
        keep = [l for l in leads if l not in post]
        card, leads, _ = build_card([{"symbol": l["fqn"], "why": l["why"]} for l in keep], hits)
    # every model pick that is not one of this run's hits is a violation (dropped before injection)
    rec["violations"] = ([{"symbol": s_, "reason": "not in this run's hits"} for s_ in rec["card_notes"]["dropped_unknown"]]
                         + [{"lead": v, "reason": "rendered lead not in hits"} for v in post])
    rec["seconds"]["card"] = round(time.perf_counter() - t2, 2)
    rec.update(card=card, leads=leads, status="card" if card else "passthrough")
    return rec


# ---------------------------------------------------------------- hook entry

def hook_output(text):
    return json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text}})


def run_hook(stdin_text, backend, search_factory, log_path=None, timeout=TIMEOUT_S, query_cache=None):
    """Returns what the hook prints (may be ""). Never raises."""
    if os.environ.get(ENV_GUARD):
        return ""
    t0 = time.perf_counter()
    rec = {"status": None}
    try:
        prompt = json.loads(stdin_text).get("prompt") or ""
    except (json.JSONDecodeError, AttributeError):
        prompt = ""
    box = {}

    def step():
        try:
            box["rec"] = make_card(prompt, backend, search_factory(), timeout, query_cache)
        except BaseException as e:  # noqa: BLE001 - fail open
            box["error"] = f"{type(e).__name__}: {e}"[:300]
    th = threading.Thread(target=step, daemon=True)
    th.start()
    th.join(timeout)
    if "rec" in box:
        rec = box["rec"]
    else:
        reason = box.get("error") or f"timed out after {timeout}s"
        rec = {"version": PROMPT_VERSION, "status": "failed", "error": reason, "card": None}
    rec["latency_s"] = round(time.perf_counter() - t0, 2)
    if rec["status"] == "failed":
        timed_out = "error" not in box or box["error"].startswith("TimeoutError")
        rec["failure"] = "timeout" if timed_out else "error"
        out = hook_output(FAILED.format(reason=rec["failure"]))
    elif rec.get("card"):
        out = hook_output(rec["card"])
    else:
        out = ""
    rec["injected"] = json.loads(out)["hookSpecificOutput"]["additionalContext"] if out else None
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def lazy_searcher(out, repo, tag, threads, fuse_bm25=False):
    """Starts loading the index at once (in parallel with the first model call); search waits for it."""
    box, ready = {}, threading.Event()

    def load():
        try:
            from . import mcp
            box["index"] = mcp.load_index(out, repo, tag, threads, fuse_bm25)[0]
        except BaseException as e:  # noqa: BLE001
            box["error"] = e
        finally:
            ready.set()
    threading.Thread(target=load, daemon=True).start()

    def search(q):
        ready.wait()
        if "error" in box:
            raise box["error"]
        return box["index"].search(q, HITS_PER_QUERY)
    return search


def make_backend(args):
    if args.backend == "claude":
        return ClaudeCLI(args.model, args.workdir or os.path.join(common.data_root(), "cc-bench", "hint-scratch"),
                         args.debug_dir)
    if args.backend == "openai":
        if not args.url:
            raise SystemExit("--backend openai needs --url")
        return OpenAICompat(args.url, args.model, args.api_key_env)
    raise SystemExit(f"unknown backend {args.backend}")


def run(args):
    if os.environ.get(ENV_GUARD):  # inside our own model call: do nothing, print nothing
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    real_stdout, sys.stdout = sys.stdout, sys.stderr  # only the hook JSON goes to stdout
    stdin_text = sys.stdin.read()
    try:
        backend = make_backend(args)
        searcher = lazy_searcher(args.out, args.repo, args.tag, args.threads, args.fuse_bm25)
        out = run_hook(stdin_text, backend, lambda: searcher, args.log, args.timeout, args.query_cache)
    except BaseException as e:  # noqa: BLE001 - fail open even on setup errors
        out = hook_output(FAILED.format(reason="error"))
        if args.log:
            with open(args.log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"version": PROMPT_VERSION, "status": "failed", "card": None,
                                     "error": f"setup: {type(e).__name__}: {e}"[:300], "injected": None}) + "\n")
    real_stdout.write(out)
    real_stdout.flush()


def add_parser(sub):
    p = sub.add_parser("hint", help="UserPromptSubmit hook: search before the agent starts, inject a hint card")
    p.add_argument("--tag", required=True, help="vector tag, e.g. lfm25-emb (stock) or lfm25-ft-lora (tuned); "
                                                "'bm25' for BM25 alone")
    p.add_argument("--fuse-bm25", action="store_true", help="BM25 fused with the dense index by reciprocal rank fusion")
    p.add_argument("--backend", choices=["claude", "openai"], default="claude")
    p.add_argument("--model", default="claude-haiku-5-5", help="claude model id, or the endpoint's model id")
    p.add_argument("--url", help="--backend openai: base URL of an OpenAI-compatible server")
    p.add_argument("--api-key-env", help="--backend openai: env var holding a bearer token")
    p.add_argument("--workdir", help="--backend claude: empty directory to run the model call in")
    p.add_argument("--debug-dir", help="--backend claude: write the model calls' debug logs here")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=float, default=TIMEOUT_S)
    p.add_argument("--log", help="append one JSON record per prompt")
    p.add_argument("--query-cache", help="share queries across runs that see the same prompt (one file per prompt)")
    p.set_defaults(func=run)
    return p
