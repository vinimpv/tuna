"""Does a coding agent find code faster or cheaper with tuna search? Claude Code, headless, several arms.

  .venv/bin/python scripts/cc-bench.py prepare --repo saleor
  .venv/bin/python scripts/cc-bench.py sample  --repo saleor --name pilot --take claude-m3:25 issues-v1:25 --seed 20261112
  .venv/bin/python scripts/cc-bench.py canary  --repo saleor --name pilot
  .venv/bin/python scripts/cc-bench.py run     --repo saleor --name pilot --workers 4
  .venv/bin/python scripts/cc-bench.py report  --repo saleor --name pilot
  .venv/bin/python scripts/cc-bench.py report  --repo saleor zulip posthog --name full   # + pooled

Arms (same questions, same prompt, same model; only the tool list differs):
  A  Read, Grep, Glob                      (read-only stock tools)
  B  A + `search` over the stock LFM2.5 index        (tuna mcp --tag lfm25-emb)
  C  A + `search` over the tuned LFM2.5 LoRA index   (tuna mcp --tag lfm25-ft-lora)
  D  A + `search`: BM25 fused with tuned Qwen3-0.6B LoRA by RRF (tuna mcp --tag qwen3-0.6b-ft-lora --fuse-bm25),
     the best issues row of the retrieval tables; run after A-C
Hook arms (A's tools, plus a hint card injected before the agent starts by scripts/tuna-hint-hook;
see src/tuna/hint.py):
  H-stock  card over the stock LFM2.5 index      H-tuned  card over the tuned LFM2.5 LoRA index
  H-bm25   card over BM25 alone                   H-fused  card over BM25 + tuned Qwen3-0.6B LoRA (RRF, as D)

Each run is `claude -p --output-format stream-json` in a clean, read-only export
of the repo at the pinned commit (prepare), with no Bash, edit, web, agents or
other MCP servers. Isolation, all asserted by `canary` and re-checked on every run:
  - env -i: only HOME, PATH, LANG, TERM, plus switches that turn off CLAUDE.md
    loading, auto-memory, bundled skills, claude.ai connectors, nonessential traffic
  - --setting-sources "" (no user/project/local settings, so no settings hooks),
    --settings with the built-in plugins off, hooks off, deny rules on $TUNA_DATA,
    ~/.claude, ~/.config and this checkout
  - --restricted: file tools confined to the checkout
  - --strict-mcp-config: only the arm's own server
  - --disable-slash-commands, --no-session-persistence, --permission-mode dontAsk
`canary` asks the agent to report any injected CLAUDE.md / memory / hook text and to
read outside the checkout, then checks the init event, the event stream (no hook
events) and the debug log (0 hooks, 0 plugins, connectors disabled). `run` refuses to
start without a passing canary for the same flag set.

Grading, no judge: the answer's final `FILE:` line names a gold file, and its
`SYMBOL:` line names a gold FQN (after normalising `path.py::Class.f`, backticks,
`.__init__` modules). Also recorded: tokens, cost (Claude Code's reported
total_cost_usd), tool calls by name, turns, wall time, exit status, model ids,
and for B/C every search the agent ran (from the server's log).

Outputs under $TUNA_OUT/<repo>/cc-bench/<name>/: questions.jsonl, runs.jsonl (one
row per run), events/ (stream-json, debug log, search log per run), canary.json,
report.json. Every run also appends a ledger line (stage cc-bench-run). Agent runs and the canary log
under $TUNA_OUT/runs/; the checkouts and scratch directories live in $TUNA_BENCH_HOME (default
$TUNA_DATA/cc-bench).
Stops launching new runs on a rate-limit rejection, when the account's rate-limit window reaches
--max-utilization (default 0.985), or after
3 failed runs in a row.
"""
import argparse
import collections
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tuna import common, hint, metrics  # noqa: E402

MODEL = "claude-sonnet-5-5"
BASE_TOOLS = ["Read", "Grep", "Glob"]
SEARCH_TOOL = "mcp__tuna__search"
ARMS = {"A": None, "B": "lfm25-emb", "C": "lfm25-ft-lora", "D": "qwen3-0.6b-ft-lora+bm25",
        "H-stock": None, "H-tuned": None, "H-bm25": None, "H-fused": None}  # value: the search tool's index (None = no search tool)
# Hint-card arms: no search tool; a UserPromptSubmit hook (tuna hint) searches this index first.
HOOK_ARMS = {"H-stock": "lfm25-emb", "H-tuned": "lfm25-ft-lora", "H-bm25": "bm25",
             "H-fused": "qwen3-0.6b-ft-lora+bm25"}  # "+bm25": RRF with BM25, as arm D
HINT_BACKEND = {"backend": "claude", "model": "claude-haiku-5-5"}
# claude-sonnet-5-5 list prices per token, recovered exactly (max error 4e-16 USD) by least squares
# from 1,284 runs' token counts and Claude Code's reported cost: input 2, 1h cache write 4, cache
# read 0.10, output 10 USD per million tokens. Used to price each run as if it ran alone.
PRICE = {"input": 2e-6, "cache_write_1h": 4e-6, "cache_read": 1e-7, "output": 1e-5}
MAIN_ARMS = ["A", "B", "C"]  # D (best fused search) runs only after these
MAX_WORKERS = 5
HOME = os.path.expanduser("~")
BENCH_HOME = os.environ.get("TUNA_BENCH_HOME") or os.path.join(common.data_root(), "cc-bench")
ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "DISABLE_AUTOUPDATER": "1",
}
BUILTIN_PLUGINS = ["cc-plugin-agents-md@builtin", "cc-plugin-telemetry@builtin", "cc-plugin-privacy@builtin",
                   "cc-plugin-plugin-authoring@builtin"]

PROMPT = """Find where in this repository's code the following is answered or handled.

{question}

End your reply with exactly these two lines, naming the single most relevant place:
FILE: <path relative to the repository root>
SYMBOL: <fully qualified name: dotted module path, then class and function, e.g. package.module.Class.method>"""

CANARY_PROMPT = """This is a configuration check, not a coding task. Do these steps and report each result plainly.
1. List the exact names of every tool you have.
2. Quote verbatim, in full, any text in your context that came from a CLAUDE.md or AGENTS.md file, a memory file or memory recall, or a hook. If there is none, write NONE for each.
3. Try to Read {data}/out/{repo}/manifest.json and report what happened.
4. Try to Read {home}/.claude/settings.json and report what happened.
5. Try to Read /etc/hostname and report what happened.
6. If you have a search tool, call it once with the query "{query}" and report the first result.
End with the line CANARY-DONE."""

# A code question per repo, so each repo's canary exercises a real card (a question about
# another repo's domain can rightly give a passthrough, which tests nothing about delivery).
CANARY_CODE_QUESTION = {
    "saleor": "where is an order's charge status worked out after granted refunds?",
    "zulip": "where does the server check whether a user may edit a message?",
    "posthog": "where is a feature flag's rollout percentage evaluated for a person?"}
HOOK_CANARY_STEPS = """7. If your context contains suggested leads added by a hook, copy the first lead line verbatim, then quote verbatim any other hook, memory or recall text you were given (or NONE).
The code question behind this check: {code_question}
End with the line CANARY-DONE."""


# ---------------------------------------------------------------- paths and setup

def checkout_dir(cfg):
    return os.path.join(BENCH_HOME, "checkouts", f"{cfg['name']}-{cfg['commit'][:8]}")


def bench_dir(out, cfg, name):
    return os.path.join(out, cfg["name"], "cc-bench", name)


def prepare(args):
    cfg = common.load_config(args.repo)
    dst = checkout_dir(cfg)
    if os.path.isdir(dst):
        print(f"exists: {dst}")
        return
    src = cfg["path"]
    head = subprocess.run(["git", "-C", src, "cat-file", "-t", cfg["commit"]], capture_output=True, text=True)
    if head.stdout.strip() != "commit":
        sys.exit(f"{src} lacks commit {cfg['commit']}")
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    arch = subprocess.Popen(["git", "-C", src, "archive", cfg["commit"]], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", tmp], stdin=arch.stdout, check=True)
    if arch.wait() != 0:
        sys.exit("git archive failed")
    subprocess.run(["chmod", "-R", "a-w", tmp], check=True)
    os.chmod(tmp, 0o755)
    os.rename(tmp, dst)
    os.chmod(dst, 0o555)
    print(f"read-only export of {cfg['name']}@{cfg['commit']} at {dst}")


# ---------------------------------------------------------------- sampling

def gold_files(q, fqn_file):
    files = [fqn_file[f] for f in q["gold_fqns"] if f in fqn_file]
    return sorted(set(files))


def sample(args):
    """Seeded; within each set, prefer questions whose gold file is not taken yet (spreads gold files)."""
    cfg = common.load_config(args.repo)
    work = os.path.join(args.out, cfg["name"])
    fqn_file = {s["fqn"]: s["file_path"] for s in common.load_symbols(work)}
    rng = random.Random(args.seed)
    used, picked = set(), []
    for spec in args.take:
        qset, n = spec.split(":")
        qs = common.load_questions(os.path.join(work, "questions", f"{qset}.jsonl"))
        qs = [q for q in qs if gold_files(q, fqn_file)]
        rng.shuffle(qs)
        fresh = [q for q in qs if not set(gold_files(q, fqn_file)) & used]
        rest = [q for q in qs if q not in fresh]
        take = []
        for q in fresh:  # greedy: a question on a not-yet-used gold file while any is left
            if len(take) < int(n) and not set(gold_files(q, fqn_file)) & used:
                take.append(q)
                used.update(gold_files(q, fqn_file))
        for q in fresh + rest:  # then fill, in the same seeded order
            if len(take) < int(n) and q not in take:
                take.append(q)
                used.update(gold_files(q, fqn_file))
        if len(take) < int(n):
            sys.exit(f"{qset}: only {len(take)} usable questions")
        for q in take:
            picked.append({"id": q["id"], "set": qset, "question": q["question"], "gold_fqns": q["gold_fqns"],
                           "gold_files": gold_files(q, fqn_file), "leak": q.get("leak")})
    d = bench_dir(args.out, cfg, args.name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "questions.jsonl")
    if os.path.exists(path) and not args.force:
        sys.exit(f"{path} exists (--force to resample)")
    common.write_jsonl(path, picked)
    files = collections.Counter(f for q in picked for f in q["gold_files"])
    meta = {"repo": cfg["name"], "commit": cfg["commit"], "take": args.take, "seed": args.seed,
            "questions": len(picked), "distinct_gold_files": len(files), "max_per_file": max(files.values()),
            "sha256": common.file_sha256(path)}
    with open(os.path.join(d, "questions.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))


# ---------------------------------------------------------------- one claude run

def settings_json(cfg, out, hook_cmd=None):
    data = os.path.realpath(common.data_root())
    deny = sorted({f"Read(/{p}/**)" for p in (data, os.path.realpath(out), os.path.join(HOME, ".claude"),
                                              os.path.join(HOME, ".config"), ROOT, BENCH_HOME + "/canary")})
    s = {"enabledPlugins": {p: False for p in BUILTIN_PLUGINS}, "disableAllHooks": True,
         "autoMemoryEnabled": False, "permissions": {"deny": deny}}
    if hook_cmd:  # exactly one hook: ours (user/project settings are not loaded at all)
        s["disableAllHooks"] = False
        s["hooks"] = {"UserPromptSubmit": [{"matcher": "", "hooks": [
            {"type": "command", "command": hook_cmd, "timeout": int(hint.TIMEOUT_S) + 30}]}]}
    return s


HINT_HOOK = os.path.join(ROOT, "scripts", "tuna-hint-hook")


def hook_command(cfg, out, arm, hint_log, debug_dir, query_cache):
    env = [f"TUNA_OUT={out}"] + [f"{k}={os.environ[k]}" for k in ("HF_HOME", "TUNA_DATA") if os.environ.get(k)]
    tag = HOOK_ARMS[arm]
    fuse = tag.endswith("+bm25")
    args = ["--repo", cfg["_file"], "--tag", tag.removesuffix("+bm25")] + (["--fuse-bm25"] if fuse else []) + [
            "--backend", HINT_BACKEND["backend"],
            "--model", HINT_BACKEND["model"], "--log", hint_log, "--debug-dir", debug_dir,
            "--query-cache", query_cache, "--workdir", os.path.join(BENCH_HOME, "hint-model-cwd")]
    return " ".join(env + [HINT_HOOK] + args)


def mcp_json(cfg, out, tag, search_log):
    fuse = tag.endswith("+bm25")
    tag = tag.removesuffix("+bm25")
    env = {"PATH": "/usr/bin:/bin", "TUNA_OUT": out}
    for k in ("HF_HOME", "TUNA_DATA"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return {"mcpServers": {"tuna": {"type": "stdio", "command": os.path.join(ROOT, "scripts", "tuna-mcp"),
                                    "args": ["--repo", cfg["_file"], "--tag", tag, "--threads", "4",
                                             "--log", search_log] + (["--fuse-bm25"] if fuse else []), "env": env}}}


def claude_command(arm, settings_path, mcp_path, prompt, debug_path, model):
    tools = BASE_TOOLS + ([SEARCH_TOOL] if ARMS[arm] else [])
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose", "--include-hook-events",
           "--model", model, "--restricted", "--setting-sources", "", "--settings", settings_path,
           "--strict-mcp-config"]
    if mcp_path:
        cmd += ["--mcp-config", mcp_path]
    cmd += ["--tools", ",".join(BASE_TOOLS), "--allowedTools", ",".join(tools),
            "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--no-session-persistence", "--disable-slash-commands", "--debug-file", debug_path, "--", prompt]
    return cmd


def flags_hash(arm, model):
    """The command line minus per-run paths and the prompt: what a canary vouches for."""
    cmd = claude_command(arm, "S", "M" if ARMS[arm] else None, "P", "D", model)
    parts = [cmd, ISOLATION_ENV, BUILTIN_PLUGINS]
    if arm in HOOK_ARMS:
        parts.append([HOOK_ARMS[arm], HINT_BACKEND, hint.PROMPT_VERSION, hint.QUERY_SYSTEM, hint.CARD_SYSTEM,
                      hint.TIMEOUT_S, hint.CARD_MAX_CHARS, hint.MAX_QUERIES, hint.MAX_CANDIDATES, hint.WRAPPER])
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16]


def claude_env():
    env = {"HOME": HOME, "PATH": f"{HOME}/.local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "TERM": "dumb"}
    env.update(ISOLATION_ENV)
    return env


def parse_events(path):
    init, result, rate, hooks, tool_calls, texts, denied = None, None, None, [], collections.Counter(), [], []
    hook_events, msg_ids, first_usage = [], set(), {}
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t, st = e.get("type"), e.get("subtype")
        if t == "system" and st == "init":
            init = e
        elif t == "system" and st and st.startswith("hook"):
            hooks.append(st)
            hook_events.append({"subtype": st, "event": e.get("hook_event"), "output": e.get("output"),
                                "stdout": e.get("stdout")})
        elif t == "system" and st == "permission_denied":
            denied.append(e.get("decision_reason"))
        elif t == "assistant":
            if not msg_ids:
                first_usage = e["message"].get("usage") or {}
            msg_ids.add(e["message"].get("id"))
            for c in e["message"].get("content", []):
                if c.get("type") == "tool_use":
                    tool_calls[c["name"]] += 1
                elif c.get("type") == "text":
                    texts.append(c["text"])
        elif t == "rate_limit_event":
            rate = e.get("rate_limit_info")
        elif t == "result":
            result = e
    return {"init": init, "result": result, "rate": rate, "hooks": hooks, "tool_calls": dict(tool_calls),
            "texts": texts, "denied": denied, "hook_events": hook_events, "api_calls": len(msg_ids),
            "first_cache_read": first_usage.get("cache_read_input_tokens", 0),
            "first_cache_write": first_usage.get("cache_creation_input_tokens", 0)}


def isolation_problems(ev, arm, debug_path):
    """Per-run check of what reached the agent."""
    p = []
    init = ev["init"] or {}
    want = sorted(BASE_TOOLS + ([SEARCH_TOOL] if ARMS[arm] else []))
    if sorted(init.get("tools") or []) != want:
        p.append(f"tools {init.get('tools')} != {want}")
    servers = [s.get("name") for s in init.get("mcp_servers") or []]
    if servers != (["tuna"] if ARMS[arm] else []):
        p.append(f"mcp servers {init.get('mcp_servers')}")
    if ARMS[arm] and any(s.get("status") != "connected" for s in init.get("mcp_servers") or []):
        p.append(f"mcp not connected: {init.get('mcp_servers')}")
    for k in ("plugins", "skills", "slash_commands"):
        if init.get(k):
            p.append(f"{k}: {init.get(k)}")
    if init.get("memory_paths"):
        p.append(f"memory_paths: {init.get('memory_paths')}")
    try:
        dbg = open(debug_path, encoding="utf-8", errors="replace").read()
    except OSError:
        dbg = ""
    if arm in HOOK_ARMS:
        got = [(h["subtype"], h["event"]) for h in ev.get("hook_events", [])]
        core = [g for g in got if g[0] != "hook_progress"]  # a hook running for seconds also reports progress
        if core != [("hook_started", "UserPromptSubmit"), ("hook_response", "UserPromptSubmit")] or \
                any(e_ != "UserPromptSubmit" for _, e_ in got):
            p.append(f"hook events {got} != one UserPromptSubmit hook")
        ran = re.findall(r"Hooks: (\w+) \[([^\]]*)\]", dbg)
        if not ran or any(ev_ != "UserPromptSubmit" or HINT_HOOK not in cmd for ev_, cmd in ran):
            p.append(f"debug log: hooks run {ran} are not only ours")
    elif ev["hooks"]:
        p.append(f"hook events: {ev['hooks']}")
    if "Registered 0 hooks from 0 plugins" not in dbg:
        p.append("debug log: hooks/plugins registered")
    if re.search(r"hooks module \S+ loaded", dbg):
        p.append("debug log: a hooks module loaded")
    if "[claudeai-mcp] Disabled via env var" not in dbg:
        p.append("debug log: claude.ai connectors not disabled")
    if re.search(r"CLAUDE\.md|AGENTS\.md", "\n".join(l for l in dbg.splitlines() if "Loaded" in l or "memory" in l)):
        p.append("debug log: CLAUDE.md/AGENTS.md loaded")
    return p


def run_claude(cfg, out, arm, prompt, run_dir, run_id, model, timeout):
    os.makedirs(run_dir, exist_ok=True)
    ev_path = os.path.join(run_dir, f"{run_id}.events.jsonl")
    dbg_path = os.path.join(run_dir, f"{run_id}.debug.log")
    slog = os.path.join(run_dir, f"{run_id}.search.jsonl")
    for p in (ev_path, dbg_path, slog):
        if os.path.exists(p):
            os.remove(p)
    settings_path = os.path.join(run_dir, f"{run_id}.settings.json")
    hint_log = os.path.join(run_dir, f"{run_id}.hint.jsonl")
    hint_dbg = os.path.join(run_dir, f"{run_id}.hint-debug")
    hook_cmd = None
    if arm in HOOK_ARMS:
        if os.path.exists(hint_log):
            os.remove(hint_log)
        shutil.rmtree(hint_dbg, ignore_errors=True)
        os.makedirs(hint_dbg)
        os.makedirs(os.path.join(BENCH_HOME, "hint-model-cwd"), exist_ok=True)
        hook_cmd = hook_command(cfg, out, arm, hint_log, hint_dbg,
                                os.path.join(os.path.dirname(run_dir), "query-cache"))
    with open(settings_path, "w") as fh:
        json.dump(settings_json(cfg, out, hook_cmd), fh)
    mcp_path = None
    if ARMS[arm]:
        mcp_path = os.path.join(run_dir, f"{run_id}.mcp.json")
        with open(mcp_path, "w") as fh:
            json.dump(mcp_json(cfg, out, ARMS[arm], slog), fh)
    cmd = claude_command(arm, settings_path, mcp_path, prompt, dbg_path, model)
    start = metrics.utc_now()
    t0 = time.perf_counter()
    with open(ev_path, "w") as fh:
        try:
            proc = subprocess.run(cmd, cwd=checkout_dir(cfg), env=claude_env(), stdin=subprocess.DEVNULL,
                                  stdout=fh, stderr=subprocess.PIPE, timeout=timeout, text=True)
            rc, err = proc.returncode, proc.stderr[-2000:]
        except subprocess.TimeoutExpired:
            rc, err = "timeout", ""
    wall = time.perf_counter() - t0
    ev = parse_events(ev_path)
    searches = [json.loads(l) for l in open(slog)] if os.path.exists(slog) else []
    iso = isolation_problems(ev, arm, dbg_path)
    hint_rec = None
    if arm in HOOK_ARMS:
        recs = [json.loads(l) for l in open(hint_log)] if os.path.exists(hint_log) else []
        hint_rec = recs[-1] if recs else {"status": "missing", "card": None, "injected": None}
        if len(recs) > 1:
            iso.append(f"hook ran {len(recs)} times")
        iso += hint_model_problems(hint_dbg)
        if hint_rec.get("injected") and hint_rec["injected"] not in hook_contexts(ev):
            iso.append("the injected card is not in the transcript's hook_response")
    return ev, {"exit": rc, "stderr": err, "wall_s": round(wall, 2), "start_utc": metrics.iso(start),
                "end_utc": metrics.iso(metrics.utc_now()), "events": ev_path, "debug": dbg_path,
                "isolation": iso, "searches": searches, "hint": hint_rec}


def hook_contexts(ev):
    """additionalContext of every hook_response in the transcript (parsed, not substring-matched)."""
    out = []
    for h in ev.get("hook_events", []):
        raw = h.get("stdout") or ""  # "output" merges stderr into stdout; the hook's JSON is on stdout
        if h["subtype"] == "hook_response" and raw:
            try:
                out.append(json.loads(raw)["hookSpecificOutput"]["additionalContext"])
            except (ValueError, KeyError, TypeError):
                out.append(raw)
    return out


def hint_model_problems(debug_dir):
    """The hook's own model calls: no hooks, plugins, connectors or CLAUDE.md, from their debug logs."""
    p = []
    for fn in sorted(f for f in os.listdir(debug_dir) if f.endswith(".debug.log")) if os.path.isdir(debug_dir) else []:
        dbg = open(os.path.join(debug_dir, fn), encoding="utf-8", errors="replace").read()
        if "Registered 0 hooks from 0 plugins" not in dbg:
            p.append(f"{fn}: hooks/plugins registered")
        if re.search(r"Hooks: \w+ \[", dbg) or re.search(r"hooks module \S+ loaded", dbg):
            p.append(f"{fn}: a hook ran inside the hint model call")
        if "[claudeai-mcp] Disabled via env var" not in dbg:
            p.append(f"{fn}: claude.ai connectors not disabled")
    return p


# ---------------------------------------------------------------- grading

def norm_path(p, root=""):
    p = p.strip().strip("`*'\" ")
    if root and p.startswith(root):
        p = p[len(root):]
    p = re.sub(r"[:#]L?\d+(-L?\d+)?$", "", p)  # path:12-30
    return p.lstrip("./") if p.startswith("./") else p.lstrip("/")


def norm_fqn(s):
    s = s.strip().strip("`*'\" ")
    s = re.sub(r"\(.*?\)$", "", s)  # f() / f(a, b)
    s = s.strip().rstrip(".")
    m = re.match(r"^([\w./-]+)\.py(?:::|:|\.)(.+)$", s)  # path/to/mod.py::Class.f
    if m:
        s = m.group(1).replace("/", ".") + "." + m.group(2).replace("::", ".")
    s = s.replace("::", ".").replace("/", ".")
    s = re.sub(r"\.__init__(?=\.)", "", s)  # package __init__ module, not a trailing __init__ method
    return s


def last_line(text, key):
    found = None
    for line in text.splitlines():
        m = re.match(rf"^\W*{key}\W*:\s*(.+?)\s*$", line.strip(), re.I)
        if m:
            found = m.group(1)
    return found


def grade(answer, q, root):
    ans_file = last_line(answer, "FILE")
    ans_sym = last_line(answer, "SYMBOL")
    golds = {norm_fqn(f) for f in q["gold_fqns"]}
    file_hit = bool(ans_file) and norm_path(ans_file, root) in q["gold_files"]
    sym_hit = bool(ans_sym) and norm_fqn(ans_sym) in golds
    # a symbol whose module path is missing but whose tail and file match (e.g. "Class.method" + right FILE)
    tail_hit = bool(ans_sym) and file_hit and any(g.endswith("." + norm_fqn(ans_sym)) for g in golds)
    flat = answer.replace(root, "")
    return {"answer_file": ans_file, "answer_symbol": ans_sym, "file_hit": file_hit, "symbol_hit": sym_hit,
            "symbol_tail_hit": sym_hit or tail_hit,
            "file_mentioned": any(f in flat for f in q["gold_files"]),
            "symbol_mentioned": any(g in flat for g in q["gold_fqns"])}


def search_stats(searches, q):
    golds = set(q["gold_fqns"])
    ranks = [s["ranked"].index(g) + 1 for s in searches for g in golds if g in s["ranked"]]
    return {"searches": len(searches), "search_gold_best_rank": min(ranks) if ranks else None,
            "search_ms_mean": round(sum(s["search_ms"] for s in searches) / len(searches), 1) if searches else None,
            "search_load_wait_s_max": max((s.get("load_wait_s", 0) for s in searches), default=None)}


# ---------------------------------------------------------------- canary

def canary(args):
    cfg = common.load_config(args.repo)
    if not os.path.isdir(checkout_dir(cfg)):
        sys.exit("run prepare first")
    d = bench_dir(args.out, cfg, args.name)
    run_dir = os.path.join(d, "canary")
    report = {"time_utc": metrics.iso(metrics.utc_now()), "model": args.model, "arms": {}}
    ok = True
    for arm in args.arms:
        prompt = CANARY_PROMPT.format(data=os.path.realpath(common.data_root()), repo=cfg["name"], home=HOME,
                                      query="order charge status after refunds")
        if arm in HOOK_ARMS:  # a code question so the hook writes a card, and a step that quotes it
            prompt = prompt.replace("End with the line CANARY-DONE.", HOOK_CANARY_STEPS.format(
                code_question=CANARY_CODE_QUESTION.get(cfg["name"], CANARY_CODE_QUESTION["saleor"])))
        ev, info = run_claude(cfg, args.out, arm, prompt, run_dir, f"canary_{arm}", args.model, 600)
        res = ev["result"] or {}
        text = res.get("result") or ""
        problems = list(info["isolation"])
        if res.get("subtype") != "success":
            problems.append(f"result {res.get('subtype')}")
        # every outside read must have failed: count successful Read results outside the checkout
        reads_out = outside_reads(info["events"], checkout_dir(cfg))
        if reads_out:
            problems.append(f"read outside the checkout succeeded: {reads_out}")
        if ARMS[arm] and not info["searches"]:
            problems.append("search tool never answered")
        if "CANARY-DONE" not in text:
            problems.append("no CANARY-DONE")
        hint_rec = info.get("hint")
        if arm in HOOK_ARMS:
            if (hint_rec or {}).get("status") != "card":
                problems.append(f"hook status {(hint_rec or {}).get('status')}: {(hint_rec or {}).get('error')}")
            else:
                first = hint_rec["leads"][0]["fqn"]
                if first not in text:
                    problems.append(f"agent did not quote the card's first lead {first}")
                if hint_rec["card"] not in hook_contexts(ev):
                    problems.append("card not in the transcript")
        models = sorted((res.get("modelUsage") or {}).keys())
        if models != [args.model]:
            problems.append(f"models used {models}")
        report["arms"][arm] = {"flags_hash": flags_hash(arm, args.model), "problems": problems,
                               "tools": (ev["init"] or {}).get("tools"), "mcp_servers": (ev["init"] or {}).get("mcp_servers"),
                               "plugins": (ev["init"] or {}).get("plugins"), "skills": (ev["init"] or {}).get("skills"),
                               "hook_events": ev["hooks"], "denied": ev["denied"], "models": models,
                               "searches": len(info["searches"]), "cost_usd": res.get("total_cost_usd"),
                               "agent_report": text,
                               "hint": {k: (hint_rec or {}).get(k) for k in ("status", "queries", "query_source", "card",
                                                                             "violations", "latency_s", "calls")}
                               if arm in HOOK_ARMS else None}
        ok &= not problems
        print(f"[canary {arm}] {'PASS' if not problems else 'FAIL'} {problems}", flush=True)
    try:  # keep other arms' canaries (e.g. A-C) when checking one more arm (D)
        prev = json.load(open(os.path.join(d, "canary.json")))
        report["arms"] = {**{a: v for a, v in prev["arms"].items() if a not in report["arms"]}, **report["arms"]}
    except (OSError, ValueError, KeyError):
        pass
    report["passed"] = ok
    with open(os.path.join(d, "canary.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    sys.exit(0 if ok else 1)


def outside_reads(ev_path, root):
    """Read/Grep/Glob calls on a path outside the checkout whose result was not an error."""
    calls, bad = {}, []
    for line in open(ev_path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "assistant":
            for c in e["message"].get("content", []):
                if c.get("type") == "tool_use" and c["name"] in BASE_TOOLS:
                    p = (c.get("input") or {}).get("file_path") or (c.get("input") or {}).get("path") or ""
                    if p.startswith("/") and not os.path.realpath(p).startswith(root):
                        calls[c["id"]] = p
        elif e.get("type") == "user":
            for c in e["message"].get("content", []):
                if isinstance(c, dict) and c.get("tool_use_id") in calls:
                    body = json.dumps(c.get("content"))
                    if not c.get("is_error") and "tool_use_error" not in body and "outside" not in body:
                        bad.append(calls[c["tool_use_id"]])
    return bad


# ---------------------------------------------------------------- the bench

SERVICE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
SERVICE_TEXT = re.compile(r"rate.?limit|usage limit|overloaded|\b(429|5\d\d)\b|api error|internal server error|"
                          r"service unavailable|ECONNRESET|ETIMEDOUT|socket hang up|network", re.I)


def failure_reason(res, info):
    if not res:
        return f"no result event (exit {info['exit']}): {info['stderr'][-300:]}"
    return f"{res.get('subtype')} {res.get('api_error_status')}: {str(res.get('result'))[:300]}"


def failure_kind(res, info):
    """service: the API failed us (rate limit, overload, 5xx, network); rerun it. agent: anything
    else (turn or budget limits, the run timing out, a tool loop); it counts as a miss at its cost."""
    if res and res.get("api_error_status") in SERVICE_STATUS:
        return "service"
    text = (json.dumps(res.get("result")) if res else "") + " " + info["stderr"]
    if SERVICE_TEXT.search(text) and (not res or res.get("is_error")):
        return "service"
    return "agent"


def alone(cost, first_cache_read):
    """The run priced as if no other run had warmed the cache: its first API call's cache reads
    become 1-hour cache writes. Later calls read only what this run itself wrote."""
    if cost is None or first_cache_read is None:
        return None
    return round(cost + first_cache_read * (PRICE["cache_write_1h"] - PRICE["cache_read"]), 6)


def hint_fields(rec, q, answer, root, agent_cost):
    """Hook metrics for a hint-arm run; for other arms only total_cost_usd (= the agent's cost)."""
    if rec is None:
        return {"total_cost_usd": agent_cost}
    calls = rec.get("calls") or []
    cost = sum(c.get("cost_usd") or 0 for c in calls)
    leads = rec.get("leads") or []
    golds = {norm_fqn(f) for f in q["gold_fqns"]}
    ans_file, ans_sym = last_line(answer, "FILE"), last_line(answer, "SYMBOL")
    lead_files = {l["file"] for l in leads}
    lead_syms = {norm_fqn(l["fqn"]) for l in leads}
    used = bool(leads) and ((bool(ans_file) and norm_path(ans_file, root) in lead_files) or
                            (bool(ans_sym) and norm_fqn(ans_sym) in lead_syms))
    card = rec.get("card") or ""
    return {"hook_status": rec.get("status"), "hook_failure": rec.get("failure") or rec.get("error"),
            "hint_queries": len(rec.get("queries") or []), "hint_queries_list": rec.get("queries") or [],
            "query_source": rec.get("query_source"),
            "query_lock_wait_s": rec.get("query_lock_wait_s") or 0,
            "hint_passthrough": rec.get("status") == "passthrough", "hook_failed": rec.get("status") in ("failed", "missing"),
            "card_chars": len(card), "card_tokens_approx": round(len(card) / 4), "card_leads": len(leads),
            "card_gold_file": any(l["file"] in q["gold_files"] for l in leads),
            "card_gold_symbol": any(norm_fqn(l["fqn"]) in golds for l in leads),
            "answer_used_card": used, "card_violations": len(rec.get("violations") or []),
            "hook_cost_usd": round(cost, 6),
            "hook_input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
            "hook_cache_creation_tokens": sum(c.get("cache_creation_tokens") or 0 for c in calls),
            "hook_cache_read_tokens": sum(c.get("cache_read_tokens") or 0 for c in calls),
            "hook_output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
            "hook_models": sorted({c.get("model") for c in calls if c.get("model")}),
            "hook_latency_s": rec.get("latency_s"),
            "hook_latency_net_s": round((rec.get("latency_s") or 0) - (rec.get("query_lock_wait_s") or 0), 2),
            "retrieved": len(rec.get("retrieved") or []),
            "retrieved_gold_file": any(r["file"] in q["gold_files"] for r in rec.get("retrieved") or []),
            "retrieved_gold_symbol": any(norm_fqn(r["fqn"]) in golds for r in rec.get("retrieved") or []),
            "query_invented_dropped": len((rec.get("query_notes") or {}).get("dropped_invented") or []),
            "query_model_queries": (rec.get("query_notes") or {}).get("model_queries"),
            "total_cost_usd": round((agent_cost or 0) + cost, 6)}


def make_row(q, arm, ev, info, model, root, git):
    res = ev["result"] or {}
    usage = res.get("usage") or {}
    answer = res.get("result") or ""
    ok = res.get("subtype") == "success" and not res.get("is_error") and info["exit"] == 0
    kind = None if ok else failure_kind(res, info)
    return {"qid": q["id"], "set": q["set"], "arm": arm, "tag": ARMS[arm], "requested_model": model,
            "models": sorted((res.get("modelUsage") or {}).keys()),
            "status": "ok" if ok else f"{kind}_fail", "failure_reason": None if ok else failure_reason(res, info), "subtype": res.get("subtype"), "exit": info["exit"],
            "is_error": res.get("is_error"), "api_error_status": res.get("api_error_status"),
            "answer": answer, **grade(answer, q, root),
            "input_tokens": usage.get("input_tokens"), "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"), "output_tokens": usage.get("output_tokens"),
            "total_input_tokens": sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                                "cache_read_input_tokens")),
            "cost_usd": res.get("total_cost_usd"), "num_turns": res.get("num_turns"),
            "duration_api_ms": res.get("duration_api_ms"), "tool_calls": ev["tool_calls"],
            "tool_calls_total": sum(ev["tool_calls"].values()), "wall_s": info["wall_s"],
            "start_utc": info["start_utc"], "end_utc": info["end_utc"], "isolation_problems": info["isolation"],
            "stderr": info["stderr"] if not ok else "",
            **search_stats(info["searches"], q), "git": git,
            "flags_hash": flags_hash(arm, model),
            "api_calls": ev.get("api_calls"), "first_cache_read": ev.get("first_cache_read"),
            "first_cache_write": ev.get("first_cache_write"),
            "cost_alone_usd": alone(res.get("total_cost_usd"), ev.get("first_cache_read")),
            "grep_calls": ev["tool_calls"].get("Grep", 0),
            "read_calls": ev["tool_calls"].get("Read", 0), "glob_calls": ev["tool_calls"].get("Glob", 0),
            **hint_fields(info.get("hint"), q, answer, root, res.get("total_cost_usd"))}


def recover(args):
    """Rows for runs whose claude process finished after the harness was stopped: the events file
    has a result event but runs.jsonl has no row. Wall time is Claude's own duration_ms; the
    row is marked recovered. Unfinished runs (no result event) are left out."""
    cfg = common.load_config(args.repo)
    d = bench_dir(args.out, cfg, args.name)
    runs_path = os.path.join(d, "runs.jsonl")
    have = {(r["qid"], r["arm"]) for r in common.read_jsonl(runs_path)} if os.path.exists(runs_path) else set()
    qs = {q["id"]: q for q in common.read_jsonl(os.path.join(d, "questions.jsonl"))}
    root, git, n = checkout_dir(cfg) + "/", metrics.git_info(), 0
    edir = os.path.join(d, "events")
    for fn in sorted(os.listdir(edir)):
        if not fn.endswith(".events.jsonl"):
            continue
        run_id = fn[:-len(".events.jsonl")]
        qid, arm = run_id.rsplit("_", 1)
        if (qid, arm) in have or qid not in qs:
            continue
        ev = parse_events(os.path.join(edir, fn))
        res = ev["result"]
        if not res:
            print(f"unfinished, skipped: {run_id}")
            continue
        slog = os.path.join(edir, f"{run_id}.search.jsonl")
        end = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(edir, fn)), datetime.timezone.utc)
        wall = (res.get("duration_ms") or 0) / 1000
        info = {"exit": 0, "stderr": "", "wall_s": round(wall, 2),
                "start_utc": metrics.iso(end - datetime.timedelta(seconds=wall)), "end_utc": metrics.iso(end),
                "isolation": isolation_problems(ev, arm, os.path.join(edir, f"{run_id}.debug.log")),
                "searches": [json.loads(l) for l in open(slog)] if os.path.exists(slog) else []}
        row = dict(make_row(qs[qid], arm, ev, info, args.model, root, git), recovered=True)
        with open(runs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        print(f"recovered {run_id} {row['status']} file={row['file_hit']}")
    print(f"{n} rows recovered")


def reuse(args):
    """Copy finished rows from earlier benches on the same repo into this one: only status ok, only
    questions in this bench's set, only when the run's flags hash equals today's for that arm (same
    command line, isolation and model). Each copy is marked reused_from; existing rows win."""
    cfg = common.load_config(args.repo)
    d = bench_dir(args.out, cfg, args.name)
    runs_path = os.path.join(d, "runs.jsonl")
    have = {(r["qid"], r["arm"]) for r in common.read_jsonl(runs_path)} if os.path.exists(runs_path) else set()
    qids = {q["id"] for q in common.read_jsonl(os.path.join(d, "questions.jsonl"))}
    counts = collections.Counter()
    for src in args.sources:
        p = os.path.join(bench_dir(args.out, cfg, src), "runs.jsonl")
        for r in common.read_jsonl(p):
            key = (r["qid"], r["arm"])
            if key in have or r["qid"] not in qids or r["status"] != "ok" or r["isolation_problems"]:
                continue
            if r["flags_hash"] != flags_hash(r["arm"], r["requested_model"]):
                counts[f"{src}: flags differ"] += 1
                continue
            with open(runs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(r, reused_from=src), ensure_ascii=False) + "\n")
            have.add(key)
            counts[f"{src}: {r['arm']}"] += 1
    print(json.dumps(dict(sorted(counts.items()))))


def run(args):
    cfg = common.load_config(args.repo)
    if args.workers > MAX_WORKERS:
        sys.exit(f"--workers at most {MAX_WORKERS}")
    d = bench_dir(args.out, cfg, args.name)
    try:
        can = json.load(open(os.path.join(d, "canary.json")))
    except OSError:
        sys.exit("no canary.json: run `canary` first")
    for arm in args.arms:
        a = can["arms"].get(arm)
        if not a or a["problems"] or a["flags_hash"] != flags_hash(arm, args.model):
            sys.exit(f"arm {arm}: no passing canary for this flag set; rerun `canary`")
    qs = common.read_jsonl(os.path.join(d, "questions.jsonl"))
    if args.limit:
        qs = qs[:args.limit]
    runs_path = os.path.join(d, "runs.jsonl")
    done, service_failed = set(), {}
    if os.path.exists(runs_path):
        for r in common.read_jsonl(runs_path):
            if r.get("status") in ("ok", "agent_fail"):  # an agent failure is a final result (a miss)
                done.add((r["qid"], r["arm"]))
            elif r.get("status") == "service_fail":
                service_failed[(r["qid"], r["arm"])] = r.get("failure_reason")
    for k in done:
        service_failed.pop(k, None)
    # Interleaved: issues in a seeded random order, each issue's arms back to back in a seeded
    # random order, so every arm meets the same prompt-cache state.
    rng = random.Random(args.seed)
    order = list(qs)
    rng.shuffle(order)
    todo = []
    for q in order:
        arms = list(args.arms)
        rng.shuffle(arms)
        todo += [(q, a) for a in arms if (q["id"], a) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already done, {len(service_failed)} service-failure reruns), "
          f"{args.workers} workers, batch {args.batch}", flush=True)
    index = {(q["id"], a): i for i, (q, a) in enumerate(todo)}
    root = checkout_dir(cfg) + "/"
    lock, stop = threading.Lock(), threading.Event()
    state = {"cost": 0.0, "n": 0, "fails_in_row": 0, "t0": time.time()}
    git = metrics.git_info()

    def one(item):
        q, arm = item
        if stop.is_set():
            return None
        run_id = f"{q['id']}_{arm}"
        ev, info = run_claude(cfg, args.out, arm, PROMPT.format(question=q["question"]),
                              os.path.join(d, "events"), run_id, args.model, args.timeout)
        row = make_row(q, arm, ev, info, args.model, root, git)
        row.update(batch=args.batch, order_seed=args.seed, order_index=index[(q["id"], arm)])
        rerun_of = service_failed.get((q["id"], arm))
        if rerun_of:
            row["rerun_reason"] = rerun_of
        res, ok = ev["result"] or {}, row["status"] == "ok"
        line = {"stage": "cc-bench-run", "repo": cfg["name"], "model": args.model,
                "config": {"bench": args.name, "arm": arm, "tag": ARMS[arm], "qid": q["id"], "set": q["set"]},
                "counts": {"items": 1, "tool_calls": row["tool_calls_total"], "turns": row["num_turns"]},
                "tokens": {"prompt": row["total_input_tokens"], "completion": row["output_tokens"],
                           "cache_read": row["cache_read_tokens"], "cache_creation": row["cache_creation_tokens"]},
                "outputs": [], "extra": {"cost_usd": row["cost_usd"], "file_hit": row["file_hit"],
                                         "symbol_hit": row["symbol_hit"], "models": row["models"]},
                "start_utc": info["start_utc"], "end_utc": info["end_utc"], "wall_seconds": info["wall_s"],
                "status": "ok" if ok else f"failed: {res.get('subtype')} {res.get('api_error_status')}",
                "git": git}
        line["extra"].update(status=row["status"], failure_reason=row["failure_reason"], batch=args.batch,
                             order_index=row["order_index"])
        with lock:
            with open(runs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics.append(line)
            if rerun_of:
                metrics.append({"stage": "cc-bench-rerun", "repo": cfg["name"], "model": args.model,
                                "config": {"bench": args.name, "arm": arm, "qid": q["id"]},
                                "extra": {"reason": rerun_of, "rerun_status": row["status"]},
                                "end_utc": info["end_utc"], "git": git, "status": "ok"})
            state["n"] += 1
            state["cost"] += row["cost_usd"] or 0
            state["fails_in_row"] = 0 if ok else state["fails_in_row"] + 1  # agent or service
            rate = ev["rate"] or {}
            why = None
            if (rate.get("utilization") or 0) >= args.max_utilization:
                why = f"rate-limit {rate.get('rateLimitType')} window at {rate.get('utilization')}"
            elif rate.get("status") not in (None, "allowed", "allowed_warning"):
                why = f"rate limit {rate.get('status')}"
            elif row["status"] == "service_fail":
                why = f"service failure (will be rerun): {row['failure_reason'][:200]}"
            elif state["fails_in_row"] >= 3:
                why = "3 failed runs in a row"
            elif row["isolation_problems"]:
                why = f"isolation problem: {row['isolation_problems']}"
            if why and not stop.is_set():
                stop.set()
                print(f"STOPPING: {why}", flush=True)
            el = time.time() - state["t0"]
            print(f"[{state['n']}/{len(todo)}] {run_id} {row['status']} file={row['file_hit']} sym={row['symbol_hit']} "
                  f"${row['cost_usd'] or 0:.3f} tools={row['tool_calls_total']} {row['wall_s']:.0f}s | "
                  f"total ${state['cost']:.2f}, {el / 60:.1f} min, 7d util {rate.get('utilization')}", flush=True)
        return row

    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))
    summary = {"stage": "cc-bench", "repo": cfg["name"], "model": args.model,
               "config": {"bench": args.name, "arms": args.arms, "workers": args.workers},
               "counts": {"items": state["n"]}, "extra": {"cost_usd": round(state["cost"], 4),
                                                          "stopped": stop.is_set()},
               "wall_seconds": round(time.time() - state["t0"], 1), "git": git,
               "end_utc": metrics.iso(metrics.utc_now()), "status": "stopped" if stop.is_set() else "ok"}
    metrics.append(summary)
    print(json.dumps(summary))
    if stop.is_set():
        sys.exit(3)


# ---------------------------------------------------------------- report

# Analysis plan (recorded in the ledger as stage cc-bench-analysis-plan before results were read):
# primary outcome cost per task; primary comparisons B-A, C-A, C-B pooled over all issues; paired
# bootstrap resampling gold-file clusters within repo; everything else secondary; arm D exploratory.
PRIMARY = "cost_usd"
PRIMARY_PAIRS = [("B", "A"), ("C", "A"), ("C", "B")]
EXPLORATORY_PAIRS = [("D", "A"), ("D", "C"), ("D", "B")]
METRICS = ["cost_usd", "file_hit", "symbol_hit", "symbol_tail_hit", "tool_calls_total", "wall_s", "wall_s_net",
           "output_tokens", "input_tokens", "cache_creation_tokens", "cache_read_tokens", "total_input_tokens",
           "num_turns", "search_used", "searches", "grep_calls", "read_calls", "glob_calls", "api_calls",
           "input_side_tokens", "total_cost_usd", "hook_cost_usd", "hook_latency_s", "cost_alone_usd",
           "total_cost_alone_usd", "first_cache_read"]
# hint-card pilot: the main comparison of the later full run is H-tuned minus H-stock on gold file
HINT_PAIRS = [("H-tuned", "H-stock"), ("H-tuned", "H-bm25"), ("H-fused", "H-tuned"),
              ("H-stock", "A"), ("H-tuned", "A"), ("H-bm25", "A"), ("H-fused", "A")]
HINT_ROLES = {("H-tuned", "H-stock"): "registered main outcome (gold file); second named outcome (gold symbol)",
              ("H-tuned", "H-bm25"): "secondary", ("H-fused", "H-tuned"): "secondary"}
CARD_ARMS = ["H-stock", "H-tuned", "H-bm25", "H-fused"]
CARD_METRICS = ["retrieved", "retrieved_gold_file", "retrieved_gold_symbol", "hint_queries", "hint_passthrough", "hook_failed", "card_chars", "card_tokens_approx", "card_leads",
                "card_gold_file", "card_gold_symbol", "answer_used_card", "card_violations", "hook_cost_usd",
                "hook_input_tokens", "hook_cache_creation_tokens", "hook_cache_read_tokens", "hook_output_tokens",
                "hook_latency_s", "hook_latency_net_s", "query_lock_wait_s"]
N_BOOT = 10000


def analysis_plan(latest):
    """The registered analysis plan of the bench the report covers: the issues bench (arms A-D) registered cost per
    task; the hint benches registered gold file, H-tuned minus H-stock, with each card arm minus A as secondary."""
    common_ = {"bootstrap": f"{N_BOOT} replicates, gold-file clusters resampled within repo",
               "failures": "agent failure = miss at its cost; service failures rerun",
               "pilot_questions_excluded_in": "without_pilot"}
    if any(a in HOOK_ARMS for _, a in latest):
        cards = sorted(a for a in CARD_ARMS if any(k[1] == a for k in latest))
        return {"primary": "file_hit", "primary_pairs": ["H-tuned-H-stock"],
                "secondary": [f"{a}-A" for a in cards], **common_}
    return {"primary": PRIMARY, "primary_pairs": ["-".join(p) for p in PRIMARY_PAIRS],
            "exploratory": "arm D and its pairs", **common_}


def bootstrap(per_q, n=N_BOOT, seed=0):
    """per_q: list of values (one per question). 95% percentile interval of the mean (iid questions)."""
    import numpy as np
    x = np.asarray(per_q, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return [round(float(x.mean()), 4), round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def cluster_bootstrap(vals, clusters, n=N_BOOT, seed=0):
    """vals: {qkey: value}. clusters: {repo: [[qkey, ...], ...]} (one list per gold file). Each replicate
    draws, within every repo, as many clusters as it has, with replacement; the statistic is the mean
    over the questions drawn. Same seed for every metric, so every metric sees the same replicates.
    Returns [mean, 2.5%, 97.5%]."""
    import numpy as np
    rng = np.random.default_rng(seed)
    tot_s, tot_c = np.zeros(n), np.zeros(n)
    for repo in sorted(clusters):
        cl = clusters[repo]
        S = np.array([sum(vals[q] for q in c) for c in cl], dtype=float)
        C = np.array([len(c) for c in cl], dtype=float)
        idx = rng.integers(0, len(cl), size=(n, len(cl)))
        tot_s += S[idx].sum(axis=1)
        tot_c += C[idx].sum(axis=1)
    m = tot_s / tot_c
    point = sum(vals.values()) / len(vals)
    return [round(point, 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def batch_of(r):
    if r.get("reused_from") == "pilot":
        return "pilot"
    if r.get("reused_from"):
        return f"{r['reused_from']}-shuffled"
    return r.get("batch") or "shuffled"


def load_latest(path, repo, bench_root=None):
    """Final row per (repo/qid, arm): the latest row that is not a service failure (those are rerun).
    An agent failure is final and counts as a miss at its reported cost."""
    latest, reruns = {}, collections.Counter()
    for r in common.read_jsonl(path):
        r = dict(r, repo=repo, qkey=f"{repo}/{r['qid']}", search_used=bool(r.get("searches")),
                 searches=r.get("searches") or 0, batch=batch_of(r), grep_calls=(r.get("tool_calls") or {}).get("Grep", 0),
                 wall_s_net=r["wall_s"] - (r.get("search_load_wait_s_max") or 0))  # minus waiting for the index to load
        r["status"] = {"error": "agent_fail"}.get(r["status"], r["status"])
        tc = r.get("tool_calls") or {}
        r.setdefault("read_calls", tc.get("Read", 0))
        r.setdefault("glob_calls", tc.get("Glob", 0))
        r.setdefault("total_cost_usd", r.get("cost_usd"))
        r["input_side_tokens"] = sum(r.get(k) or 0 for k in ("input_tokens", "cache_read_tokens", "cache_creation_tokens"))
        if r.get("cost_alone_usd") is not None:
            r["total_cost_alone_usd"] = round(r["cost_alone_usd"] + (r.get("hook_cost_usd") or 0), 6)
        if r["status"] == "service_fail":
            reruns[r["arm"]] += 1
            continue
        if r["status"] == "agent_fail":
            r.update(file_hit=False, symbol_hit=False, symbol_tail_hit=False)
        latest[(r["qkey"], r["arm"])] = r
    return latest, reruns


def search_use(rows, gold_files, fqn_file, bench_dirs):
    """Observational (the agent chose when to search): per arm, how often it searched, calls per
    run, whether a returned symbol sat in a gold file, and gold-file rate with vs without a search."""
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rs = [r for r in rows if r["arm"] == arm]
        if not ARMS.get(arm):
            continue
        used = [r for r in rs if r["searches"]]
        found = []
        for r in used:
            d = bench_dirs[(r["repo"], r.get("reused_from") or "")]
            p = os.path.join(d, "events", f"{r['qid']}_{arm}.search.jsonl")
            logs = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
            files = {fqn_file[r["repo"]].get(f) for s in logs for f in s["ranked"]}
            found.append(bool(files & set(gold_files[r["qkey"]])))
        unused = [r for r in rs if not r["searches"]]
        out[arm] = {"runs": len(rs), "runs_with_search": len(used),
                    "share_with_search": round(len(used) / len(rs), 3) if rs else None,
                    "search_calls_per_run": round(sum(r["searches"] for r in rs) / len(rs), 2) if rs else None,
                    "search_calls_per_searching_run": round(sum(r["searches"] for r in used) / len(used), 2) if used else None,
                    "gold_file_in_search_results": round(sum(found) / len(found), 3) if found else None,
                    "gold_file_rate_with_search": round(sum(r["file_hit"] for r in used) / len(used), 3) if used else None,
                    "gold_file_rate_without_search": round(sum(r["file_hit"] for r in unused) / len(unused), 3) if unused else None,
                    "gold_symbol_rate_with_search": round(sum(r["symbol_hit"] for r in used) / len(used), 3) if used else None,
                    "gold_symbol_rate_without_search": round(sum(r["symbol_hit"] for r in unused) / len(unused), 3) if unused else None}
    return out


def summarize(latest, gold_files, population, arms=None):
    """population: the qkeys to use. Paired over questions with a final row in every arm."""
    arms = arms or sorted({a for _, a in latest})
    qids = sorted(q for q in population if all((q, a) in latest for a in arms))
    out = {"arms": arms, "questions": len(qids), "per_arm": {}, "paired_diff": {}}
    if not qids:
        return out
    clusters = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in qids:
        clusters[q.split("/", 1)[0]][min(gold_files[q])].append(q)
    clusters = {repo: list(c.values()) for repo, c in clusters.items()}
    out["clusters"] = {repo: len(c) for repo, c in clusters.items()}

    def vals(arm, m):
        return {q: float(latest[(q, arm)].get(m) or 0) for q in qids}
    for arm in arms:
        rs = [latest[(q, arm)] for q in qids]
        pa = out["per_arm"][arm] = {m: cluster_bootstrap(vals(arm, m), clusters) for m in METRICS}
        w = sorted(r["wall_s"] for r in rs)
        pa.update(wall_s_median=round(w[len(w) // 2], 1), wall_s_max=round(max(w), 1),
                  agent_failures=sum(r["status"] == "agent_fail" for r in rs),
                  agent_failures_cost_unknown=sum(r["status"] == "agent_fail" and r.get("cost_usd") is None for r in rs),
                  cost_total_usd=round(sum(r.get("cost_usd") or 0 for r in rs), 2),
                  models=sorted({m for r in rs for m in r["models"]}),
                  isolation_problems=sum(bool(r["isolation_problems"]) for r in rs),
                  batches=dict(collections.Counter(r["batch"] for r in rs)))
        tc = collections.Counter()
        for r in rs:
            tc.update(r["tool_calls"])
        pa["tool_calls_by_name_mean"] = {k: round(v / len(qids), 2) for k, v in sorted(tc.items())}
    for a, b in PRIMARY_PAIRS + EXPLORATORY_PAIRS + HINT_PAIRS:
        if a in arms and b in arms:
            out["paired_diff"][f"{a}-{b}"] = {
                "role": ("primary" if (a, b) in PRIMARY_PAIRS else "exploratory" if (a, b) in EXPLORATORY_PAIRS
                         else HINT_ROLES.get((a, b), "secondary (card arm - A)")),
                **{m: cluster_bootstrap({q: x - y for (q, x), y in zip(vals(a, m).items(), vals(b, m).values())},
                                        clusters) for m in METRICS}}
    return out


def hint_report(latest, gold_files, keys, args):
    """Card metrics per hint arm (pooled and per repo), whether both hint arms used identical queries,
    cost by run position within an issue (all three arms share the cached system prompt and first
    message, so position matters), and the full-run projection with two priced extra arms."""
    import datetime
    arms = sorted({a for _, a in latest})
    hint_arms = [a for a in arms if a in HOOK_ARMS]
    out = {"card": {}, "per_repo_card": {}}

    def card_stats(ks, arm):
        rs = [latest[(q, arm)] for q in ks if (q, arm) in latest]
        st = {m: round(sum(float(r.get(m) or 0) for r in rs) / len(rs), 4) for m in CARD_METRICS} if rs else {}
        cards = [r for r in rs if r.get("hook_status") == "card"]
        st.update(runs=len(rs), cards=len(cards), passthrough=sum(bool(r.get("hint_passthrough")) for r in rs),
                  hook_failures=sum(bool(r.get("hook_failed")) for r in rs),
                  violations_total=sum(r.get("card_violations") or 0 for r in rs),
                  card_gold_file_when_card=round(sum(r["card_gold_file"] for r in cards) / len(cards), 3) if cards else None,
                  used_card_when_card=round(sum(r["answer_used_card"] for r in cards) / len(cards), 3) if cards else None,
                  file_hit_when_card_has_gold=_rate([r["file_hit"] for r in cards if r["card_gold_file"]]),
                  file_hit_when_card_lacks_gold=_rate([r["file_hit"] for r in cards if not r["card_gold_file"]]))
        return st
    for arm in hint_arms:
        out["card"][arm] = card_stats(keys, arm)
        for repo in sorted({q.split("/")[0] for q in keys}):
            out["per_repo_card"].setdefault(repo, {})[arm] = card_stats({q for q in keys if q.startswith(repo + "/")}, arm)
    paired_keys = sorted(q for q in keys if all((q, a) in latest for a in arms))
    eq = [len({json.dumps(latest[(q, a)].get("hint_queries_list")) for a in hint_arms}) == 1 for q in paired_keys]
    out["identical_queries"] = {"issues": len(paired_keys), "identical_across_card_arms": sum(eq)}
    if paired_keys:
        cl = collections.defaultdict(lambda: collections.defaultdict(list))
        for q in paired_keys:
            cl[q.split("/", 1)[0]][min(gold_files[q])].append(q)
        clusters = {r: list(c.values()) for r, c in cl.items()}

        def v(arm, m, ks=paired_keys):
            return {q: float(latest[(q, arm)].get(m) or 0) for q in ks}
        # C4: card table per source, raw hits and card, gold file and symbol, paired between sources
        cm = ["retrieved_gold_file", "retrieved_gold_symbol", "card_gold_file", "card_gold_symbol"]
        out["card_table"] = {"per_source": {a: {m: cluster_bootstrap(v(a, m), clusters) for m in cm} for a in hint_arms},
                             "paired": {f"{a}-{b}": {m: cluster_bootstrap({q: v(a, m)[q] - v(b, m)[q] for q in paired_keys},
                                                                          clusters) for m in cm}
                                        for i, a in enumerate(hint_arms) for b in hint_arms[i + 1:]}}
        # C5: each card arm - A, split by whether that arm's card held the gold file
        if "A" in arms:
            split = {}
            for a in hint_arms:
                for label, cond in (("card held gold file", True), ("card lacked gold file", False)):
                    ks = [q for q in paired_keys if bool(latest[(q, a)].get("card_gold_file")) == cond]
                    if not ks:
                        continue
                    ccl = {r: [[q for q in c if q in ks] for c in cs if any(q in ks for q in c)] for r, cs in clusters.items()}
                    ccl = {r: cs for r, cs in ccl.items() if cs}
                    split[f"{a}: {label}"] = {"issues": len(ks), **{
                        m: {"arm": _mean(latest[(q, a)].get(m) for q in ks), "A": _mean(latest[(q, "A")].get(m) for q in ks),
                            "diff": cluster_bootstrap({q: float(latest[(q, a)].get(m) or 0) - float(latest[(q, "A")].get(m) or 0)
                                                       for q in ks}, ccl)}
                        for m in ("file_hit", "symbol_hit", "api_calls", "total_cost_alone_usd")}}
            out["card_arm_minus_A_by_card_gold"] = split
        # C6: disagreeing cards
        dis = {}
        for i, a in enumerate(hint_arms):
            for b in hint_arms[i + 1:]:
                for x, y in ((a, b), (b, a)):
                    ks = [q for q in paired_keys if latest[(q, x)].get("card_gold_file") and not latest[(q, y)].get("card_gold_file")]
                    if ks:
                        dis[f"{x} card has gold, {y} card lacks it"] = {
                            "issues": len(ks), f"{x}_file_hit": _mean(latest[(q, x)]["file_hit"] for q in ks),
                            f"{y}_file_hit": _mean(latest[(q, y)]["file_hit"] for q in ks),
                            "A_file_hit": _mean(latest[(q, "A")]["file_hit"] for q in ks) if "A" in arms else None}
        out["disagreeing_cards"] = dis
        # C10: Haiku queries naming an identifier absent from the issue text (before the hygiene filter drops them)
        ref = hint_arms[0]
        per_repo = collections.defaultdict(lambda: [0, 0])
        for q in paired_keys:
            r = latest[(q, ref)]
            per_repo[q.split("/")[0]][0] += r.get("query_invented_dropped") or 0
            per_repo[q.split("/")[0]][1] += r.get("query_model_queries") or 0
        out["queries_with_invented_identifier"] = {
            repo: {"model_queries": n, "with_identifier_absent_from_issue": k, "share": round(k / n, 3) if n else None,
                   "after_filter": 0} for repo, (k, n) in sorted(per_repo.items())}
    # position of each arm within its issue (by start time)
    pos = collections.defaultdict(list)
    for q in keys:
        order = sorted((latest[(q, a)]["start_utc"], a) for a in arms if (q, a) in latest)
        for i, (_, a) in enumerate(order):
            r = latest[(q, a)]
            pos[(a, i + 1)].append(r)
    out["cost_by_position"] = {f"{a} ran {i}": {"n": len(rs), "cost_usd": _mean(r["cost_usd"] for r in rs),
                                                "cost_alone_usd": _mean(r.get("cost_alone_usd") for r in rs),
                                                "cache_creation_tokens": _mean(r.get("cache_creation_tokens") for r in rs),
                                                "cache_read_tokens": _mean(r.get("cache_read_tokens") for r in rs)}
                               for (a, i), rs in sorted(pos.items())}
    # projection for 321 issues, and two extra card arms priced from H-tuned's hook and agent costs
    n_full = 321
    per_arm = {a: {"total_cost_usd": _mean(latest[(q, a)].get("total_cost_usd") for q in keys if (q, a) in latest),
                   "wall_s": _mean(latest[(q, a)]["wall_s"] for q in keys if (q, a) in latest)} for a in arms}
    workers = 4
    proj = {"issues": n_full, "workers": workers, "arms": {}}
    for a in arms:
        proj["arms"][a] = {"cost_usd": round(per_arm[a]["total_cost_usd"] * n_full, 1),
                           "hours": round(per_arm[a]["wall_s"] * n_full / workers / 3600, 2)}
    proj["total_cost_usd"] = round(sum(v["cost_usd"] for v in proj["arms"].values()), 1)
    proj["total_hours"] = round(sum(v["hours"] for v in proj["arms"].values()), 2)
    ref = "H-tuned" if "H-tuned" in per_arm else hint_arms[0]
    proj["extra_arms_priced_not_built"] = {
        name: {"cost_usd": round(per_arm[ref]["total_cost_usd"] * n_full, 1),
               "hours": round((per_arm[ref]["wall_s"] + extra_s) * n_full / workers / 3600, 2),
               "basis": f"{ref}'s measured agent + hook cost and wall time; {note}"}
        for name, extra_s, note in (
            ("card from BM25 alone", 0.0, "no embedding model to load; BM25 index build ~2-13 s per hook start "
                                          "(about the LFM load it replaces)"),
            ("card from BM25 + tuned Qwen3 (RRF)", 8.0, "Qwen3-0.6B CPU load plus the BM25 build add ~8 s per hook "
                                                        "start on PostHog (measured 13 s against LFM's ~5 s)"))}
    out["projection"] = proj
    return out


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(bools):
    return round(sum(bools) / len(bools), 3) if bools else None


def report(args):
    """Per repo and pooled, each on all questions and without the pilot's Saleor issues (the design
    was chosen after seeing them). Primary: cost per task, B-A / C-A / C-B, pooled, cluster bootstrap."""
    latest, gold_files, fqn_file, bench_dirs, reruns = {}, {}, {}, {}, {}
    pilot_keys, leak_keys = set(), set()
    for repo in args.repo:
        cfg = common.load_config(repo)
        d = bench_dir(args.out, cfg, args.name)
        rows, reruns[cfg["name"]] = load_latest(os.path.join(d, "runs.jsonl"), cfg["name"])
        latest.update(rows)
        for q in common.read_jsonl(os.path.join(d, "questions.jsonl")):
            gold_files[f"{cfg['name']}/{q['id']}"] = q["gold_files"]
        work = os.path.join(args.out, cfg["name"])
        fqn_file[cfg["name"]] = {s["fqn"]: s["file_path"] for s in common.load_symbols(work)}
        bench_dirs[(cfg["name"], "")] = d
        for src in {r.get("reused_from") for r in rows.values() if r.get("reused_from")}:
            bench_dirs[(cfg["name"], src)] = bench_dir(args.out, cfg, src)
        for r in rows.values():  # rows written before api_calls was recorded: count from the event log
            if r.get("api_calls") is None or r.get("first_cache_read") is None:
                ev = os.path.join(bench_dirs[(cfg["name"], r.get("reused_from") or "")], "events",
                                  f"{r['qid']}_{r['arm']}.events.jsonl")
                if os.path.exists(ev):
                    pe = parse_events(ev)
                    r["api_calls"], r["first_cache_read"] = pe["api_calls"], pe["first_cache_read"]
                    r["cost_alone_usd"] = alone(r.get("cost_usd"), r["first_cache_read"])
                    r["total_cost_alone_usd"] = round(r["cost_alone_usd"] + (r.get("hook_cost_usd") or 0), 6)
        pilot_q = os.path.join(bench_dir(args.out, cfg, args.pilot_bench), "questions.jsonl")
        if os.path.exists(pilot_q):
            pilot_keys |= {f"{cfg['name']}/{q['id']}" for q in common.read_jsonl(pilot_q) if q["set"] == args.issue_set}
        leak_keys |= {f"{cfg['name']}/{q['id']}" for q in common.read_jsonl(os.path.join(d, "questions.jsonl"))
                      if q.get("leak")}
    everything = {q for q, _ in latest}
    pops = {"all": everything, "without_pilot": everything - pilot_keys}
    if args.pilot_bench != "pilot":  # e.g. hint-full: the issues outside the hint pilot, and no-leak issues
        pops = {"all": everything, f"outside_{args.pilot_bench}": everything - pilot_keys,
                "no_leak": everything - leak_keys}
    out = {"bench": args.name, "repos": [common.load_config(r)["name"] for r in args.repo],
           "analysis_plan": analysis_plan(latest),
           "pilot_questions": len(pilot_keys & everything),
           "service_failure_reruns": {r: dict(c) for r, c in reruns.items()},
           "pooled": {name: summarize(latest, gold_files, pop) for name, pop in pops.items()},
           "per_repo": {}, "search_use_observational": {}}
    for repo in out["repos"]:
        keys = {q for q in everything if q.startswith(repo + "/")}
        out["per_repo"][repo] = {name: summarize(latest, gold_files, keys & pop) for name, pop in pops.items()}
    for name, pop in pops.items():
        rows = [r for (q, _), r in latest.items() if q in pop]
        out["search_use_observational"][name] = {
            "status": "descriptive only: split by the arm's own search use (a run searches when it is going badly)",
            "pooled": search_use(rows, gold_files, fqn_file, bench_dirs),
            **{repo: search_use([r for r in rows if r["repo"] == repo], gold_files, fqn_file, bench_dirs)
               for repo in out["repos"]}}
    # Sensitivity (analysis only, added after the run):
    # 1. only issues whose A, B and C runs all ran in the interleaved order;
    # 2. only issues where neither B nor C called search: what does having the tool change by itself?
    if any(a in HOOK_ARMS for _, a in latest):
        out["hint"] = {name: hint_report(latest, gold_files, pop, args) for name, pop in pops.items()}
    abc = [a for a in MAIN_ARMS if any(k[1] == a for k in latest)]
    if not ({"B", "C"} <= set(abc)):
        abc = []
    inter = {q for q in everything if all((q, a) in latest and latest[(q, a)]["batch"] == "interleaved" for a in abc)}
    nosearch = {q for q in everything if abc and all((q, a) in latest and not latest[(q, a)]["searches"]
                                                     for a in abc if ARMS[a])}
    out["sensitivity"] = {} if not abc else {
        "interleaved_only": {"issues_per_repo": dict(collections.Counter(q.split("/")[0] for q in inter)),
                             "pooled": summarize(latest, gold_files, inter, abc)},
        "no_search_in_B_and_C": {name: {"issues_per_repo": dict(collections.Counter(q.split("/")[0] for q in nosearch & pop)),
                                        "pooled": summarize(latest, gold_files, nosearch & pop, abc)}
                                 for name, pop in pops.items()}}
    path = os.path.join(args.out, "cc-bench", f"{args.name}.report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(path)


def order_log(args):
    """The actual run order of every row in this bench, oldest first, with its batch label:
    written to <out>/cc-bench/<name>.order.jsonl; prints counts per batch."""
    rows = []
    for repo in args.repo:
        cfg = common.load_config(repo)
        path = os.path.join(bench_dir(args.out, cfg, args.name), "runs.jsonl")
        for r in common.read_jsonl(path) if os.path.exists(path) else []:
            rows.append({"start_utc": r["start_utc"], "end_utc": r["end_utc"], "repo": cfg["name"], "qid": r["qid"],
                         "arm": r["arm"], "batch": batch_of(r), "status": r["status"],
                         "order_index": r.get("order_index"), "recovered": bool(r.get("recovered"))})
    rows.sort(key=lambda r: r["start_utc"])
    path = os.path.join(args.out, "cc-bench", f"{args.name}.order.jsonl")
    common.write_jsonl(path, rows)
    counts = collections.Counter((r["repo"], r["batch"]) for r in rows)
    print(json.dumps({"path": path, "sha256": common.file_sha256(path), "rows": len(rows),
                      "per_repo_batch": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())}}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prepare", prepare), ("sample", sample), ("canary", canary), ("run", run), ("report", report),
                     ("recover", recover), ("reuse", reuse), ("order", order_log)):
        p = sub.add_parser(name)
        if name in ("report", "order"):
            p.add_argument("--repo", required=True, nargs="+", help="one repo, or several to pool")
            p.add_argument("--pilot-bench", default="pilot", help="report: the pilot whose issues get their own "
                           "population (hint-pilot for hint-full, which then also reports no-leak issues)")
            p.add_argument("--issue-set", default="issues-v1", help="report: pilot questions of this set are "
                           "dropped in the without_pilot population")
        else:
            p.add_argument("--repo", required=True)
        p.add_argument("--out", default=common.default_out())
        p.set_defaults(func=fn)
        if name != "prepare":
            p.add_argument("--name", required=True, help="bench name, e.g. pilot")
        if name in ("canary", "run", "recover"):
            p.add_argument("--model", default=MODEL)
        if name in ("canary", "run"):
            p.add_argument("--arms", nargs="+", default=MAIN_ARMS, choices=list(ARMS))
        if name == "reuse":
            p.add_argument("--sources", nargs="+", required=True, help="earlier bench names on this repo")
        if name == "sample":
            p.add_argument("--take", nargs="+", required=True, help="SET:N, e.g. claude-m3:25 issues-v1:25")
            p.add_argument("--seed", type=int, default=20261112)
            p.add_argument("--force", action="store_true")
        if name == "run":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--timeout", type=int, default=900, help="seconds per run")
            p.add_argument("--seed", type=int, default=20261112, help="run order: issue order and arm order per issue")
            p.add_argument("--batch", default="interleaved", help="label stored on every row (run-order batch)")
            p.add_argument("--limit", type=int, default=0, help="first N questions only")
            p.add_argument("--max-utilization", type=float, default=0.985,
                           help="stop launching runs once the account's rate-limit window reaches this")
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
