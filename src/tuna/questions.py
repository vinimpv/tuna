"""Write a test set: one LLM-written search question per randomly drawn file.

  tuna questions --repo saleor [--n 100] [--seed 20261111] [--name qwen-m1]
  tuna review    --repo saleor [--n 20] [--seed 20261112] [--name qwen-m1]

Files are drawn at random from the repo's symbols.jsonl, weighted by how many
symbols each file has, without replacement (Efraimidis-Spirakis keys from a
seeded RNG, so the order is a pure function of the seed). For each file the
writer model, through scripts/pi-ask (read-only tools), reads the file and
answers prompts/<prompt>.txt with one {"question", "gold_fqn"} object. A
draft is rejected, and the file retried up to --max-attempts times, when:

  unparseable          no single JSON object with question and gold_fqn, after
                       stripping think blocks and code fences
  multiple_golds       more than one object, or more than one gold
  gold_not_in_file     the gold is not a symbol of that file
  gold_is_constant     the gold is not a function, method or class
  leaks_identifier     the question contains the gold's own name: all of its
                       snake_case/CamelCase words, consecutively
  leaks_path           the question contains the file path (case-insensitive)
  too_short            fewer than 6 words
  gold_already_used    the gold was already accepted for another question
  pi_failed            pi exited non-zero
  pi_runaway           pi was stopped: more than --max-tool-calls tool calls or
                       longer than --timeout seconds (pi itself has no turn limit)
  gold_mismatch        (--gold-list) the reply names another gold than the given one

--writer picks the agent that writes the drafts, through a read-only ask
script with one interface (PROMPT in, answer on stdout, --events FILE):

  pi           scripts/pi-ask: pi with read, grep, find, ls on the served writer model
  claude-code  scripts/claude-ask: headless Claude Code with Read, Grep, Glob,
               --safe-mode (no CLAUDE.md, skills, plugins, hooks), default model

Sampling, prompt, filter and output fields are the same for both, so item i of
two sets with the same seed targets the same file. The reason names pi_failed
and pi_runaway stand for any writer. A claude-code failure (usage limit, API
error) is not a judgement on the file: the run stops dispatching, the file is
not recorded, and a rerun retries it.

With the v2 filter (prompt question-v2, or --filter v2) a question is also
rejected when it holds an identifier-like token (question_filter.py); the
reason names the rule and the detail lists every offending token:

  id_underscore, id_mixed_case, id_dotted, id_code_syntax, id_file_word

--workers N works on N files at once, each with its own pi process. Files are
handed out in sample order and the output is the first --n accepted files in
sample order, so the result does not depend on N.

Options for sets built around other sets:

  --only-files LIST        sample only these files (a test set on held-out files)
  --used-golds-from SET..  those sets' golds stay taken: each file's taken golds are
                           listed to the writer (the addendum) and rejected as
                           gold_already_used; files with nothing left are skipped
  --gold-list JSONL        fixed golds ({file, gold_fqn} rows, e.g. another set):
                           files in the list's order, one slot per gold, prompt
                           question-v2-gold (question-v2 with the symbol given); a
                           gold that fails its attempts loses only its own slot
  --split-like SET         train/val split by the files SET puts in val
  --max-cost USD           stop dispatching once the set's writer cost reaches USD

Every attempt's full pi event log is kept under questions/logs/<name>/ (its path is in the progress file only).
Progress is appended per file, so an interrupted run resumes where it stopped.
"""
import collections
import datetime
import hashlib
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time

from . import common, heldout, langs, metrics, question_filter

PROMPTS_DIR = os.path.join(common.PKG_DIR, "prompts")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(common.PKG_DIR)), "scripts")
DEFAULT_PI_ASK = os.path.join(SCRIPTS_DIR, "pi-ask")
# Writer backends: default ask script, and the event-log marker of one tool call.
WRITERS = {"pi": {"ask": os.environ.get("TUNA_PI_ASK", DEFAULT_PI_ASK), "tool_marker": b'"tool_execution_start"'},
           "claude-code": {"ask": os.environ.get("TUNA_CLAUDE_ASK", os.path.join(SCRIPTS_DIR, "claude-ask")),
                           "tool_marker": b'"type":"tool_use"'}}
SYMBOL_KINDS = {"Function", "Method", "Class"}
MIN_WORDS = 6
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")
# Validation rules per prompt version: v2 adds the identifier-token filter (question_filter.py).
FILTERS = {"question-v1": "v1", "question-v2": "v2", "question-bug1": "v2", "question-v2-gold": "v2"}


# ---------------------------------------------------------------- sampling

def file_order(syms, seed, exclude=frozenset()):
    """Every eligible file not in `exclude`, in a seeded weighted random order (weight =
    symbols in the file)."""
    weight = collections.Counter(s["file_path"] for s in syms)
    eligible = sorted({s["file_path"] for s in syms if s["kind"] in SYMBOL_KINDS} - set(exclude))
    rng = random.Random(seed)
    keyed = [(math.log(rng.random() or 1e-300) / weight[f], f) for f in eligible]
    return [f for _, f in sorted(keyed, key=lambda kf: (-kf[0], kf[1]))]


# ---------------------------------------------------------------- prompt

def load_prompt(version):
    path = os.path.join(PROMPTS_DIR, f"{version}.txt")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return text, hashlib.sha256(text.encode()).hexdigest()


def render(template, file_path, gold=None):
    module = langs.python_module_name(file_path)
    text = template.replace("{{FILE}}", file_path).replace("{{MODULE}}", module)
    return text.replace("{{GOLD}}", gold) if gold else text


# ---------------------------------------------------------------- validation

def words(text):
    """Lower-case words, with identifiers split on _ and CamelCase."""
    return [p.lower() for tok in re.findall(r"[A-Za-z0-9]+", text) for p in CAMEL.findall(tok)]


def contains_run(hay, needle):
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def parse_draft(text):
    """(object, None) or (None, (reason, detail))."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    body = "\n".join(l for l in text.strip().splitlines() if not FENCE.match(l)).strip()
    if not body:
        return None, ("unparseable", "empty reply")
    decoder, objs, i = json.JSONDecoder(), [], 0
    while i < len(body):
        if body[i].isspace():
            i += 1
            continue
        try:
            obj, i = decoder.raw_decode(body, i)
        except json.JSONDecodeError as e:
            return None, ("unparseable", f"{e.msg} at char {e.pos}: {body[:200]!r}")
        objs.append(obj)
    if len(objs) > 1:
        return None, ("multiple_golds", f"{len(objs)} JSON values in the reply")
    obj = objs[0]
    if not isinstance(obj, dict) or not isinstance(obj.get("question"), str):
        return None, ("unparseable", f"not an object with a string 'question': {body[:200]!r}")
    gold = obj.get("gold_fqn", obj.get("gold_fqns"))
    if isinstance(gold, list):
        if len(gold) != 1:
            return None, ("multiple_golds", f"{len(gold)} golds: {gold}")
        gold = gold[0]
    if not isinstance(gold, str) or not gold.strip():
        return None, ("unparseable", f"no string gold_fqn: {body[:200]!r}")
    gold = gold.strip()
    if re.search(r"[\s,;]", gold):
        return None, ("multiple_golds", f"gold_fqn holds several names: {gold!r}")
    return {"question": obj["question"].strip(), "gold_fqn": gold}, None


def validate(draft, file_path, file_syms, used):
    """None if the draft is acceptable, else (reason, detail). Checks run in the documented order."""
    gold, q = draft["gold_fqn"], draft["question"]
    sym = file_syms.get(gold)
    if sym is None:
        return "gold_not_in_file", f"{gold} is not a symbol of {file_path}"
    if sym["kind"] not in SYMBOL_KINDS:
        return "gold_is_constant", f"{gold} is a {sym['kind']}"
    name = gold.rsplit(".", 1)[-1]
    if contains_run(words(q), words(name)):
        return "leaks_identifier", f"question contains the words of {name!r}"
    if file_path.lower() in q.lower():
        return "leaks_path", f"question contains {file_path!r}"
    if len(q.split()) < MIN_WORDS:
        return "too_short", f"{len(q.split())} words"
    if gold in used:
        return "gold_already_used", f"{gold} was accepted as {used[gold]}"
    return None


# ---------------------------------------------------------------- pi

def _rel(root, path):
    p = os.path.realpath(os.path.join(root, path))
    return os.path.relpath(p, root) if p.startswith(root + os.sep) else p


def _in_tree(root, reads):
    """files_read: repo-relative paths that exist in the checkout, each once; reads outside the repo (the agent's
    own docs, other directories) and paths that do not exist are left out."""
    out = []
    for r in dict.fromkeys(reads):
        if not os.path.isabs(r) and not r.startswith("..") and os.path.exists(os.path.join(root, r)):
            out.append(r)
    return out


def _shown(path):
    """A script path as the ledger records it: relative to this checkout when inside it."""
    root = os.path.dirname(SCRIPTS_DIR)
    p = os.path.realpath(path) if path else path
    return os.path.relpath(p, root) if p and p.startswith(root + os.sep) else path


def read_claude_events(lines, root):
    """read_events for a Claude Code stream-json log. A message's content blocks
    arrive as separate events sharing the message id; usage is the result event's
    (main model only: Claude Code's own small helper calls are not the writer)."""
    reads, calls, models, usage, msg_ids = [], [], set(), collections.Counter(), set()
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = e.get("message") or {}
        if e.get("type") == "assistant" and not e.get("parent_tool_use_id"):
            msg_ids.add(m.get("id"))
            if m.get("model"):
                models.add(m["model"])
            for b in m.get("content") or []:
                if b.get("type") == "tool_use":
                    args = b.get("input") or {}
                    calls.append({"tool": b.get("name"), "args": args})
                    if b.get("name") == "Read" and isinstance(args.get("file_path"), str):
                        reads.append(_rel(root, args["file_path"]))
        elif e.get("type") == "result":
            for k, v in (e.get("usage") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    usage[k] += v
            if isinstance(e.get("total_cost_usd"), (int, float)):
                usage["cost_usd"] += e["total_cost_usd"]
    return {"files_read": _in_tree(root, reads), "tool_calls": calls, "models": sorted(models),
            "turns": len(msg_ids), "usage": dict(usage)}


def read_events(path, repo_root, writer="pi"):
    """Files read and tool calls made, writer model ids and token usage, from a pi json event log
    (or a Claude Code stream-json log for writer claude-code)."""
    reads, calls, models, usage, turns = [], [], set(), collections.Counter(), 0
    root = os.path.realpath(repo_root)
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        lines = []
    if writer == "claude-code":
        return read_claude_events(lines, root)
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "tool_execution_start":
            args = e.get("args") or {}
            calls.append({"tool": e.get("toolName"), "args": args})
            if e.get("toolName") == "read" and isinstance(args.get("path"), str):
                reads.append(_rel(root, args["path"]))
        m = e.get("message") or {}
        if e.get("type") == "message_end" and m.get("role") == "assistant":
            turns += 1
            if m.get("model"):
                models.add(m["model"])
            for k, v in (m.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] += v
    return {"files_read": _in_tree(root, reads), "tool_calls": calls, "models": sorted(models),
            "turns": turns, "usage": dict(usage)}


def ask(pi_ask, repo_root, prompt, events_path, timeout, max_tool_calls, tool_marker=b'"tool_execution_start"'):
    """Run one pi attempt. pi has no turn limit, so watch its event log and stop the
    whole process group when it makes more than max_tool_calls tool calls or runs
    past timeout seconds. Returns (returncode, stdout, stderr, seconds, runaway)."""
    t0, offset, calls, runaway = time.time(), 0, 0, None
    proc = subprocess.Popen([pi_ask, "--events", events_path, "--", prompt], cwd=repo_root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    while True:
        try:
            out, err = proc.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            pass
        try:
            with open(events_path, "rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
            cut = chunk.rfind(b"\n") + 1  # count whole lines only
            calls += chunk[:cut].count(tool_marker)
            offset += cut
        except OSError:
            pass
        if calls > max_tool_calls:
            runaway = f"{calls} tool calls, over the cap of {max_tool_calls}"
        elif time.time() - t0 > timeout:
            runaway = f"still running after {timeout}s ({calls} tool calls)"
        if runaway:
            for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
                try:
                    os.killpg(proc.pid, sig)
                    out, err = proc.communicate(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except ProcessLookupError:
                    out, err = proc.communicate()
                    break
            break
    return proc.returncode, out, err, time.time() - t0, runaway


# ---------------------------------------------------------------- run

def paths(work, name):
    qdir = os.path.join(work, "questions")
    return {"dir": qdir, "logs": os.path.join(qdir, "logs", name), "out": os.path.join(qdir, f"{name}.jsonl"),
            "progress": os.path.join(qdir, f"{name}.progress.jsonl"),
            "rejections": os.path.join(qdir, f"{name}.rejections.jsonl"),
            "manifest": os.path.join(qdir, f"{name}.manifest.json")}


def load_progress(p, settings):
    if not os.path.exists(p["progress"]):
        return []
    rows = common.read_jsonl(p["progress"])
    # Older runs predate "filter" (all v1) and "writer_backend" (all pi).
    stored = {"filter": "v1", "writer_backend": "pi", "exclude_files_sha256": None, "per_file": 1,
              "symbols_per_question": 0, "only_files_sha256": None, "used_golds_from": None,
              "gold_list_sha256": None, "split_like": None, "writer_engine": None,
              **rows[0].get("settings", {})} if rows else None
    if rows and stored != settings:
        sys.exit(f"{p['progress']} was written with other settings {rows[0].get('settings')}; "
                 "use another --name or delete it")
    return rows[1:]


# Rejection reasons that are the writer failing (no usable reply), not the question filters.
WRITER_FAILURES = {"unparseable", "pi_failed", "pi_runaway", "gold_mismatch", "gold_not_in_file"}


def reject_shares(attempts):
    """Shares of writer attempts rejected: by anything, by the question filters (identifier, leak, length,
    constant gold, reused gold), and by writer failures. The spot-check share is added by
    scripts/spot-check.py once a review file is ticked."""
    n = len(attempts)
    bad = [a["reason"] for a in attempts if a["reason"]]
    filt = [r for r in bad if r not in WRITER_FAILURES]
    return {"reject_share": round(len(bad) / n, 4) if n else None,
            "filter_reject_share": round(len(filt) / n, 4) if n else None,
            "writer_failure_share": round((len(bad) - len(filt)) / n, 4) if n else None}


def split_by_file(questions, fraction, seed):
    """Validation files, drawn in a seeded order until they hold `fraction` of the questions."""
    files = sorted({q["file"] for q in questions})
    random.Random(seed).shuffle(files)
    per_file = collections.Counter(q["file"] for q in questions)
    target, val, n = round(fraction * len(questions)), set(), 0
    for f in files:
        if n >= target:
            break
        val.add(f)
        n += per_file[f]
    return val


def split_like(questions, ref_path):
    """Validation files of a reference set (its "split" field): a set written on the same
    files then shares the reference's split. Files the reference does not have go to train."""
    val = {r["file"] for r in common.read_jsonl(ref_path) if r.get("split") == "val"}
    return {q["file"] for q in questions} & val


def summarize(cfg, p, settings, done, order_len, finished, guards=None, segment=None, heldout_files=frozenset(),
              val_fraction=0.0, split_ref=None):
    done = sorted(done, key=lambda r: r["index"])
    # The set is the first n accepted questions in sample order (file, then slot); ids follow that order.
    all_accepted = [q for r in done for q in record_questions(r)]
    accepted, surplus = all_accepted[:settings["n"]], all_accepted[settings["n"]:]
    width = max(3, len(str(settings["n"])))
    for rank, q in enumerate(accepted, 1):
        q.pop("log", None)   # progress records written before the log path left the released files
        q["id"] = f"{cfg['name']}-{settings['name']}-{rank:0{width}d}"
    leaks = [q["id"] for q in accepted if q["file"] in heldout_files]
    if leaks:  # cannot happen: held-out files are never sampled
        raise RuntimeError(f"questions on held-out files: {leaks[:5]}")
    splits = {}
    if (val_fraction or split_ref) and finished:
        val_files = (split_like(accepted, split_ref) if split_ref
                     else split_by_file(accepted, val_fraction, settings["seed"] + 1))
        for q in accepted:
            q["split"] = "val" if q["file"] in val_files else "train"
        for name in ("train", "val"):
            rows = [q for q in accepted if q["split"] == name]
            common.write_jsonl(os.path.join(p["dir"], f"{settings['name']}.{name}.jsonl"), rows)
            splits[name] = {"questions": len(rows), "files": len({q["file"] for q in rows})}
    attempts = [a for r in done for a in r["attempts"]]
    runs = []
    if os.path.exists(p["manifest"]):
        with open(p["manifest"]) as fh:
            runs = json.load(fh).get("runs", [])
    if segment:
        runs.append(segment)
    reasons = collections.Counter(a["reason"] for a in attempts if a["reason"])
    file_secs = [q["seconds"] for q in accepted]
    common.write_jsonl(p["out"], accepted)
    common.write_jsonl(p["rejections"], [{**{k: v for k, v in a.items() if k != "log"}, "file": r["file"]}
                                         for r in done for a in r["attempts"] if a["reason"]])
    manifest = dict(settings, guards=guards, runs=runs, surplus_accepted=[q["file"] for q in surplus], writer=sorted({m for a in attempts for m in a.get("models", [])}),
                    finished=finished, accepted=len(accepted), files_tried=len(done),
                    files_exhausted=sum(1 for r in done if not record_questions(r)), files_eligible=order_len,
                    files_with_questions=len({q["file"] for q in accepted}), heldout_files=len(heldout_files),
                    val_fraction=val_fraction, splits=splits,
                    attempts=len(attempts), rejections=dict(reasons.most_common()),
                    **reject_shares(attempts),
                    cost_usd=round(sum((a.get("usage") or {}).get("cost_usd", 0) for a in attempts), 4),
                    golds_given=sum(len(r.get("gold_slots") or []) for r in done) or None,
                    median_attempt_of_accepted=statistics.median([q["attempt"] for q in accepted]) if accepted else None,
                    median_seconds_per_accepted=round(statistics.median(file_secs), 1) if file_secs else None,
                    median_seconds_per_attempt=round(statistics.median([a["seconds"] for a in attempts]), 1) if attempts else None,
                    writer_seconds_total=round(sum(a["seconds"] for a in attempts), 1),
                    accepted_without_reading_target=sum(1 for q in accepted if q["file"] not in q["files_read"]),
                    questions_per_minute=[{"workers": r["workers"], "qpm": round(60 * r["accepted"] / r["wall_seconds"], 1)}
                                          for r in runs if r.get("wall_seconds") and r.get("accepted")],
                    updated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    with open(p["manifest"], "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def process_file(ctx, idx):
    """All questions and attempts for one sampled file, in a worker thread with its own
    writer processes. The file gets ctx["quota"][file] questions (one per slot), asked one
    after another so that every slot after the first can be told which symbols of the
    file already have a question (the prompt addendum). A slot gets --max-attempts
    attempts; when one slot runs out of attempts, the file is done."""
    args, cfg, p = ctx["args"], ctx["cfg"], ctx["paths"]
    file_path = ctx["order"][idx]
    if file_path in ctx["heldout"]:  # cannot happen: held-out files are not in the order
        raise RuntimeError(f"held-out file {file_path} was sampled")
    file_syms = ctx["by_file"][file_path]
    given = ctx["gold_slots"].get(file_path) if ctx["gold_slots"] else None
    base_prompt = render(ctx["template"], file_path)
    slug = file_path.replace("/", "__")
    quota = len(given) if given else ctx["quota"].get(file_path, 1)
    file_words = None
    record = {"index": idx, "file": file_path, "quota": quota, "attempts": [], "questions": [], "question": None}
    if given:
        record["gold_slots"] = list(given)
    writer = WRITERS[args.writer]
    for slot in range(1, quota + 1):
        # Symbols of this file that already have a question: earlier sets' golds (--used-golds-from), then this run's.
        taken = ctx["pre_taken"].get(file_path, []) + [q["gold_fqn"] for q in record["questions"]]
        prompt = base_prompt
        if given:  # --gold-list: the gold is fixed, the writer only writes the question
            prompt = render(ctx["template"], file_path, given[slot - 1])
        elif taken:
            prompt = base_prompt.rstrip("\n") + "\n\n" + ctx["addendum"].replace("{{FILE}}", file_path).replace(
                "{{TAKEN}}", "\n".join(f"- {g}" for g in taken)).rstrip("\n") + "\n"
        accepted = None
        for attempt in range(1, args.max_attempts + 1):
            if ctx["halt"]:
                return None
            stem = f"{idx:04d}_{slug}" + (f".s{slot}" if quota > 1 else "")
            log = os.path.join(p["logs"], f"{stem}.a{attempt}.jsonl")
            rc, stdout, stderr, secs, runaway = ask(ctx["ask"], cfg["path"], prompt, log, args.timeout,
                                                    args.max_tool_calls, writer["tool_marker"])
            ev = read_events(log, cfg["path"], args.writer)
            # "log" (the attempt's event log) stays in the progress file only (reject-shares.py reads it); the
            # released rejections and questions carry no log path
            a = {"slot": slot, "attempt": attempt, "seconds": round(secs, 2), "log": log,
                 "files_read": ev["files_read"], "tool_calls": len(ev["tool_calls"]), "turns": ev["turns"],
                 "models": ev["models"], "usage": ev["usage"], "reply": stdout[:2000], "reason": None,
                 "detail": None}
            draft = None
            if runaway:
                a["reason"], a["detail"] = "pi_runaway", runaway
            elif rc != 0:
                a["reason"], a["detail"] = "pi_failed", f"exit {rc}: {stderr.strip()[-500:]}"
                if args.writer == "claude-code":  # usage limit or API error: stop, keep the file for a rerun
                    with ctx["lock"]:
                        ctx["halt"] = f"{file_path} a{attempt}: {a['detail']}"
                        print(f"[questions] {idx:4d} {file_path} a{attempt} writer failed, stopping: {a['detail']}",
                              flush=True)
                    return None
            else:
                draft, err = parse_draft(stdout)
                bad = err
                if not bad and given and draft["gold_fqn"] != given[slot - 1]:
                    bad = ("gold_mismatch", f"answered {draft['gold_fqn']}, given {given[slot - 1]}")
                bad = bad or validate(draft, file_path, file_syms, {})
                if not bad and ctx["filter"] == "v2":
                    if file_words is None:
                        with open(os.path.join(cfg["path"], file_path), "rb") as fh:
                            file_words = question_filter.identifier_words(fh.read())
                    hits = question_filter.identifier_tokens(draft["question"], file_words)
                    if hits:
                        bad = ("id_" + hits[0][0], "; ".join(f"{r}: {t!r}" for r, t in hits))
                if not bad:
                    with ctx["lock"]:  # the only shared check: a gold may serve one question
                        holder = ctx["used"].get(draft["gold_fqn"])
                        if holder is None:
                            ctx["used"][draft["gold_fqn"]] = idx
                    if holder is not None:
                        bad = ("gold_already_used", f"{draft['gold_fqn']} was accepted for "
                               + (holder if isinstance(holder, str) else f"sampled file {holder}"))
                if bad:
                    a["reason"], a["detail"] = bad
            if draft and not a["reason"]:
                sym = file_syms[draft["gold_fqn"]]
                slot_secs = sum(x["seconds"] for x in record["attempts"] if x["slot"] == slot) + secs
                accepted = {
                    "id": None, "question": draft["question"], "gold_fqn": draft["gold_fqn"],
                    "gold_fqns": [draft["gold_fqn"]], "file": file_path, "kind": sym["kind"],
                    "line_start": sym["line_start"], "line_end": sym["line_end"],
                    "writer": ev["models"][0] if len(ev["models"]) == 1 else ev["models"],
                    "prompt_version": args.prompt, "prompt_sha256": ctx["prompt_sha"], "filter": ctx["filter"],
                    "seed": args.seed, "sample_index": idx, "slot": slot, "attempt": attempt,
                    "files_read": ev["files_read"], "writer_backend": args.writer, "seconds": round(slot_secs, 2)}
                if given:
                    accepted["gold_given"] = True
                elif taken:
                    accepted.update(prompt_addendum=ctx["addendum_name"], prompt_addendum_sha256=ctx["addendum_sha"])
            record["attempts"].append(a)
            status = "ok" if accepted else f"reject {a['reason']}"
            with ctx["lock"]:
                tag = f"s{slot}/{quota} " if quota > 1 else ""
                print(f"[questions] {idx:4d} {file_path} {tag}a{attempt} {secs:5.1f}s {status}", flush=True)
            if accepted:
                break
        if not accepted:
            if given:  # a fixed gold that failed loses only its own slot
                continue
            break
        record["questions"].append(accepted)
    record["question"] = record["questions"][0] if record["questions"] else None
    return record


def token_totals(attempts):
    """LLM token totals over attempts, from pi or Claude Code usage fields."""
    t = collections.Counter()
    for a in attempts:
        u = a.get("usage") or {}
        t["prompt"] += u.get("input", 0) + u.get("cacheRead", 0) + u.get("cacheWrite", 0) + u.get(
            "input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
        t["completion"] += u.get("output", 0) + u.get("output_tokens", 0)
        t["cache_read"] += u.get("cacheRead", 0) + u.get("cache_read_input_tokens", 0)
        t["reasoning"] += u.get("reasoning", 0)
        t["cost_usd"] += u.get("cost_usd", 0)
    return dict(t)


def record_questions(r):
    """Accepted questions of a progress record (older records hold one, under "question")."""
    if "questions" in r:
        return r["questions"]
    return [r["question"]] if r.get("question") else []


def _main(args, stage):
    import concurrent.futures as cf
    import threading
    cfg = common.load_config(args.repo)
    common.check_commit(cfg, args.allow_mismatch)
    work = common.workdir(args.out, cfg)
    syms = common.load_symbols(work)
    by_file = collections.defaultdict(dict)
    for s in syms:
        by_file[s["file_path"]][s["fqn"]] = s
    template, prompt_sha = load_prompt(args.prompt)
    qfilter = args.filter or FILTERS.get(args.prompt, "v1")
    settings = {"repo": cfg["name"], "name": args.name, "seed": args.seed, "prompt_version": args.prompt,
                "prompt_sha256": prompt_sha, "n": args.n, "max_attempts": args.max_attempts, "filter": qfilter,
                "writer_backend": args.writer,
                "exclude_files_sha256": common.file_sha256(args.exclude_files) if args.exclude_files else None,
                "per_file": args.per_file, "symbols_per_question": args.symbols_per_question,
                "only_files_sha256": common.file_sha256(args.only_files) if args.only_files else None,
                "used_golds_from": args.used_golds_from or None,
                "gold_list_sha256": common.file_sha256(args.gold_list) if args.gold_list else None,
                "split_like": args.split_like,
                # free-text label of the server that ran the writer model (one per set), passed in by the caller
                "writer_engine": os.environ.get("TUNA_WRITER_ENGINE") or None}
    p = paths(work, args.name)
    os.makedirs(p["logs"], exist_ok=True)
    done = load_progress(p, settings)
    if not os.path.exists(p["progress"]):
        common.write_jsonl(p["progress"], [{"settings": settings}])
    held = heldout.load_list(args.exclude_files) if args.exclude_files else set()
    # --used-golds-from: golds of earlier sets stay taken (told to the writer, rejected as gold_already_used).
    pre_used = {}
    for name in args.used_golds_from or []:
        for r in common.read_jsonl(paths(work, name)["out"]):
            for g in heldout.golds(r):
                pre_used.setdefault(g, f"set {name}")
    file_of = {s["fqn"]: s["file_path"] for s in syms}
    pre_taken = collections.defaultdict(list)
    for g in sorted(pre_used):
        if g in file_of:
            pre_taken[file_of[g]].append(g)
    gold_slots = None
    if args.gold_list:  # fixed golds: files in the list's order, one slot per gold
        gold_slots = collections.OrderedDict()
        for r in common.read_jsonl(args.gold_list):
            gold_slots.setdefault(r["file"], []).append(r["gold_fqn"])
        bad = [(f, g) for f, gs in gold_slots.items() for g in gs
               if file_of.get(g) != f or by_file[f][g]["kind"] not in SYMBOL_KINDS or f in held or g in pre_used]
        dup = [g for g, c in collections.Counter(g for gs in gold_slots.values() for g in gs).items() if c > 1]
        if bad or dup:
            sys.exit(f"--gold-list: {len(bad)} golds not usable (not in file, not a function/method/class, "
                     f"held out or already used), e.g. {bad[:3]}; {len(dup)} duplicates, e.g. {dup[:3]}")
        order = list(gold_slots)
    else:
        exclude = set(held)
        if args.only_files:
            exclude |= {s["file_path"] for s in syms} - heldout.load_list(args.only_files)
        order = file_order(syms, args.seed, exclude=exclude)
    eligible = collections.Counter(s["file_path"] for s in syms
                                   if s["kind"] in SYMBOL_KINDS and s["fqn"] not in pre_used)
    if pre_used and not gold_slots:  # files whose symbols are all taken cannot get a question
        order = [f for f in order if eligible[f]]
    quota = {f: (min(args.per_file, eligible[f], max(1, math.ceil(eligible[f] / args.symbols_per_question)))
                 if args.symbols_per_question else 1) for f in order}
    addendum, addendum_sha = load_prompt(args.addendum)
    ctx = {"args": args, "cfg": cfg, "paths": p, "order": order, "by_file": by_file, "template": template,
           "prompt_sha": prompt_sha, "filter": qfilter, "lock": threading.Lock(), "halt": None,
           "ask": args.ask or (args.pi_ask if args.writer == "pi" else WRITERS[args.writer]["ask"]),
           "heldout": held, "quota": quota, "addendum": addendum, "addendum_name": args.addendum,
           "addendum_sha": addendum_sha, "gold_slots": gold_slots, "pre_taken": pre_taken,
           "used": dict(pre_used, **{q["gold_fqn"]: r["index"] for r in done for q in record_questions(r)})}
    n_accepted = sum(len(record_questions(r)) for r in done)
    cost = sum((a.get("usage") or {}).get("cost_usd", 0) for r in done for a in r["attempts"])
    done_idx = {r["index"] for r in done}
    pending = collections.deque(i for i in range(len(order)) if i not in done_idx)
    t_start, seg_files, seg_accepted, seg_records = time.time(), 0, 0, []
    print(f"[questions] {cfg['name']}: {len(order)} eligible files, seed {args.seed}, prompt {args.prompt} "
          f"({prompt_sha[:12]}), filter {qfilter}, writer {args.writer}, {args.workers} workers, "
          f"{n_accepted} accepted from {len(done)} files so far", flush=True)

    def can_submit():
        if args.max_cost and cost >= args.max_cost and not ctx["halt"]:
            ctx["halt"] = f"budget: ${cost:.2f} spent on this set, --max-cost {args.max_cost}"
        return (n_accepted < args.n and pending and not ctx["halt"]
                and not (args.max_seconds and time.time() - t_start > args.max_seconds))

    # Files are handed out strictly in sample order and all in-flight files are
    # finished before stopping, so every file up to the last one dispatched is
    # decided: the output (first n accepted, in sample order) does not depend on
    # how many workers ran or in which order they finished.
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        running = {}
        while running or can_submit():
            while len(running) < args.workers and can_submit():
                idx = pending.popleft()
                running[ex.submit(process_file, ctx, idx)] = idx
            finished_now, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for fut in finished_now:
                running.pop(fut)
                record = fut.result()
                if record is None:  # writer failed: not recorded, retried on a rerun
                    continue
                with ctx["lock"]:
                    with open(p["progress"], "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                done.append(record)
                seg_records.append(record)
                seg_files += 1
                n_accepted += len(record["questions"])
                seg_accepted += len(record["questions"])
                cost += sum((a.get("usage") or {}).get("cost_usd", 0) for a in record["attempts"])

    budget_stop = bool(ctx["halt"]) and ctx["halt"].startswith("budget:")
    if budget_stop:  # a budget stop ends dispatch cleanly: every recorded file is decided
        print(f"[questions] {ctx['halt']}", flush=True)
    finished = not ctx["halt"] and (n_accepted >= args.n or not pending)
    seg_attempts = [a for r in seg_records for a in r["attempts"]]
    stage.config(**settings)
    stage.counts(items=seg_accepted, questions=seg_accepted, files=seg_files, attempts=len(seg_attempts),
                 accepted_total=n_accepted, rejections=dict(collections.Counter(a["reason"] for a in seg_attempts
                                                                                if a["reason"])))
    stage.tokens(**token_totals(seg_attempts))
    stage.extra(writer_models=sorted({m for a in seg_attempts for m in a.get("models", [])}), workers=args.workers,
                cost_usd=round(sum((a.get("usage") or {}).get("cost_usd", 0) for a in seg_attempts), 4),
                set_cost_usd=round(cost, 4))
    stage.outputs(p["out"], p["progress"])
    segment = {"started": datetime.datetime.fromtimestamp(t_start, datetime.timezone.utc).isoformat(timespec="seconds"),
               "workers": args.workers, "wall_seconds": round(time.time() - t_start, 1), "files": seg_files,
               "accepted": seg_accepted}
    m = summarize(cfg, p, settings, done, len(order), finished,
                  {"timeout_s": args.timeout, "max_tool_calls": args.max_tool_calls}, segment, held,
                  args.val_fraction, paths(work, args.split_like)["out"] if args.split_like else None)
    print(f"[questions] {cfg['name']}: {m['accepted']}/{args.n} accepted from {m['files_tried']} files, "
          f"{m['attempts']} attempts, rejections {m['rejections']}; this run {seg_accepted} accepted in "
          f"{segment['wall_seconds']}s with {args.workers} workers", flush=True)
    if budget_stop:
        sys.exit(5)
    if ctx["halt"]:
        print(f"[questions] stopped: writer failed ({ctx['halt']}); run again to resume", flush=True)
        sys.exit(4)
    if not finished:
        print(f"[questions] stopped after --max-seconds {args.max_seconds}; run again to resume", flush=True)
        sys.exit(3)
    print(f"[questions] wrote {p['out']}", flush=True)


def main(args):
    config = metrics.config_of(args)
    for k in ("pi_ask", "ask"):
        if config.get(k):
            config[k] = _shown(config[k])
    with metrics.Stage('questions', args.repo, getattr(args, "writer", None), config) as stage:
        return _main(args, stage)


def add_parser(sub):
    p = sub.add_parser("questions", help="LLM-written test questions, one per randomly drawn file")
    p.add_argument("--n", type=int, default=100, help="questions to accept")
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--name", default="qwen-m1", help="output name under questions/")
    p.add_argument("--prompt", default="question-v1", help="prompt version, a file under tuna/prompts/")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--timeout", type=int, default=180, help="seconds per pi attempt before it is stopped")
    p.add_argument("--max-tool-calls", type=int, default=40, help="tool calls per pi attempt before it is stopped")
    p.add_argument("--max-seconds", type=int, default=0, help="stop cleanly after this long (0 = no limit); "
                                                               "a rerun resumes")
    p.add_argument("--workers", type=int, default=4, help="files worked on in parallel, each with its own pi")
    p.add_argument("--filter", choices=["v1", "v2"], help="validation rules (default: by prompt version)")
    p.add_argument("--writer", choices=sorted(WRITERS), default="pi", help="agent that writes the drafts")
    p.add_argument("--exclude-files", help="file list (tuna heldout) never to sample: test-set files")
    p.add_argument("--per-file", type=int, default=1, help="questions per file at most (distinct symbols)")
    p.add_argument("--symbols-per-question", type=float, default=0,
                   help="with --per-file: a file gets one question per this many eligible symbols (0 = one per file)")
    p.add_argument("--addendum", default="question-v2-more",
                   help="prompt addendum for a file's later questions (lists the symbols already taken)")
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="hold out this share of questions as a validation split, grouped by file")
    p.add_argument("--only-files", help="file list to sample from, and no other (e.g. heldout-files.txt for a test set)")
    p.add_argument("--used-golds-from", nargs="+", metavar="SET",
                   help="sets under questions/ whose golds stay taken: listed to the writer, rejected if chosen again")
    p.add_argument("--gold-list", help="jsonl of {file, gold_fqn} (e.g. another set): one question per given gold, "
                                       "files in list order; use with a prompt that has {{GOLD}} (question-v2-gold)")
    p.add_argument("--split-like", metavar="SET", help="train/val split by the files of this set's split field")
    p.add_argument("--max-cost", type=float, default=0, help="stop dispatching once this set's writer cost "
                                                             "(USD, from usage) reaches this; a rerun resumes")
    p.add_argument("--ask", help="ask script for the writer (default scripts/pi-ask or scripts/claude-ask)")
    p.add_argument("--pi-ask", default=os.environ.get("TUNA_PI_ASK", DEFAULT_PI_ASK))
    p.set_defaults(func=main)
    return p


# ---------------------------------------------------------------- review

def review_main(args):
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    p = paths(work, args.name)
    qs = common.read_jsonl(p["out"])
    syms = {s["fqn"]: s for s in common.load_symbols(work)}
    pick = sorted(random.Random(args.seed).sample(range(len(qs)), min(args.n, len(qs))))
    tag = args.name.rsplit("-", 1)[-1]  # qwen-m1 -> m1
    out = args.review_out or os.path.join(args.out, "review", f"{tag}-{cfg['name']}.md")
    lines = [f"# {cfg['name']}: {len(pick)} questions to review", "",
             f"Picked at random (seed {args.seed}) from {len(qs)} accepted questions in `{args.name}`. "
             f"Writer: {qs[0]['writer'] if qs else '?'}, prompt {qs[0]['prompt_version'] if qs else '?'}.",
             "", "Tick every box that applies.", ""]
    for k, i in enumerate(pick, 1):
        q = qs[i]
        sym = syms[q["gold_fqn"]]
        src = sym["body"].split("\n")[2:]  # drop the FQN/File header, shown above
        shown = src[:args.max_lines]
        lines += ["---", "", f"## {k}. {q['id']}", "", f"**Question:** {q['question']}", "",
                  f"**Gold:** `{q['gold_fqn']}`  ", f"{sym['kind']} in `{q['file']}`, lines {q['line_start']}–{q['line_end']}",
                  "", "```python", *shown, "```"]
        if len(src) > len(shown):
            lines += ["", f"*Source cut: showing {len(shown)} of {len(src)} lines.*"]
        lines += ["", "- [ ] good", "- [ ] too vague (several symbols fit)", "- [ ] gives the answer away", ""]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[review] {len(pick)} questions -> {out}", flush=True)


def add_review_parser(sub):
    p = sub.add_parser("review", help="markdown review sheet for a sample of accepted questions")
    p.add_argument("--name", default="qwen-m1")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--max-lines", type=int, default=60)
    p.add_argument("--review-out", help="output path (default <out>/review/<m1|m2...>-<repo>.md, from --name)")
    p.set_defaults(func=review_main)
    return p
