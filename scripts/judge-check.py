"""Writer check: a blind Claude judge on samples of the test sets (design: ledger notes of 2026-11-12T11:51:58.611Z
and 11:52:22.138Z).

  .venv/bin/python scripts/judge-check.py build      # sample, controls, blinding -> out/judge/items.jsonl, key.jsonl
  .venv/bin/python scripts/judge-check.py judge [--workers 4]   # resumable -> out/judge/results.jsonl
  .venv/bin/python scripts/judge-check.py report     # per writer valid share (Wilson 95%), failure kinds, controls

Sample: 30 questions per writer and repo (random.Random(20261117) over each set's sorted question ids). Controls:
15 per writer, spread evenly over its repos, drawn after the main sample from the same set; each is shown with a
different symbol (Function/Method/Class) from the same file, drawn with the same generator. All items are shuffled
together. The judge sees only the question, the candidate symbol's name and code, and the whole file.
Valid = Q1 yes (the symbol answers the question) and Q2 no (no other symbol in the file answers it at least as well).
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

SEED = 20261117
JUDGE_MODEL = "claude-sonnet-5-5"
CELLS = {"claude": {"saleor": "claude-m3", "zulip": "claude-m3", "posthog": "claude-m3",
                    "dispatch": "claude-t400", "homeassistant": "claude-t400"},
         "qwen": {"saleor": "qwen-m4-400", "zulip": "qwen-m4-400", "posthog": "qwen-m4-400"},
         "lfm3": {"saleor": "lfm3-t400", "zulip": "lfm3-t400", "posthog": "lfm3-t400"}}
PER_CELL, CONTROLS = 30, 15
KINDS = ("Function", "Method", "Class")
PROMPT = """You are checking one item of a code-search benchmark. Below are a developer's question, one candidate \
symbol (its fully qualified name and its code), and the whole file that contains the symbol. Answer two factual \
questions about them.

Q1: Does this symbol answer the question? Answer yes or no, with one line of reason.
Q2: Only if Q1 is yes: does another symbol in the same file answer the question at least as well? Answer yes or no.

Reply with exactly one JSON object on a single line and nothing else:
{{"q1": "yes" or "no", "reason": "one line", "q2": "yes" or "no" (null when q1 is no)}}

QUESTION:
{question}

CANDIDATE SYMBOL: {fqn}
```python
{code}
```

WHOLE FILE ({path}):
```python
{file}
```
"""
OUT = common.default_out()
J = os.path.join(OUT, "judge")
REPOS = os.path.join(common.data_root(), "repos")


def jl(path):
    return [json.loads(l) for l in open(path)]


def build():
    rng = random.Random(SEED)
    items, key = [], []
    syms_cache = {}
    for writer, cells in CELLS.items():
        ctrl_per_repo = {r: CONTROLS // len(cells) + (1 if i < CONTROLS % len(cells) else 0) for i, r in enumerate(cells)}
        for repo, st in cells.items():
            if repo not in syms_cache:
                syms = jl(os.path.join(OUT, repo, "symbols.jsonl"))
                syms_cache[repo] = ({s["fqn"]: s for s in syms}, {})
                for s in syms:
                    if s["kind"] in KINDS:
                        syms_cache[repo][1].setdefault(s["file_path"], []).append(s["fqn"])
            by_fqn, by_file = syms_cache[repo]
            qs = sorted(jl(os.path.join(OUT, repo, "questions", f"{st}.jsonl")), key=lambda q: q["id"])
            ids = [q["id"] for q in qs]
            pick = rng.sample(ids, PER_CELL)
            rest = [i for i in ids if i not in set(pick)]
            qd = {q["id"]: q for q in qs}
            for qid in pick:
                q = qd[qid]
                g = (q.get("gold_fqns") or [q["gold_fqn"]])[0]
                items.append({"repo": repo, "question": q["question"], "fqn": g})
                key.append({"writer": writer, "set": st, "repo": repo, "kind": "main", "qid": qid, "fqn": g})
            n = 0
            for qid in rng.sample(rest, len(rest)):
                if n >= ctrl_per_repo[repo]:
                    break
                q = qd[qid]
                g = (q.get("gold_fqns") or [q["gold_fqn"]])[0]
                f = by_fqn[g]["file_path"]
                others = [x for x in by_file.get(f, []) if x != g and not x.startswith(g + ".") and not g.startswith(x + ".")]
                if not others:
                    continue
                o = rng.choice(sorted(others))
                items.append({"repo": repo, "question": q["question"], "fqn": o})
                key.append({"writer": writer, "set": st, "repo": repo, "kind": "control", "qid": qid, "fqn": o, "gold": g})
                n += 1
    order = list(range(len(items)))
    rng.shuffle(order)
    os.makedirs(J, exist_ok=True)
    with open(os.path.join(J, "items.jsonl"), "w") as fi, open(os.path.join(J, "key.jsonl"), "w") as fk:
        for k, i in enumerate(order):
            iid = f"item-{k + 1:04d}"
            fi.write(json.dumps(dict(items[i], id=iid)) + "\n")
            fk.write(json.dumps(dict(key[i], id=iid)) + "\n")
    print(f"{len(items)} items ({sum(1 for k in key if k['kind'] == 'main')} main, "
          f"{sum(1 for k in key if k['kind'] == 'control')} controls) -> {J}")


def prompt_for(it, by_fqn):
    s = by_fqn[it["repo"]][it["fqn"]]
    path = s["file_path"]
    text = open(os.path.join(REPOS, it["repo"], path), encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    code = "\n".join(lines[s["line_start"] - 1:s["line_end"]])
    return PROMPT.format(question=it["question"], fqn=it["fqn"], code=code, path=path, file=text)


def ask(prompt, log):
    pf = log[:-len(".jsonl")] + ".prompt.txt"  # the prompt goes through a file: whole files exceed the argv limit
    with open(pf, "w") as fh:
        fh.write(prompt)
    cmd = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude-ask"), "--model", JUDGE_MODEL,
           "--events", log, "--prompt-file", pf]
    p = subprocess.run(cmd, cwd=J, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout, p.stderr


def parse(out):
    out = out.strip()
    s, e = out.find("{"), out.rfind("}")
    if s < 0:
        return None
    try:
        d = json.loads(out[s:e + 1])
    except ValueError:
        return None
    q1 = str(d.get("q1", "")).lower()
    q2 = d.get("q2")
    q2 = str(q2).lower() if q2 is not None else None
    if q1 not in ("yes", "no") or (q1 == "yes" and q2 not in ("yes", "no")):
        return None
    return {"q1": q1, "q2": q2 if q1 == "yes" else None, "reason": d.get("reason")}


def judge(workers):
    items = jl(os.path.join(J, "items.jsonl"))
    res_path = os.path.join(J, "results.jsonl")
    done = {r["id"] for r in jl(res_path)} if os.path.exists(res_path) else set()
    by_fqn = {}
    for r in {it["repo"] for it in items}:
        by_fqn[r] = {s["fqn"]: s for s in jl(os.path.join(OUT, r, "symbols.jsonl"))}
    logs = os.path.join(J, "logs")
    os.makedirs(logs, exist_ok=True)
    todo = [it for it in items if it["id"] not in done]
    print(f"[judge] {len(todo)} of {len(items)} items to judge with {JUDGE_MODEL}, {workers} at a time", flush=True)

    def one(it):
        prompt = prompt_for(it, by_fqn)
        for attempt in range(1, 4):
            t0 = time.time()
            try:
                rc, out, err = ask(prompt, os.path.join(logs, f"{it['id']}.a{attempt}.jsonl"))
            except subprocess.TimeoutExpired:
                rc, out, err = -1, "", "timeout"
            except OSError as e:
                rc, out, err = -1, "", repr(e)
            if rc == 4:  # usage or rate limit: back off, then retry the same item
                time.sleep(600)
                continue
            v = parse(out) if rc == 0 else None
            if v:
                return dict(v, id=it["id"], attempt=attempt, seconds=round(time.time() - t0, 1), model=JUDGE_MODEL)
        return {"id": it["id"], "error": (err or out)[-300:], "model": JUDGE_MODEL}

    with cf.ThreadPoolExecutor(workers) as ex, open(res_path, "a") as fh:
        for n, r in enumerate(ex.map(one, todo), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if n % 10 == 0:
                print(f"[judge] {n}/{len(todo)}", flush=True)


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def report():
    key = {k["id"]: k for k in jl(os.path.join(J, "key.jsonl"))}
    res = {r["id"]: r for r in jl(os.path.join(J, "results.jsonl"))}
    out = {}
    for w in CELLS:
        main = [res[i] for i, k in key.items() if k["writer"] == w and k["kind"] == "main" and i in res and "q1" in res[i]]
        ctrl = [res[i] for i, k in key.items() if k["writer"] == w and k["kind"] == "control" and i in res and "q1" in res[i]]
        valid = sum(1 for r in main if r["q1"] == "yes" and r["q2"] == "no")
        rej = sum(1 for r in ctrl if not (r["q1"] == "yes" and r["q2"] == "no"))
        out[w] = {"main_judged": len(main), "valid": valid, "valid_share": round(valid / len(main), 3) if main else None,
                  "ci95": wilson(valid, len(main)),
                  "failures": {"does_not_answer": sum(1 for r in main if r["q1"] == "no"),
                               "another_symbol_as_good": sum(1 for r in main if r["q1"] == "yes" and r["q2"] == "yes")},
                  "controls_judged": len(ctrl), "controls_rejected": rej,
                  "controls_rejected_share": round(rej / len(ctrl), 3) if ctrl else None,
                  "void": (rej / len(ctrl) < 0.80) if ctrl else None,
                  "per_repo": {r: sum(1 for i, k in key.items() if k["writer"] == w and k["kind"] == "main" and k["repo"] == r
                                      and i in res and res[i].get("q1") == "yes" and res[i].get("q2") == "no")
                               for r in CELLS[w]},
                  "errors": sum(1 for i, k in key.items() if k["writer"] == w and i in res and "error" in res[i])}
    print(json.dumps(out, indent=1))
    with open(os.path.join(J, "report.json"), "w") as fh:
        json.dump(out, fh, indent=1)


VARIANTS = {  # name: (cells, seed, run dir)
    "writers": (CELLS, SEED, "judge"),
    # training questions: qwen-train1, 30 per repo + 15 controls; descriptive only
    "train": ({"qwen-train1": {"saleor": "qwen-train1", "zulip": "qwen-train1", "posthog": "qwen-train1"}}, 20261118,
              "judge-train"),
}


def main():
    global CELLS, SEED, J
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "judge", "report"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="writers")
    a = ap.parse_args()
    CELLS, SEED, sub = VARIANTS[a.variant]
    J = os.path.join(OUT, sub)
    {"build": build, "judge": lambda: judge(a.workers), "report": report}[a.cmd]()


if __name__ == "__main__":
    main()
