"""Follow-up analysis of a cc-bench run from saved rows and event logs only (no Claude calls).
Needs the agents' event logs, which are not in this release; the reports it wrote are in data/out/cc-bench/.

  .venv/bin/python scripts/cc-analysis.py cache  --name issues     # cache cells (below)
  .venv/bin/python scripts/cc-analysis.py replay --name issues     # query replay (CPU query encoding)
  .venv/bin/python scripts/cc-analysis.py extra                    # extra cells, added after the results
  .venv/bin/python scripts/cc-analysis.py setup-check              # arm A set-up in both runs, field by field
  .venv/bin/python scripts/cc-analysis.py search-split             # first run split by search use (ledger rule)
  .venv/bin/python scripts/cc-analysis.py instruction-line         # hint run, issues with an instruction-line query
  .venv/bin/python scripts/cc-analysis.py dirty-tree               # first run rows recorded with a dirty tree

cache:
  cache_per_arm, B_C_order  cache-write and cache-read tokens per run and on the first API call, per arm; for B and
      C split by which of the pair ran first on that issue (B and C share the tool list and the
      first message, so the second can read what the first wrote; A shares with no other arm)
  cache_immune  B-A, C-A, C-B (and D pairs) on measures the cache cannot touch: API calls, tool calls by
      type, output tokens, input-side tokens (input + cache read + cache write); overall and on
      issues where neither B nor C searched
  cache_lifetime  cache lifetime: 5-minute against 1-hour cache-write tokens per arm and batch
  D_by_search  D's gain split by whether the D run searched, with A and C on the same issues
replay:
  every search query the agent wrote in B, C and D, re-run against the stock LFM, tuned LFM
      and BM25 + tuned Qwen3 (RRF) indexes; gold file in the top 10, paired by query

Intervals: the cc-bench cluster bootstrap (gold-file clusters resampled within repo).
Writes <out>/cc-bench/<name>.cache.json / .replay.json.
"""
import argparse
import collections
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
_spec = importlib.util.spec_from_file_location("cc_bench", os.path.join(ROOT, "scripts", "cc-bench.py"))
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

from tuna import common  # noqa: E402

REPOS = ["saleor", "zulip", "posthog"]


def load(out, name):
    """Final rows keyed (qkey, arm) with event-derived fields, and gold files per qkey."""
    rows, gold = {}, {}
    for repo in REPOS:
        cfg = common.load_config(repo)
        d = cc.bench_dir(out, cfg, name)
        latest, _ = cc.load_latest(os.path.join(d, "runs.jsonl"), cfg["name"])
        for q in common.read_jsonl(os.path.join(d, "questions.jsonl")):
            gold[f"{cfg['name']}/{q['id']}"] = q["gold_files"]
        for k, r in latest.items():
            src = cc.bench_dir(out, cfg, r["reused_from"]) if r.get("reused_from") else d
            r["events_dir"] = os.path.join(src, "events")
            r.update(per_call(os.path.join(r["events_dir"], f"{r['qid']}_{r['arm']}.events.jsonl")))
            tc = r["tool_calls"]
            r.update(grep=tc.get("Grep", 0), read=tc.get("Read", 0), glob=tc.get("Glob", 0),
                     search=tc.get("mcp__tuna__search", 0),
                     input_side_tokens=(r.get("input_tokens") or 0) + (r.get("cache_read_tokens") or 0)
                     + (r.get("cache_creation_tokens") or 0))
            rows[k] = r
    return rows, gold


def per_call(path):
    """Usage per API call (assistant messages deduplicated by id) and the run's cache-write split."""
    calls, result = {}, None
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "assistant":
            m = e["message"]
            calls[m.get("id")] = m.get("usage") or {}
        elif e.get("type") == "result":
            result = e
    us = list(calls.values())
    first = us[0] if us else {}
    cc_split = ((result or {}).get("usage") or {}).get("cache_creation") or {}
    return {"api_calls": len(us),
            "first_cache_write": first.get("cache_creation_input_tokens", 0),
            "first_cache_read": first.get("cache_read_input_tokens", 0),
            "write_5m": cc_split.get("ephemeral_5m_input_tokens", 0),
            "write_1h": cc_split.get("ephemeral_1h_input_tokens", 0),
            "calls_with_5m": sum(1 for u in us if (u.get("cache_creation") or {}).get("ephemeral_5m_input_tokens"))}


def clusters_for(keys, gold):
    cl = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in keys:
        cl[q.split("/", 1)[0]][min(gold[q])].append(q)
    return {r: list(c.values()) for r, c in cl.items()}


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def paired(rows, gold, keys, a, b, metrics):
    cl = clusters_for(keys, gold)
    return {m: cc.cluster_bootstrap({q: float(rows[(q, a)][m] or 0) - float(rows[(q, b)][m] or 0) for q in keys}, cl)
            for m in metrics}


def cache(args):
    rows, gold = load(args.out, args.name)
    qkeys = sorted({q for q, _ in rows})
    arms = sorted({a for _, a in rows})
    out = {"bench": args.name, "issues": len(qkeys)}

    # cache_per_arm, B_C_order
    a1 = {}
    for arm in arms:
        rs = [rows[(q, arm)] for q in qkeys]
        a1[arm] = {m: mean(r[m] or 0 for r in rs) for m in
                   ("cache_creation_tokens", "cache_read_tokens", "first_cache_write", "first_cache_read", "cost_usd")}
        a1[arm]["first_call_fully_cached"] = mean(r["first_cache_write"] == 0 for r in rs)
    order = {}
    for q in qkeys:
        b, c = rows[(q, "B")], rows[(q, "C")]
        order[q] = "B" if b["start_utc"] < c["start_utc"] else "C"
    split = {}
    for arm in ("B", "C"):
        for pos in ("first", "second"):
            ks = [q for q in qkeys if (order[q] == arm) == (pos == "first")]
            rs = [rows[(q, arm)] for q in ks]
            split[f"{arm} ran {pos}"] = {"n": len(ks), **{m: mean(r[m] or 0 for r in rs) for m in
                                         ("first_cache_write", "first_cache_read", "cache_creation_tokens",
                                          "cache_read_tokens", "cost_usd", "api_calls", "output_tokens")},
                                         "first_call_fully_cached": mean(r["first_cache_write"] == 0 for r in rs)}
    import datetime

    def ts(s):
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    gaps = [abs((ts(rows[(q, "B")]["start_utc"]) - ts(rows[(q, "C")]["start_utc"])).total_seconds()) for q in qkeys]
    a1_b_c = {"pairs": len(qkeys), "B_first": sum(o == "B" for o in order.values()),
              "gap_s_median": sorted(gaps)[len(gaps) // 2], "gap_over_1h": sum(g > 3600 for g in gaps),
              "gap_over_5m": sum(g > 300 for g in gaps), "split": split}
    # the same split for A against itself is meaningless; A's first call vs B/C first call is the contrast
    out["cache_per_arm"] = a1
    out["B_C_order"] = a1_b_c

    # cache_immune
    immune = ["api_calls", "tool_calls_total", "grep", "read", "glob", "search", "output_tokens", "input_side_tokens"]
    nosearch = [q for q in qkeys if not rows[(q, "B")]["searches"] and not rows[(q, "C")]["searches"]]
    pairs = [("B", "A"), ("C", "A"), ("C", "B")] + ([("D", "A"), ("D", "C")] if "D" in arms else [])
    out["cache_immune"] = {"no_search_in_B_and_C_status": "descriptive only, except C-B there (a null test: B and C "
                                                             "are identical when neither searches)"}
    for label, keys in (("all", qkeys), ("no_search_in_B_and_C", nosearch)):
        out["cache_immune"][label] = {
            "n": len(keys),
            "means": {arm: {m: mean(rows[(q, arm)][m] or 0 for q in keys) for m in immune} for arm in arms},
            "paired": {f"{a}-{b}": paired(rows, gold, keys, a, b, immune + ["cost_usd"]) for a, b in pairs}}

    # cache_lifetime
    a3 = collections.defaultdict(lambda: {"runs": 0, "write_5m": 0, "write_1h": 0, "runs_with_5m": 0})
    for (q, arm), r in rows.items():
        k = f"{arm}/{r['batch']}"
        a3[k]["runs"] += 1
        a3[k]["write_5m"] += r["write_5m"]
        a3[k]["write_1h"] += r["write_1h"]
        a3[k]["runs_with_5m"] += r["write_5m"] > 0 or r["calls_with_5m"] > 0
    out["cache_lifetime"] = dict(sorted(a3.items()))

    # D_by_search
    if "D" in arms:
        a5 = {}
        for label, keys in (("D searched", [q for q in qkeys if rows[(q, "D")]["searches"]]),
                            ("D did not search", [q for q in qkeys if not rows[(q, "D")]["searches"]])):
            a5[label] = {"n": len(keys),
                         "means": {arm: {m: mean(rows[(q, arm)][m] or 0 for q in keys)
                                         for m in ("file_hit", "symbol_hit", "cost_usd")} for arm in ("A", "C", "D")},
                         "paired": {f"D-{b}": paired(rows, gold, keys, "D", b, ["file_hit", "symbol_hit", "cost_usd"])
                                    for b in ("A", "C")}}
        out["D_by_search"] = {"status": "descriptive only: split by D's own search use", **a5}
    path = os.path.join(args.out, "cc-bench", f"{args.name}.cache.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(path)


REPLAY_INDEXES = [("stock LFM", "lfm25-emb", False), ("tuned LFM", "lfm25-ft-lora", False),
                  ("BM25 + tuned Qwen3", "qwen3-0.6b-ft-lora", True)]


def replay(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # query encoding on the CPU; no corpus is encoded
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from tuna import mcp
    rows, gold = load(args.out, args.name)
    queries = []  # one per search call the agent made
    for (q, arm), r in sorted(rows.items()):
        if arm not in ("B", "C", "D") or not r["searches"]:
            continue
        p = os.path.join(r["events_dir"], f"{r['qid']}_{arm}.search.jsonl")
        for i, s in enumerate(common.read_jsonl(p) if os.path.exists(p) else []):
            queries.append({"qkey": q, "repo": r["repo"], "arm": arm, "i": i, "query": s["query"], "k_asked": s["k"],
                            "file_hit_run": r["file_hit"]})
    fqn_file = {}
    for repo in REPOS:
        work = os.path.join(args.out, repo)
        fqn_file[repo] = {s["fqn"]: s["file_path"] for s in common.load_symbols(work)}
        for label, tag, fuse in REPLAY_INDEXES:
            idx = mcp.load_index(args.out, repo, tag, args.threads, fuse)[0]
            for qq in queries:
                if qq["repo"] != repo:
                    continue
                hits = idx.search(qq["query"], 10)
                files = {fqn_file[repo][h["fqn"]] for h in hits}
                qq[label] = bool(files & set(gold[qq["qkey"]]))
            del idx
            print(f"[replay] {repo} {label} done", flush=True)
    out = {"bench": args.name, "queries": len(queries), "k": 10,
           "per_writer_arm": {}, "paired": {}}
    labels = [l for l, _, _ in REPLAY_INDEXES]
    for arm in ("B", "C", "D", "all"):
        qs = [q for q in queries if arm == "all" or q["arm"] == arm]
        out["per_writer_arm"][arm] = {"queries": len(qs), **{l: mean(q[l] for q in qs) for l in labels}}
    # paired by query, intervals by gold-file cluster (all queries of an issue's gold file move together)
    cl = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, q in enumerate(queries):
        cl[q["repo"]][min(gold[q["qkey"]])].append(i)
    clusters = {r: list(c.values()) for r, c in cl.items()}
    for a, b in [("tuned LFM", "stock LFM"), ("BM25 + tuned Qwen3", "tuned LFM"), ("BM25 + tuned Qwen3", "stock LFM")]:
        out["paired"][f"{a} - {b}"] = cc.cluster_bootstrap({i: float(q[a]) - float(q[b]) for i, q in enumerate(queries)},
                                                           clusters)
    for l in labels:
        out.setdefault("rate", {})[l] = cc.cluster_bootstrap({i: float(q[l]) for i, q in enumerate(queries)}, clusters)
    out["by_repo"] = {repo: {"queries": sum(q["repo"] == repo for q in queries),
                             **{l: mean(q[l] for q in queries if q["repo"] == repo) for l in labels}} for repo in REPOS}
    out["k_asked"] = dict(collections.Counter(q["k_asked"] for q in queries))
    path = os.path.join(args.out, "cc-bench", f"{args.name}.replay.json")
    with open(path, "w") as fh:
        json.dump({**out, "rows": queries}, fh, indent=1)
    print(path)


def hint_rows(out, bench="hint-pilot"):
    rows, gold, leak = {}, {}, {}
    for repo in REPOS:
        cfg = common.load_config(repo)
        d = cc.bench_dir(out, cfg, bench)
        latest, _ = cc.load_latest(os.path.join(d, "runs.jsonl"), cfg["name"])
        rows.update(latest)
        for q in common.read_jsonl(os.path.join(d, "questions.jsonl")):
            gold[f"{cfg['name']}/{q['id']}"] = q["gold_files"]
            leak[f"{cfg['name']}/{q['id']}"] = bool(q.get("leak"))
    return rows, gold, leak


def extra(args):
    """Extra cells from saved runs, added after the results. Splits by an arm's own search use are descriptive only
    (a run searches when it is going badly); the neither-searched C-B null test is the exception."""
    rows, gold = load(args.out, args.name)
    qkeys = sorted({q for q, _ in rows})
    out = {"label": "dollars = cost the Claude Code client reports at API list prices"}
    # like-for-like dollars: an arm on the issues where it ran first (no other arm's cache) against A
    order = {q: "B" if rows[(q, "B")]["start_utc"] < rows[(q, "C")]["start_utc"] else "C" for q in qkeys}
    b1 = {}
    for arm in ("B", "C"):
        ks = [q for q in qkeys if order[q] == arm]
        b1[f"{arm}-first - A"] = {"issues": len(ks), "means": {a: mean(rows[(q, a)]["cost_usd"] for q in ks) for a in ("A", arm)},
                                  **paired(rows, gold, ks, arm, "A", ["cost_usd", "api_calls", "tool_calls_total",
                                                                      "output_tokens"])}
    out["like_for_like_dollars"] = b1
    # tool available: (B + C)/2 - A
    cl = clusters_for(qkeys, gold)
    out["tool_available"] = {"issues": len(qkeys), **{
        m: cc.cluster_bootstrap({q: (float(rows[(q, "B")][m]) + float(rows[(q, "C")][m])) / 2 - float(rows[(q, "A")][m])
                                 for q in qkeys}, cl) for m in ("api_calls", "tool_calls_total", "output_tokens")}}
    # null test kept: neither B nor C searched, C - B
    ns = [q for q in qkeys if not rows[(q, "B")]["searches"] and not rows[(q, "C")]["searches"]]
    out["null_test_C_minus_B_neither_searched"] = {"issues": len(ns), **paired(
        rows, gold, ns, "C", "B", ["api_calls", "tool_calls_total", "output_tokens", "cost_usd", "file_hit"])}
    out["descriptive_only"] = ["search_use_observational (with/without search split)", "D_by_search",
                                  "sensitivity.no_search_in_B_and_C (other than the C-B null test)"]
    # the hint pilot
    hrows, hgold, hleak = hint_rows(args.out)
    hkeys = sorted({q for q, _ in hrows})
    b6 = {}
    for arm in ("H-stock", "H-tuned"):
        for label, cond in (("card held gold file", True), ("card lacked gold file", False)):
            ks = [q for q in hkeys if bool(hrows[(q, arm)].get("card_gold_file")) == cond]
            b6[f"{arm}: {label}"] = {"issues": len(ks), "A_file_hit": mean(hrows[(q, "A")]["file_hit"] for q in ks),
                                     f"{arm}_file_hit": mean(hrows[(q, arm)]["file_hit"] for q in ks)}
    hcl = clusters_for(hkeys, hgold)
    b6["gold_symbol_rate"] = {a: cc.cluster_bootstrap({q: float(hrows[(q, a)]["symbol_hit"]) for q in hkeys}, hcl)
                              for a in ("A", "H-stock", "H-tuned")}
    out["pilot_by_card_gold"] = b6
    first_run_A = {q: rows[(q, "A")] for q in hkeys if (q, "A") in rows}
    out["pilot_A_vs_first_run_A"] = {
        "issues": len(first_run_A),
        "pilot_A_file_hit": cc.cluster_bootstrap({q: float(hrows[(q, "A")]["file_hit"]) for q in first_run_A}, hcl),
        "first_run_A_file_hit_same_issues": cc.cluster_bootstrap({q: float(r["file_hit"]) for q, r in first_run_A.items()}, hcl),
        "pilot_minus_first_run": cc.cluster_bootstrap({q: float(hrows[(q, "A")]["file_hit"]) - float(r["file_hit"])
                                                       for q, r in first_run_A.items()}, hcl),
        "agreement": mean(hrows[(q, "A")]["file_hit"] == r["file_hit"] for q, r in first_run_A.items()),
        "first_run_A_all_321": mean(rows[(q, "A")]["file_hit"] for q in qkeys)}
    path = os.path.join(args.out, "cc-bench", f"{args.name}.extra.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(path)


def hint_extra(args):
    """Cells for the agent section from the full hint run (reading only): disagreeing cards by gold
    symbol, every card-arm split difference with intervals, the agents' tool lists, and the first run's
    search-use shares."""
    rows, gold, leak = hint_rows(args.out, "hint-full")
    keys = sorted({q for q, _ in rows})
    cl_all = clusters_for(keys, gold)

    def sub_clusters(ks):
        ks = set(ks)
        return {r: [[q for q in c if q in ks] for c in cs if any(q in ks for q in c)] for r, cs in cl_all.items()
                if any(q in ks for c in cs for q in c)}
    out = {"label": "dollars = cost the client reports at API list prices"}
    # 1. disagreeing cards by gold symbol
    dis = {}
    for x, y in (("H-tuned", "H-stock"), ("H-fused", "H-tuned")):
        for a, b in ((x, y), (y, x)):
            ks = [q for q in keys if rows[(q, a)].get("card_gold_symbol") and not rows[(q, b)].get("card_gold_symbol")]
            cell = {"issues": len(ks)}
            for arm in (a, b, "A"):
                cell[arm] = {m: mean(rows[(q, arm)][m] for q in ks) for m in ("file_hit", "symbol_hit")}
            if ks:
                cell["diff"] = {f"{a}-{b}": paired(rows, gold, ks, a, b, ["file_hit", "symbol_hit"])}
            dis[f"{a} card has gold symbol, {b} card lacks it"] = cell
    out["disagreeing_cards_by_gold_symbol"] = dis
    # 2. card arm minus A by card gold file, with every interval
    c5 = {}
    for arm in ("H-stock", "H-tuned", "H-bm25", "H-fused"):
        for label, cond in (("card held gold file", True), ("card lacked gold file", False)):
            ks = [q for q in keys if bool(rows[(q, arm)].get("card_gold_file")) == cond]
            scl = sub_clusters(ks)
            c5[f"{arm}: {label}"] = {"issues": len(ks), **{
                m: {"arm": mean(rows[(q, arm)][m] or 0 for q in ks), "A": mean(rows[(q, "A")][m] or 0 for q in ks),
                    "diff": cc.cluster_bootstrap({q: float(rows[(q, arm)][m] or 0) - float(rows[(q, "A")][m] or 0)
                                                  for q in ks}, scl)}
                for m in ("file_hit", "symbol_hit", "api_calls", "tool_calls_total", "total_cost_alone_usd")}}
    out["card_arm_minus_A_by_card_gold_file"] = c5
    # 3a. the agents' tool lists, from every run's init event
    tools = collections.defaultdict(collections.Counter)
    for repo in REPOS:
        d = os.path.join(cc.bench_dir(args.out, common.load_config(repo), "hint-full"), "events")
        for fn in os.listdir(d):
            if fn.endswith(".events.jsonl"):
                arm = fn[:-len(".events.jsonl")].rsplit("_", 1)[1]
                ev = cc.parse_events(os.path.join(d, fn))
                tools[arm][json.dumps(sorted((ev["init"] or {}).get("tools") or []))] += 1
    out["tool_lists_hint_full"] = {a: dict(c) for a, c in sorted(tools.items())}
    # 4. first run: share of issues where the agent called search at least once
    frows, _ = load(args.out, "issues")
    fk = sorted({q for q, _ in frows})
    out["first_run_search_share"] = {arm: {"issues_with_search": sum(bool(frows[(q, arm)]["searches"]) for q in fk),
                                           "issues": len(fk),
                                           "share": round(sum(bool(frows[(q, arm)]["searches"]) for q in fk) / len(fk), 4)}
                                     for arm in ("B", "C", "D")}
    # 5. arm A against itself: the full hint run minus the first run, same 321 issues
    a_self = {"issues": 0}
    common_keys = [q for q in keys if (q, "A") in frows]
    if common_keys:
        diffs = {}
        for m in ("file_hit", "symbol_hit"):
            vals = {q: float(rows[(q, "A")][m]) - float(frows[(q, "A")][m]) for q in common_keys}
            up = sum(v > 0 for v in vals.values())
            down = sum(v < 0 for v in vals.values())
            diffs[m] = {"hint_run": mean(rows[(q, "A")][m] for q in common_keys),
                        "first_run": mean(frows[(q, "A")][m] for q in common_keys),
                        "diff_hint_minus_first": cc.cluster_bootstrap(vals, sub_clusters(common_keys)),
                        "disagree": round((up + down) / len(common_keys), 4),
                        "hit_only_in_hint_run": up, "hit_only_in_first_run": down}
        a_self = {"issues": len(common_keys), **diffs,
                  "first_run_batches": dict(collections.Counter(frows[(q, "A")]["batch"] for q in common_keys))}
    out["A_hint_run_minus_first_run"] = a_self
    # 6. card content, tuned LFM minus BM25
    out["card_tuned_minus_bm25"] = {m: cc.cluster_bootstrap(
        {q: float(rows[(q, "H-tuned")].get(m) or 0) - float(rows[(q, "H-bm25")].get(m) or 0) for q in keys}, cl_all)
        for m in ("card_gold_symbol", "card_gold_file")}
    # 7. query counts: written by Haiku, dropped by the hygiene filter, cut to 4, literal added, searched
    qc = collections.defaultdict(collections.Counter)
    for repo in REPOS:
        d = os.path.join(cc.bench_dir(args.out, common.load_config(repo), "hint-full"), "events")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith("_H-bm25.hint.jsonl"):  # queries are shared: one record per issue
                continue
            r = json.loads(open(os.path.join(d, fn)).read().splitlines()[-1])
            n = r["query_notes"]
            c = qc[repo]
            c["issues"] += 1
            c["written_by_haiku"] += n["model_queries"]
            c["dropped_invented_identifier"] += len(n["dropped_invented"])
            c["cut_by_cap_of_4"] += n["cut"]
            c["literal_query_added"] += int(n["literal_fallback"])
            c["searched"] += len(r["queries"])
    tot = collections.Counter()
    for c in qc.values():
        tot.update(c)
    out["query_counts"] = {"basis": "full hint run only (no pilot), one record per issue, identical in all four card arms",
                           **{r: dict(c) for r, c in qc.items()}, "total": dict(tot),
                           "per_issue_written": round(tot["written_by_haiku"] / tot["issues"], 3),
                           "per_issue_searched": round(tot["searched"] / tot["issues"], 3)}
    path = os.path.join(args.out, "cc-bench", "hint-full.extra.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(path)


def deny_placeholder(rule):
    """A deny rule with this machine's paths as placeholders: <home>, <harness checkout> (this checkout),
    <bench home> ($TUNA_BENCH_HOME), $TUNA_DATA and $TUNA_OUT. settings_sha256 is taken over the rules so written."""
    roots = [(common.default_out(), "$TUNA_OUT"), (cc.BENCH_HOME, "<bench home>"), (common.data_root(), "$TUNA_DATA"),
             (ROOT, "<harness checkout>"), (os.path.expanduser("~"), "<home>")]
    for root, name in sorted(roots, key=lambda x: -len(x[0])):
        root = root.rstrip("/")
        for pre in ("Read(/" + root + "/", "Read(" + root + "/"):
            if rule.startswith(pre):
                return "Read(" + name + "/" + rule[len(pre):]
    return rule


def setup_check(args):
    """Arm A's set-up in the first run (bench issues) and the hint-card run (bench hint-full), from
    each run's own records:
      claude_code_version, model   the run's init event (what Claude Code reported at start-up)
      prompt_sha256                the prompt as sent: the PROMPT template in scripts/cc-bench.py at
                                   the commit the run recorded, filled with the run's question
      prompt_template_sha256       that template alone (prompt path: scripts/cc-bench.py, PROMPT)
      settings_sha256              the --settings JSON file written for the run (deny list sorted, paths as
                                   placeholders: deny_placeholder)
      flags_hash                   the run's recorded hash of the command line, isolation env and plugin list
      command_flags                the claude flags from the command builder at the run's commit
      tools, allowed_tools         the init event's tool list; --allowedTools from the command
    Writes one record per run and a field-by-field comparison."""
    import hashlib
    import subprocess

    def sha(s_):
        return hashlib.sha256(s_.encode()).hexdigest()

    src_cache = {}

    def at(commit):
        if commit not in src_cache:
            src_cache[commit] = subprocess.run(["git", "-C", ROOT, "show", f"{commit}:scripts/cc-bench.py"],
                                               capture_output=True, text=True, check=True).stdout
        return src_cache[commit]

    def template(commit):
        return at(commit).split('PROMPT = """')[1].split('"""')[0]

    def command(commit):
        """The arm-A claude argv at that commit, with per-run paths and the prompt as placeholders."""
        code = at(commit)
        spec = importlib.util.spec_from_loader("cc_at", loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__dict__["__file__"] = os.path.join(ROOT, "scripts", "cc-bench.py")
        exec(compile(code, f"cc-bench.py@{commit}", "exec"), mod.__dict__)
        return mod.claude_command("A", "<settings>", None, "<prompt>", "<debug>", mod.MODEL), mod.ISOLATION_ENV
    runs = []
    for bench, label in (("issues", "first run"), ("hint-full", "hint-card run")):
        for repo in REPOS:
            cfg = common.load_config(repo)
            d = cc.bench_dir(args.out, cfg, bench)
            qs = {q["id"]: q for q in common.read_jsonl(os.path.join(d, "questions.jsonl"))}
            for r in common.read_jsonl(os.path.join(d, "runs.jsonl")):
                if r["arm"] != "A":
                    continue
                src = r.get("reused_from") or bench
                ed = os.path.join(cc.bench_dir(args.out, cfg, src), "events")
                ev = cc.parse_events(os.path.join(ed, f"{r['qid']}_A.events.jsonl"))
                init = ev["init"] or {}
                with open(os.path.join(ed, f"{r['qid']}_A.settings.json")) as fh:
                    st = json.load(fh)
                st["permissions"]["deny"] = sorted(deny_placeholder(x) for x in st["permissions"]["deny"])
                commit = r["git"]["sha"]
                cmd, env = command(commit)
                runs.append({
                    "run": label, "repo": cfg["name"], "qid": r["qid"], "bench_dir": src, "git_commit": commit,
                    "git_dirty": r["git"]["dirty"], "claude_code_version": init.get("claude_code_version"),
                    "model": init.get("model"), "models_billed": r.get("models"),
                    "prompt_path": "scripts/cc-bench.py (PROMPT)",
                    "prompt_template_sha256": sha(template(commit)),
                    "prompt_sha256": sha(template(commit).format(question=qs[r["qid"]]["question"])),
                    "settings_sha256": sha(json.dumps(st, sort_keys=True)), "settings": st,
                    "flags_hash": r["flags_hash"], "command_flags_sha256": sha(json.dumps([cmd, env])),
                    "command_flags": cmd, "isolation_env": env,
                    "tools": sorted(init.get("tools") or []),
                    "allowed_tools": cmd[cmd.index("--allowedTools") + 1].split(","),
                    "permission_mode": init.get("permissionMode"), "mcp_servers": init.get("mcp_servers"),
                    "plugins": init.get("plugins"), "skills": init.get("skills"),
                    "hook_events": ev["hooks"]})
    fields = ["claude_code_version", "model", "prompt_template_sha256", "settings_sha256", "flags_hash",
              "command_flags_sha256", "tools", "allowed_tools", "permission_mode", "mcp_servers", "plugins",
              "skills", "hook_events"]
    summary = {}
    for f in fields:
        vals = {label: sorted({json.dumps(r[f]) for r in runs if r["run"] == label})
                for label in ("first run", "hint-card run")}
        summary[f] = {"first run": [json.loads(v) for v in vals["first run"]],
                      "hint-card run": [json.loads(v) for v in vals["hint-card run"]],
                      "match": len(vals["first run"]) == 1 and vals["first run"] == vals["hint-card run"]}
    # the prompt as sent differs by question; it must match per issue across the two runs
    by_issue = collections.defaultdict(dict)
    for r in runs:
        by_issue[(r["repo"], r["qid"])][r["run"]] = r["prompt_sha256"]
    same_prompt = sum(1 for v in by_issue.values() if len(v) == 2 and v["first run"] == v["hint-card run"])
    summary["prompt_sha256 (per issue)"] = {"issues": len(by_issue), "identical_in_both_runs": same_prompt,
                                            "match": same_prompt == len(by_issue)}
    out = {"what": "arm A set-up, first run (bench issues) against hint-card run (bench hint-full), from the run records",
           "generated_by": "scripts/cc-analysis.py setup-check", "runs_per_experiment": {
               label: sum(r["run"] == label for r in runs) for label in ("first run", "hint-card run")},
           "all_fields_match": all(v["match"] for v in summary.values()),
           "summary": summary,
           "not_set_up_but_context": ["run time of day", "run order within an issue and which arms ran beside A",
                                      "harness commit (prompt and command code unchanged across them)"],
           "runs": runs}
    paths = [os.path.join(args.out, "cc-bench", "armA-setup.json")]
    for p_ in paths:
        os.makedirs(os.path.dirname(p_), exist_ok=True)
        with open(p_, "w") as fh:
            json.dump(out, fh, indent=1)
        print(p_)


SPLIT_RULE_LEDGER = "split fixed before it was computed, 2026-11-12 14:32 UTC"


def search_split(args):
    """The first run split by search use, as the reading rule logged in the ledger
    (cc-first-run-search-split-rule) defines it. For B, C and D: the arm's and A's gold-file rate
    and the paired difference on the issues where that arm called search at least once and where it
    did not. For D: how many of its search issues had a gold file among the symbols any of its
    search calls returned. Then the rule's case, applied mechanically. Observational: the split
    conditions on something the agent chose."""
    rows, gold = load(args.out, "issues")
    keys = sorted({q for q, _ in rows})
    cl_all = clusters_for(keys, gold)

    def sub(ks):
        ks = set(ks)
        return {r: [[q for q in c if q in ks] for c in cs if any(q in ks for q in c)] for r, cs in cl_all.items()
                if any(q in ks for c in cs for q in c)}
    out = {"rule": SPLIT_RULE_LEDGER, "bench": "issues (first run)", "metric": "gold-file rate",
           "interval": "95% paired bootstrap, 10000 replicates, gold-file clusters resampled within repository",
           "status": "observational: the split conditions on the agent's own choice to search", "arms": {}}
    for arm in ("B", "C", "D"):
        cells = {}
        for label, cond in (("search issues", True), ("no-search issues", False)):
            ks = [q for q in keys if bool(rows[(q, arm)]["searches"]) == cond]
            cells[label] = {"issues": len(ks), f"{arm}_rate": mean(rows[(q, arm)]["file_hit"] for q in ks),
                            "A_rate": mean(rows[(q, "A")]["file_hit"] for q in ks),
                            f"{arm}-A": cc.cluster_bootstrap({q: float(rows[(q, arm)]["file_hit"]) -
                                                              float(rows[(q, "A")]["file_hit"]) for q in ks}, sub(ks))}
        out["arms"][arm] = cells
    # D: gold file among the symbols returned to the agent by any of its search calls (any rank)
    fqn_file = {repo: {s_["fqn"]: s_["file_path"] for s_ in common.load_symbols(os.path.join(args.out, repo))}
                for repo in REPOS}
    found, n = 0, 0
    for q in keys:
        r = rows[(q, "D")]
        if not r["searches"]:
            continue
        n += 1
        p_ = os.path.join(r["events_dir"], f"{r['qid']}_D.search.jsonl")
        files = {fqn_file[r["repo"]].get(f) for s_ in common.read_jsonl(p_) for f in s_["ranked"]}
        found += bool(files & set(gold[q]))
    out["D_search_issues_with_gold_file_in_results"] = {"issues": n, "with_gold_file": found,
                                                        "basis": "union of every D search call's returned symbols"}
    d_ns = out["arms"]["D"]["no-search issues"]["D-A"][0]
    d_s = out["arms"]["D"]["search issues"]["D-A"][0]
    if d_ns >= 0.03 and d_s < 0.15:
        case = "the D-A gain is present on issues without a search call, so it is not credited to search"
    elif d_ns < 0.02 and d_s >= 0.15:
        case = "the D-A gain sits where search was called: an observational lead, with its confounds"
    else:
        case = "neither case of the rule: both cells are printed and nothing is read into them"
    out["rule_case"] = {"D-A no-search": d_ns, "D-A search": d_s, "case": case}
    paths = [os.path.join(args.out, "cc-bench", "first-run-search-split.json")]
    for p_ in paths:
        os.makedirs(os.path.dirname(p_), exist_ok=True)
        with open(p_, "w") as fh:
            json.dump(out, fh, indent=1)
        print(p_)


def instruction_lines():
    """The instruction lines of the task prompt (scripts/cc-bench.py PROMPT, all but the question),
    as the hook's literal fallback would cut them (first 25 words)."""
    tmpl = cc.PROMPT
    return [" ".join(l.split()[:25]) for l in tmpl.splitlines() if l.strip() and "{question}" not in l]


def instruction_line(args):
    """Ledger cc-hint-instruction-line-cell (2026-11-12T14:49:59.227Z; added after the results).
    An issue got an instruction-line query if one of its searched queries is an instruction line of
    the task prompt. Two splits are reported: the logged one (any searched query) and the narrower
    one (only queries added by the literal fallback)."""
    rows, gold, leak = hint_rows(args.out, "hint-full")
    keys = sorted({q for q, _ in rows})
    cl_all = clusters_for(keys, gold)
    instr = instruction_lines()
    hit = {}
    for repo in REPOS:
        d = os.path.join(cc.bench_dir(args.out, common.load_config(repo), "hint-full"), "events")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith("_H-bm25.hint.jsonl"):  # the queries are shared by all four card arms
                continue
            r = json.loads(open(os.path.join(d, fn)).read().splitlines()[-1])
            q = f"{repo}/{fn[:-len('_H-bm25.hint.jsonl')]}"
            bad = [x for x in r["queries"] if x in instr]
            if bad:
                fb = bool(r["query_notes"]["literal_fallback"]) and r["queries"][0] in bad
                hit[q] = {"queries": bad, "added_by_literal_fallback": fb,
                          "written_by_haiku": [x for x in bad if not (fb and x == r["queries"][0])]}

    def sub(ks):
        ks = set(ks)
        return {r: [[q for q in c if q in ks] for c in cs if any(q in ks for q in c)] for r, cs in cl_all.items()
                if any(q in ks for c in cs for q in c)}
    arms = ["A", "H-stock", "H-tuned", "H-bm25", "H-fused"]

    def cells(clean, flagged, label):
        pairs = [("H-tuned", "H-stock"), ("H-stock", "A"), ("H-tuned", "A"), ("H-bm25", "A"), ("H-fused", "A")]
        return {"definition": label, "issues_without": len(clean), "issues_with": len(flagged),
                "without": {f"{a}-{b}": cc.cluster_bootstrap({q: float(rows[(q, a)]["file_hit"]) -
                                                              float(rows[(q, b)]["file_hit"]) for q in clean}, sub(clean))
                            for a, b in pairs},
                "with": [{"issue": q, "repo": q.split("/")[0], "query": hit[q]["queries"],
                          "added_by_literal_fallback": hit[q]["added_by_literal_fallback"],
                          **{a: bool(rows[(q, a)]["file_hit"]) for a in arms}} for q in flagged]}
    logged = sorted(hit)
    fallback = sorted(q for q in hit if hit[q]["added_by_literal_fallback"])
    out = {"note": "post-hoc check, added after the results (2026-11-12)",
           "metric": "gold file", "interval": "95% paired bootstrap, 10000 replicates, gold-file clusters within repository",
           "instruction_lines": instr,
           "as_logged_any_searched_query": cells([q for q in keys if q not in hit], logged,
                                                 "any searched query is an instruction line (the logged definition)"),
           "literal_fallback_only": cells([q for q in keys if q not in fallback], fallback,
                                          "the instruction line was added by the literal fallback (narrower)"),
           "all_issues_for_reference": {f"{a}-{b}": cc.cluster_bootstrap(
               {q: float(rows[(q, a)]["file_hit"]) - float(rows[(q, b)]["file_hit"]) for q in keys}, cl_all)
               for a, b in [("H-tuned", "H-stock"), ("H-stock", "A"), ("H-tuned", "A"), ("H-bm25", "A"), ("H-fused", "A")]}}
    for p_ in (os.path.join(args.out, "cc-bench", "hint-instruction-line.json"),):
        os.makedirs(os.path.dirname(p_), exist_ok=True)
        with open(p_, "w") as fh:
            json.dump(out, fh, indent=1)
        print(p_)


# Keyed by the release's commit labels (dev-NN: commits of the unpublished development history, as in the
# released runs.jsonl).
DIRTY_NOTES = {
    "dev-14": {"what": "the pilot's runs (07:46-07:54 UTC) ran with one uncommitted edit: the harness stop guard "
                       "(--max-utilization), committed at 07:54:25 as dev-15 (scripts/cc-bench.py only, 8+/2-)",
               "agent_runs_ran_in_dirty_tree": True},
    "dev-17": {"what": "rows written by `cc-bench recover` after the agent runs had finished (08:14 UTC) under a "
                       "harness started from a clean dev-17; the uncommitted state at recovery held the "
                       "recover/reuse/arm-D harness code and the --fuse-bm25 change to src/tuna/mcp.py, all "
                       "committed at 08:19:53 as dev-18; those runs never used it",
               "agent_runs_ran_in_dirty_tree": False},
    "dev-19": {"what": "rows written by `cc-bench recover` after the agent runs had finished (08:34-08:35 UTC) under "
                       "a harness started from a clean dev-19; the uncommitted state at recovery held harness "
                       "failure-classification, run-order and report code (scripts/cc-bench.py, tests), committed "
                       "at 08:37:50 as dev-23",
               "agent_runs_ran_in_dirty_tree": False}}


def dirty_tree(args):
    """How many of the first experiment's final rows (A-D, 1,284) recorded a dirty git tree, by commit."""
    rows, _ = load(args.out, "issues")
    c = collections.Counter()
    for (q, arm), r in rows.items():
        if r["git"]["dirty"]:
            c[(r["git"]["sha"], arm, bool(r.get("recovered")), r.get("reused_from") or "issues")] += 1
    by_commit = collections.defaultdict(lambda: {"rows": 0, "by_arm": collections.Counter(), "recovered": 0,
                                                 "bench": set()})
    for (sha_, arm, rec, src), n in c.items():
        b = by_commit[sha_ if sha_.startswith("dev-") else sha_[:7]]
        b["rows"] += n
        b["by_arm"][arm] += n
        b["recovered"] += n if rec else 0
        b["bench"].add(src)
    out = {"what": "first experiment (bench issues) final rows that recorded a dirty git tree",
           "generated_by": "scripts/cc-analysis.py dirty-tree", "final_rows": len(rows),
           "dirty_rows": sum(c.values()),
           "by_commit": {k: {"rows": v["rows"], "by_arm": dict(v["by_arm"]), "recovered_rows": v["recovered"],
                             "rows_came_from_bench": sorted(v["bench"]), **DIRTY_NOTES.get(k, {})}
                         for k, v in sorted(by_commit.items())},
           "prompt_and_command": "the PROMPT template and the claude command builder are identical at every commit "
                                 "(see armA-setup.json); none of the uncommitted edits touched them"}
    for p_ in (os.path.join(args.out, "cc-bench", "first-run-dirty-tree.json"),):
        with open(p_, "w") as fh:
            json.dump(out, fh, indent=1)
        print(p_)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cache", "replay", "extra", "hint-extra", "setup-check", "search-split",
                                    "instruction-line", "dirty-tree"])
    ap.add_argument("--name", default="issues")
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"cache": cache, "replay": replay, "extra": extra, "hint-extra": hint_extra,
     "setup-check": setup_check, "search-split": search_split, "instruction-line": instruction_line,
     "dirty-tree": dirty_tree}[args.cmd](args)


if __name__ == "__main__":
    main()
