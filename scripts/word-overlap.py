"""Word overlap between test questions and their gold symbols, per repo and test set.

  .venv/bin/python scripts/word-overlap.py [--repos ...] [--sets ...] [--json OUT]

Words: split on non-alphanumerics and identifier case (fooBar, FOOBar, foo_bar -> foo, bar), lowercased,
English stopwords and single characters dropped. Per question: the share of its distinct words that
also occur in the gold symbol's body (as embedded, with its FQN and file header; for several golds, the union). Reports the mean and median share per set.
"""
import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just let me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves get gets got one ones s t""".split())
_CASE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def words(text):
    out = set()
    for part in re.split(r"[^A-Za-z0-9]+", text):
        for w in _CASE.findall(part):
            w = w.lower()
            if len(w) > 1 and w not in STOP:
                out.add(w)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["saleor", "zulip", "posthog"])
    ap.add_argument("--sets", nargs="+", default=["qwen-m2", "claude-m2", "claude-m3", "qwen-m4-400", "issues-v1",
                                                    "issues-v1-title"])
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = []
    for repo in args.repos:
        w = os.path.join(args.out, repo)
        body = {}
        for l in open(os.path.join(w, "symbols.jsonl")):
            s = json.loads(l)
            body[s["fqn"]] = s["body"]
        cache = {}
        for st in args.sets:
            p = os.path.join(w, "questions", f"{st}.jsonl")
            if not os.path.exists(p):
                continue
            shares = []
            for l in open(p):
                q = json.loads(l)
                qw = words(q["question"])
                if not qw:
                    continue
                gw = set()
                for g in q.get("gold_fqns") or [q["gold_fqn"]]:
                    if g not in cache:
                        cache[g] = words(body.get(g, ""))
                    gw |= cache[g]
                shares.append(len(qw & gw) / len(qw))
            rows.append({"repo": repo, "set": st, "n": len(shares), "mean": round(statistics.mean(shares), 4),
                         "median": round(statistics.median(shares), 4)})
    print(f"{'repo':8s} {'set':16s} {'n':>4s} {'mean':>6s} {'median':>6s}")
    for r in rows:
        print(f"{r['repo']:8s} {r['set']:16s} {r['n']:4d} {r['mean']:6.3f} {r['median']:6.3f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
