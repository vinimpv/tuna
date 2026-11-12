"""Write the reject shares into every question set's manifest (backfill for sets written before tuna
questions recorded them), and list them with the writer engine.

  .venv/bin/python scripts/reject-shares.py [--write] [--repos ...]

Shares are over writer attempts (from <set>.progress.jsonl): reject_share (any reason),
filter_reject_share (the question filters) and writer_failure_share (no usable reply); see
tuna.questions.reject_shares. Unparseable replies are split further (from the pi event logs):
empty_at_cap_share, empty_other_share and prose_before_json_share. A ticked spot check (scripts/spot-check.py) is shown when present.
Needs the writers' progress files (<set>.progress.jsonl) and event logs, which are not in this release; the shares it
wrote are in each set's manifest.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

from tuna.questions import reject_shares  # noqa: E402


def stop_reason(log):
    """stopReason of the writer's last assistant message in a pi event log (None if unknown)."""
    last = None
    try:
        for l in open(log):
            if '"message_end"' in l:
                e = json.loads(l)
                if e.get("type") == "message_end" and e.get("message", {}).get("role") == "assistant":
                    last = e["message"].get("stopReason")
    except (OSError, ValueError):
        return None
    return last


def unparseable_split(attempts):
    """The unparseable replies split by cause, as shares of all attempts: empty_at_cap (no answer text, the
    writer's last turn stopped at the output-token cap: stopReason "length"), empty_other (no answer text,
    any other stop), prose_before_json (answer text that is not the bare JSON object the rules require)."""
    n = len(attempts)
    un = [a for a in attempts if a["reason"] == "unparseable"]
    empty = [a for a in un if not (a.get("reply") or "").strip()]
    at_cap = sum(1 for a in empty if stop_reason(a.get("log", "")) == "length")
    return {"empty_at_cap_share": round(at_cap / n, 4) if n else None,
            "empty_other_share": round((len(empty) - at_cap) / n, 4) if n else None,
            "prose_before_json_share": round((len(un) - len(empty)) / n, 4) if n else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["saleor", "zulip", "posthog", "dispatch", "homeassistant"])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    print(f"{'repo':14s} {'set':28s} {'attempts':>8s} {'reject':>7s} {'filter':>7s} {'writer':>7s} {'cap':>6s} {'prose':>6s} {'spot':>6s}  engine")
    for repo in args.repos:
        for prog in sorted(glob.glob(os.path.join(args.out, repo, "questions", "*.progress.jsonl"))):
            name = os.path.basename(prog)[:-len(".progress.jsonl")]
            rows = [json.loads(l) for l in open(prog)]
            attempts = [a for r in rows[1:] for a in r.get("attempts", [])]
            if not attempts:
                continue
            sh = reject_shares(attempts)
            sh.update(unparseable_split(attempts))
            mp = os.path.join(args.out, repo, "questions", f"{name}.manifest.json")
            m = json.load(open(mp)) if os.path.exists(mp) else None
            if m is not None and args.write and any(m.get(k) != v for k, v in sh.items()):
                m.update(sh)
                with open(mp, "w") as fh:
                    json.dump(m, fh, indent=1)
            spot = (m or {}).get("spot_check", {}).get("reject_share")
            eng = (m or {}).get("writer_engine") or rows[0].get("settings", {}).get("writer_engine") or ("claude-code" if (m or {}).get("writer_backend") == "claude-code"
                                                     else "local writer")
            print(f"{repo:14s} {name:28s} {len(attempts):8d} {sh['reject_share']:7.3f} {sh['filter_reject_share']:7.3f} "
                  f"{sh['writer_failure_share']:7.3f} {sh['empty_at_cap_share']:6.3f} {sh['prose_before_json_share']:6.3f} {'' if spot is None else f'{spot:.3f}':>6s}  {eng}")


if __name__ == "__main__":
    main()
