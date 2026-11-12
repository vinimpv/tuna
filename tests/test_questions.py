"""Question validation and file sampling.  Run: .venv/bin/python -m unittest discover -s tests"""
import collections
import unittest

from tuna import questions as Q

FILE = "saleor/checkout/tasks.py"
SYMS = {
    "saleor.checkout.tasks.delete_expired_checkouts": {"kind": "Function"},
    "saleor.checkout.tasks.CheckoutCleaner": {"kind": "Class"},
    "saleor.checkout.tasks.CheckoutCleaner.run": {"kind": "Method"},
    "saleor.checkout.tasks.BATCH_SIZE": {"kind": "Constant"},
}
GOOD_Q = "How long can an abandoned cart stay around before a scheduled job removes it?"


def check(reply, used=None):
    draft, err = Q.parse_draft(reply)
    bad = err or Q.validate(draft, FILE, SYMS, used or {})
    return bad[0] if bad else None


class Parse(unittest.TestCase):
    def test_plain_fenced_and_thinking(self):
        obj = '{"question": "%s", "gold_fqn": "saleor.checkout.tasks.delete_expired_checkouts"}' % GOOD_Q
        for reply in (obj, f"```json\n{obj}\n```", f"<think>pick one {{not json}}</think>\n{obj}", f"\n{obj}\n"):
            self.assertIsNone(check(reply), reply)

    def test_unparseable(self):
        self.assertEqual(check("Sure! Here is a question."), "unparseable")
        self.assertEqual(check(""), "unparseable")
        self.assertEqual(check('{"question": "x"}'), "unparseable")

    def test_multiple_golds(self):
        two = '{"question": "%s", "gold_fqn": "a"}\n{"question": "%s", "gold_fqn": "b"}' % (GOOD_Q, GOOD_Q)
        self.assertEqual(check(two), "multiple_golds")
        self.assertEqual(check('{"question": "%s", "gold_fqn": ["a", "b"]}' % GOOD_Q), "multiple_golds")
        self.assertEqual(check('{"question": "%s", "gold_fqn": "a, b"}' % GOOD_Q), "multiple_golds")
        # a one-element list is one gold
        one = '{"question": "%s", "gold_fqns": ["saleor.checkout.tasks.delete_expired_checkouts"]}' % GOOD_Q
        self.assertIsNone(check(one))


class Validate(unittest.TestCase):
    def reply(self, q, gold):
        return '{"question": "%s", "gold_fqn": "%s"}' % (q, gold)

    def test_gold_must_be_in_file_and_not_a_constant(self):
        self.assertEqual(check(self.reply(GOOD_Q, "saleor.order.tasks.other")), "gold_not_in_file")
        self.assertEqual(check(self.reply(GOOD_Q, "saleor.checkout.tasks.BATCH_SIZE")), "gold_is_constant")
        self.assertIsNone(check(self.reply(GOOD_Q, "saleor.checkout.tasks.CheckoutCleaner.run")))

    def test_identifier_leak(self):
        g = "saleor.checkout.tasks.delete_expired_checkouts"
        self.assertEqual(check(self.reply("Where do we delete expired checkouts every night?", g)), "leaks_identifier")
        self.assertEqual(check(self.reply("Which job calls delete_expired_checkouts on a schedule?", g)),
                         "leaks_identifier")
        self.assertEqual(check(self.reply("What does DeleteExpiredCheckouts do for old carts?", g)), "leaks_identifier")
        # the same words, not together, are fine
        self.assertIsNone(check(self.reply("Which expired checkouts does the nightly job delete first?", g)))
        c = "saleor.checkout.tasks.CheckoutCleaner"
        self.assertEqual(check(self.reply("What is the checkout cleaner responsible for in this shop?", c)),
                         "leaks_identifier")

    def test_path_leak_short_and_reused(self):
        g = "saleor.checkout.tasks.delete_expired_checkouts"
        self.assertEqual(check(self.reply("What happens in Saleor/Checkout/Tasks.py when carts get old?", g)),
                         "leaks_path")
        self.assertEqual(check(self.reply("How are old carts removed?", g)), "too_short")
        self.assertEqual(check(self.reply(GOOD_Q, g), used={g: "saleor-m1-001"}), "gold_already_used")

    def test_words(self):
        self.assertEqual(Q.words("HTTPServer get_user_ID v2"), ["http", "server", "get", "user", "id", "v", "2"])


class Sampling(unittest.TestCase):
    syms = ([{"file_path": "big.py", "kind": "Function"}] * 50 + [{"file_path": "small.py", "kind": "Class"}]
            + [{"file_path": "consts.py", "kind": "Constant"}] * 5
            + [{"file_path": f"f{i}.py", "kind": "Method"} for i in range(20)])

    def test_deterministic_complete_and_eligible_only(self):
        a, b = Q.file_order(self.syms, 7), Q.file_order(self.syms, 7)
        self.assertEqual(a, b)
        self.assertEqual(sorted(a), sorted({"big.py", "small.py"} | {f"f{i}.py" for i in range(20)}))
        self.assertNotEqual(a, Q.file_order(self.syms, 8))

    def test_weighted_by_symbol_count(self):
        first = collections.Counter(Q.file_order(self.syms, s)[0] for s in range(2000))
        # big.py holds 50 of 71 eligible symbols: it should come first about 70% of the time
        self.assertGreater(first["big.py"] / 2000, 0.6)
        self.assertLess(first["small.py"] / 2000, 0.05)




class ClaudeEvents(unittest.TestCase):
    def test_reads_calls_turns_usage(self):
        import json, os, tempfile
        root = tempfile.mkdtemp()
        a = lambda mid, blocks, **kw: {"type": "assistant", "message": {"id": mid, "model": "m1", "content": blocks}, **kw}
        events = [
            {"type": "system", "subtype": "init", "model": "m1"},
            a("x", [{"type": "tool_use", "name": "Grep", "input": {"pattern": "def", "path": f"{root}/{FILE}"}}]),
            a("y", [{"type": "text", "text": "looking"}]),
            a("y", [{"type": "tool_use", "name": "Read", "input": {"file_path": f"{root}/{FILE}"}}]),
            a("z", [{"type": "tool_use", "name": "Read", "input": {"file_path": "/elsewhere/x.py"}}],
              parent_tool_use_id="t1"),  # a subagent's turn is not the writer's
            {"type": "result", "subtype": "success", "result": "{}", "total_cost_usd": 0.5,
             "usage": {"input_tokens": 8, "output_tokens": 50, "service_tier": "standard"}},
        ]
        os.makedirs(os.path.dirname(os.path.join(root, FILE)), exist_ok=True)
        open(os.path.join(root, FILE), "w").close()   # files_read keeps only files that exist in the checkout
        events.insert(-1, a("w", [{"type": "tool_use", "name": "Read", "input": {"file_path": f"{root}/no/such.py"}}]))
        log = os.path.join(root, "log.jsonl")
        with open(log, "w") as fh:
            fh.write("\n".join(json.dumps(e, separators=(",", ":")) for e in events) + "\n")
        ev = Q.read_events(log, root, "claude-code")
        self.assertEqual(ev["files_read"], [FILE])
        self.assertEqual([c["tool"] for c in ev["tool_calls"]], ["Grep", "Read", "Read"])
        self.assertEqual((ev["turns"], ev["models"]), (3, ["m1"]))
        self.assertEqual(ev["usage"], {"input_tokens": 8, "output_tokens": 50, "cost_usd": 0.5})
        with open(log, "rb") as fh:  # the runaway guard counts tool calls by this marker
            self.assertEqual(fh.read().count(Q.WRITERS["claude-code"]["tool_marker"]), 4)


class FixedAndTakenGolds(unittest.TestCase):
    """process_file with --gold-list (fixed golds) and --used-golds-from (pre-taken golds), fake writer."""

    def run_file(self, replies, gold_slots=None, pre_taken=None, used=None, quota=1):
        import os, stat, tempfile, threading, types
        root = tempfile.mkdtemp()
        os.makedirs(os.path.join(root, os.path.dirname(FILE)))
        with open(os.path.join(root, FILE), "w") as fh:
            fh.write("def delete_expired_checkouts():\n    pass\n")
        # The fake writer answers with the next reply and records each prompt.
        rfile, plog = os.path.join(root, "replies"), os.path.join(root, "prompts")
        with open(rfile, "w") as fh:
            fh.write("\n".join(replies) + "\n")
        ask = os.path.join(root, "ask")
        with open(ask, "w") as fh:
            fh.write(f'#!/bin/bash\nshift 3\nprintf "%s\\n===\\n" "$1" >> {plog}\n'
                     f'head -1 {rfile}; tail -n +2 {rfile} > {rfile}.t && mv {rfile}.t {rfile}; : > "${{2:-/dev/null}}"\n')
        os.chmod(ask, os.stat(ask).st_mode | stat.S_IEXEC)
        args = types.SimpleNamespace(writer="pi", max_attempts=2, timeout=30, max_tool_calls=40, prompt="p", seed=1)
        ctx = {"args": args, "cfg": {"path": root}, "paths": {"logs": root}, "order": [FILE], "heldout": set(),
               "by_file": {FILE: {g: dict(v, line_start=1, line_end=2) for g, v in SYMS.items()}},
               "template": "TEMPLATE {{FILE}} {{GOLD}}", "prompt_sha": "x", "filter": "v1", "lock": threading.Lock(),
               "halt": None, "ask": ask, "quota": {FILE: quota}, "addendum": "TAKEN:\n{{TAKEN}}",
               "addendum_name": "a", "addendum_sha": "y", "gold_slots": gold_slots, "pre_taken": pre_taken or {},
               "used": dict(used or {})}
        rec = Q.process_file(ctx, 0)
        with open(plog) as fh:
            return rec, fh.read().split("===\n")[:-1]

    def test_fixed_golds(self):
        g1, g2 = "saleor.checkout.tasks.CheckoutCleaner", "saleor.checkout.tasks.CheckoutCleaner.run"
        r = lambda g: '{"question": "%s", "gold_fqn": "%s"}' % (GOOD_Q, g)
        # slot 1: wrong gold, then right; slot 2: wrong twice (lost), the file goes on to no further slots
        rec, prompts = self.run_file([r(g2), r(g1), r(g1), r(g1)], gold_slots={FILE: [g1, g2]})
        self.assertEqual([q["gold_fqn"] for q in rec["questions"]], [g1])
        self.assertEqual([a["reason"] for a in rec["attempts"]],
                         ["gold_mismatch", None, "gold_mismatch", "gold_mismatch"])
        self.assertIn(g1, prompts[0])
        self.assertIn(g2, prompts[2])
        self.assertTrue(rec["questions"][0]["gold_given"])
        # a lost slot does not end the file: the next given gold is still asked
        rec, _ = self.run_file([r(g2), r(g2), r(g2)], gold_slots={FILE: [g1, g2]})
        self.assertEqual([q["gold_fqn"] for q in rec["questions"]], [g2])

    def test_pre_taken_golds(self):
        g = "saleor.checkout.tasks.delete_expired_checkouts"
        other = "saleor.checkout.tasks.CheckoutCleaner"
        r = lambda g: '{"question": "%s", "gold_fqn": "%s"}' % (GOOD_Q, g)
        rec, prompts = self.run_file([r(g), r(other)], pre_taken={FILE: [g]}, used={g: "set claude-m2"})
        self.assertEqual([a["reason"] for a in rec["attempts"]], ["gold_already_used", None])
        self.assertIn("set claude-m2", rec["attempts"][0]["detail"])
        self.assertIn("TAKEN:\n- " + g, prompts[0])


if __name__ == "__main__":
    unittest.main()
