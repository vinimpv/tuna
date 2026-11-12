"""tuna hint: query and card hygiene, fail-open, recursion guard, the OpenAI-compatible backend,
and an end-to-end hook process (model = a local fake server, search = the real Saleor index).

Run: .venv/bin/python -m unittest tests.test_hint
"""
import http.server
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest

from tuna import common, hint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "Find the code.\n\nBug: Giftcard created event is not triggered when gift card is bought via draft_order_complete"


def hit(fqn, rank, path="pkg/a.py", a=1, b=9, score=0.5):
    return {"fqn": fqn, "file": path, "line_start": a, "line_end": b, "rank": rank, "score": score,
            "snippet": f"def {fqn.rsplit('.', 1)[-1]}():\n    pass"}


class Scripted:
    """A backend that replies with canned texts, in order."""
    name = "scripted"

    def __init__(self, *replies, delay=0.0, error=None):
        self.replies, self.delay, self.error, self.prompts = list(replies), delay, error, []

    def complete(self, system, user, timeout):
        self.prompts.append((system, user))
        if self.error:
            raise self.error
        time.sleep(self.delay)
        return self.replies.pop(0), hint._usage(0.001, "scripted", 10, 0, 0, 5, self.delay)


class QueryTest(unittest.TestCase):
    def test_invented_identifiers_dropped_and_capped(self):
        raw = ["Giftcard created event is not triggered", "giftCardCreate webhook", "draft_order_complete gift card",
               "gift card event on order", "webhook when gift card bought", "another one", "and more"]
        qs, notes = hint.clean_queries(raw, PROMPT)
        self.assertNotIn("giftCardCreate webhook", qs)  # camelCase not in the prompt
        self.assertIn("draft_order_complete gift card", qs)  # snake_case that is in the prompt
        self.assertEqual(len(qs), hint.MAX_QUERIES)
        self.assertEqual(notes["dropped_invented"][0]["identifiers"], ["giftCardCreate"])
        self.assertEqual(notes["cut"], 2)

    def test_literal_fallback(self):
        qs, notes = hint.clean_queries(["gift card webhook missing"], PROMPT)
        self.assertTrue(notes["literal_fallback"])
        self.assertTrue(qs[0].startswith("Bug: Giftcard created event"))
        self.assertIn("gift card webhook missing", qs)

    def test_zero_queries_stay_zero(self):
        self.assertEqual(hint.clean_queries([], PROMPT)[0], [])

    def test_dotted_identifier_is_an_identifier(self):
        self.assertIn("saleor.giftcard.utils", hint.identifiers("look at saleor.giftcard.utils please"))


class CardTest(unittest.TestCase):
    HITS = [hit("pkg.a.f", 1, "pkg/a.py", 10, 20), hit("pkg.b.g", 2, "pkg/b.py", 3, 4), hit("pkg.c.h", 3)]

    def test_only_hits_and_paths_from_the_index(self):
        card, leads, notes = hint.build_card([{"symbol": "pkg.a.f", "why": "creates the thing"},
                                              {"symbol": "pkg.zz.invented", "why": "x"},
                                              {"symbol": "`pkg.b.g`", "why": "sends it"}], self.HITS)
        self.assertEqual([l["fqn"] for l in leads], ["pkg.a.f", "pkg.b.g"])
        self.assertEqual(notes["dropped_unknown"], ["pkg.zz.invented"])
        self.assertTrue(card.startswith(hint.WRAPPER + "\n- pkg/a.py:10-20  pkg.a.f — creates the thing"))

    def test_absence_conclusions_dropped(self):
        for why in ("this event is never recorded", "nothing about refunds here", "the webhook does not exist",
                    "missing the call to send"):
            card, leads, notes = hint.build_card([{"symbol": "pkg.a.f", "why": why}], self.HITS)
            self.assertEqual(leads[0]["why"], "", why)
            self.assertEqual(notes["why_dropped"], 1)
            self.assertNotIn(why, card)

    def test_cap(self):
        hits = [hit(f"pkg.m{i}.f", i) for i in range(1, 10)]
        cands = [{"symbol": h["fqn"], "why": "word " * 40} for h in hits]
        card, leads, notes = hint.build_card(cands, hits)
        self.assertLessEqual(len(card), hint.CARD_MAX_CHARS)
        self.assertLessEqual(len(leads), hint.MAX_CANDIDATES)
        self.assertTrue(all(len(l["why"].split()) <= hint.WHY_MAX_WORDS for l in leads))

    def test_no_leads_no_card(self):
        self.assertIsNone(hint.build_card([], self.HITS)[0])


class HintStepTest(unittest.TestCase):
    def search(self, q):
        return [hit("pkg.a.f", 1), hit("pkg.b.g", 2)]

    def test_card(self):
        b = Scripted('{"queries": ["Giftcard created event is not triggered"]}',
                     '```json\n{"candidates": [{"symbol": "pkg.b.g", "why": "fires the event"}]}\n```')
        rec = hint.make_card(PROMPT, b, self.search)
        self.assertEqual(rec["status"], "card")
        self.assertIn("pkg.b.g", rec["card"])
        self.assertEqual(len(rec["calls"]), 2)
        self.assertIn("pkg.a.f", b.prompts[1][1])  # the card model saw the hits

    def test_passthrough(self):
        rec = hint.make_card(PROMPT, Scripted('{"queries": []}'), self.search)
        self.assertEqual((rec["status"], rec["card"]), ("passthrough", None))

    def run_hook(self, backend, timeout=hint.TIMEOUT_S):
        log = os.path.join(tempfile.mkdtemp(), "log.jsonl")
        out = hint.run_hook(json.dumps({"prompt": PROMPT}), backend, lambda: self.search, log, timeout)
        return out, json.loads(open(log).read().splitlines()[-1])

    def test_hook_injects_card(self):
        out, rec = self.run_hook(Scripted('{"queries": ["gift card"]}', '{"candidates": [{"symbol": "pkg.a.f"}]}'))
        ctx = json.loads(out)["hookSpecificOutput"]
        self.assertEqual(ctx["hookEventName"], "UserPromptSubmit")
        self.assertTrue(ctx["additionalContext"].startswith(hint.WRAPPER))
        self.assertEqual(rec["injected"], ctx["additionalContext"])

    def test_fail_open_on_error(self):
        out, rec = self.run_hook(Scripted(error=RuntimeError("api down")))
        self.assertIn("Hint search failed", json.loads(out)["hookSpecificOutput"]["additionalContext"])
        self.assertEqual(rec["status"], "failed")
        self.assertIn("api down", rec["error"])

    def test_fail_open_on_timeout(self):
        t0 = time.monotonic()
        out, rec = self.run_hook(Scripted('{"queries": []}', delay=5), timeout=1)
        self.assertLess(time.monotonic() - t0, 3)
        self.assertIn("timeout", json.loads(out)["hookSpecificOutput"]["additionalContext"])
        self.assertEqual(rec["status"], "failed")

    def test_passthrough_prints_nothing(self):
        out, rec = self.run_hook(Scripted('{"queries": []}'))
        self.assertEqual(out, "")
        self.assertEqual(rec["status"], "passthrough")

    def test_recursion_guard(self):
        b = Scripted('{"queries": ["x"]}')
        os.environ[hint.ENV_GUARD] = "1"
        try:
            self.assertEqual(hint.run_hook(json.dumps({"prompt": PROMPT}), b, lambda: self.search), "")
        finally:
            del os.environ[hint.ENV_GUARD]
        self.assertEqual(b.prompts, [])  # no model call at all

    def test_cli_backend_isolation_flags(self):
        cmd = []

        class Fake(hint.ClaudeCLI):
            def complete(self, system, user, timeout):
                return super().complete(system, user, timeout)
        orig = subprocess.run

        def fake_run(c, **kw):
            cmd.extend(c)
            cmd.append(kw["env"])
            return subprocess.CompletedProcess(c, 0, json.dumps({"subtype": "success", "result": "{}",
                                                                 "total_cost_usd": 0.0, "usage": {}}), "")
        subprocess.run = fake_run
        try:
            Fake("claude-haiku-5-5", tempfile.mkdtemp()).complete("S", "U", 5)
        finally:
            subprocess.run = orig
        env = cmd[-1]
        self.assertEqual(env[hint.ENV_GUARD], "1")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"], "1")
        for flag in ("--strict-mcp-config", "--restricted", "--no-session-persistence", "--disable-slash-commands"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(cmd[cmd.index("--setting-sources") + 1], "")
        self.assertTrue(json.loads(cmd[cmd.index("--settings") + 1])["disableAllHooks"])


class SharedQueriesTest(unittest.TestCase):
    def test_two_arms_one_query_call(self):
        """H-stock and H-tuned on one issue, concurrently: one query call, identical queries, own hits."""
        cache = tempfile.mkdtemp()
        backend = Scripted('{"queries": ["Giftcard created event is not triggered"]}',
                           '{"candidates": [{"symbol": "pkg.a.f"}]}', '{"candidates": [{"symbol": "pkg.s.g"}]}',
                           delay=0.3)
        lock = threading.Lock()

        class Safe:  # serialise the scripted replies; the hint step's own lock is what is under test
            name, model = "scripted", "s"

            def complete(self, system, user, timeout):
                with lock:
                    return backend.complete(system, user, timeout)
        recs = {}

        def arm(name, fqn):
            recs[name] = hint.make_card(PROMPT, Safe(), lambda q: [hit(fqn, 1)], query_cache=cache)
        ts = [threading.Thread(target=arm, args=a) for a in (("stock", "pkg.s.g"), ("tuned", "pkg.a.f"))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        query_calls = [p for p in backend.prompts if p[0] == hint.QUERY_SYSTEM]
        self.assertEqual(len(query_calls), 1)
        self.assertEqual(recs["stock"]["queries"], recs["tuned"]["queries"])
        self.assertEqual(sorted(r["query_source"] for r in recs.values()), ["cache", "model"])
        for r in recs.values():  # each arm is charged for the query call
            self.assertEqual(r["calls"][0]["cost_usd"], 0.001)

    def test_violations_logged_and_not_injected(self):
        b = Scripted('{"queries": ["gift card"]}',
                     '{"candidates": [{"symbol": "saleor.giftcard.utils.gift_cards_create", "why": "from memory"},'
                     ' {"symbol": "pkg.a.f", "why": "a hit"}]}')
        rec = hint.make_card(PROMPT, b, lambda q: [hit("pkg.a.f", 1)])
        self.assertEqual(rec["violations"], [{"symbol": "saleor.giftcard.utils.gift_cards_create",
                                              "reason": "not in this run's hits"}])
        self.assertNotIn("gift_cards_create", rec["card"])
        self.assertEqual(hint.check_card(rec["card"], rec["leads"], [hit("pkg.a.f", 1)]), [])

    def test_check_card_catches_a_foreign_line(self):
        card = hint.WRAPPER + "\n- pkg/zz.py:1-2  pkg.zz.nope — x"
        self.assertEqual(len(hint.check_card(card, [], [hit("pkg.a.f", 1)])), 1)


class FakeOpenAI(http.server.BaseHTTPRequestHandler):
    """Answers chat completions: queries for the query prompt, the first hit for the card prompt."""
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.seen.append(body)
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if system == hint.QUERY_SYSTEM:
            text = json.dumps({"queries": ["Giftcard created event is not triggered when gift card is bought",
                                           "gift card creation from order lines"]})
        else:
            first = user.split("Search hits:\n1. ", 1)[1].split()[0]
            text = json.dumps({"candidates": [{"symbol": first, "why": "creates gift cards for order lines"}]})
        out = json.dumps({"model": body["model"], "choices": [{"message": {"content": text}}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class OpenAIBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_backend(self):
        text, usage = hint.OpenAICompat(self.url, "lfm3-1.7b").complete(hint.QUERY_SYSTEM, PROMPT, 10)
        self.assertIn("queries", json.loads(text))
        self.assertEqual((usage["input_tokens"], usage["output_tokens"], usage["cost_usd"]), (100, 20, None))
        self.assertEqual(FakeOpenAI.seen[-1]["model"], "lfm3-1.7b")

    @unittest.skipUnless(os.path.exists(os.path.join(common.default_out(), "saleor", "vectors", "lfm25-emb.npy")),
                         "Saleor lfm25-emb index not on this machine")
    def test_hook_process_end_to_end(self):
        log = os.path.join(tempfile.mkdtemp(), "hint.jsonl")
        p = subprocess.run([os.path.join(ROOT, "scripts", "tuna-hint-hook"), "--repo", "saleor", "--tag", "lfm25-emb",
                            "--backend", "openai", "--url", self.url, "--model", "fake", "--log", log],
                           input=json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": PROMPT}),
                           capture_output=True, text=True, timeout=300, cwd=tempfile.mkdtemp())
        self.assertEqual(p.returncode, 0, p.stderr[-1500:])
        ctx = json.loads(p.stdout)["hookSpecificOutput"]["additionalContext"]  # stdout is only the hook JSON
        rec = json.loads(open(log).read())
        self.assertEqual(rec["status"], "card", rec)
        self.assertEqual(ctx, rec["card"])
        lead = rec["leads"][0]
        self.assertIn(f"{lead['file']}:{lead['line_start']}-{lead['line_end']}  {lead['fqn']}", ctx)

    def test_hook_process_guard(self):
        env = dict(os.environ, TUNA_HINT_INSIDE="1")
        p = subprocess.run([os.path.join(ROOT, "scripts", "tuna-hint-hook"), "--repo", "saleor", "--tag", "x"],
                           input="{}", capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual((p.returncode, p.stdout), (0, ""))


if __name__ == "__main__":
    unittest.main()
