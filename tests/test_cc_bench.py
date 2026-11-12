"""cc-bench grading and isolation checks (no Claude calls).  Run: .venv/bin/python -m unittest tests.test_cc_bench"""
import importlib.util
import json
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("cc_bench", os.path.join(ROOT, "scripts", "cc-bench.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

Q = {"gold_fqns": ["saleor.order.utils.update_order_charge_status"], "gold_files": ["saleor/order/utils.py"]}
ROOT_PREFIX = "/home/x/checkout/"


class GradeTest(unittest.TestCase):
    def g(self, answer, q=Q):
        return cc.grade(answer, q, ROOT_PREFIX)

    def test_exact(self):
        r = self.g("It is here.\nFILE: saleor/order/utils.py\nSYMBOL: saleor.order.utils.update_order_charge_status")
        self.assertTrue(r["file_hit"] and r["symbol_hit"])

    def test_markdown_and_variants(self):
        for f, s in [("`saleor/order/utils.py`", "`saleor.order.utils.update_order_charge_status()`"),
                     ("./saleor/order/utils.py:954-979", "saleor/order/utils.py::update_order_charge_status"),
                     (ROOT_PREFIX + "saleor/order/utils.py", "saleor.order.utils.update_order_charge_status")]:
            r = self.g(f"**FILE:** {f}\n**SYMBOL:** {s}")
            self.assertTrue(r["file_hit"], f)
            self.assertTrue(r["symbol_hit"], s)

    def test_last_lines_win(self):
        r = self.g("FILE: saleor/order/utils.py\nSYMBOL: saleor.order.utils.update_order_charge_status\n"
                   "Actually no.\nFILE: saleor/order/actions.py\nSYMBOL: saleor.order.actions.x")
        self.assertFalse(r["file_hit"] or r["symbol_hit"])
        self.assertTrue(r["file_mentioned"] and r["symbol_mentioned"])

    def test_wrong_and_missing(self):
        r = self.g("I could not find it.")
        self.assertFalse(r["file_hit"] or r["symbol_hit"] or r["symbol_tail_hit"])
        r = self.g("FILE: saleor/order/utils.py\nSYMBOL: update_order_charge_status")
        self.assertTrue(r["file_hit"])
        self.assertFalse(r["symbol_hit"])
        self.assertTrue(r["symbol_tail_hit"])  # bare name, right file

    def test_init_module(self):
        q = {"gold_fqns": ["saleor.order.__init__.OrderStatus"], "gold_files": ["saleor/order/__init__.py"]}
        r = self.g("FILE: saleor/order/__init__.py\nSYMBOL: saleor.order.OrderStatus", q)
        self.assertTrue(r["file_hit"] and r["symbol_hit"])
        self.assertEqual(cc.norm_fqn("pkg.mod.Class.__init__"), "pkg.mod.Class.__init__")  # a method stays

    def test_any_of_several_golds(self):
        q = {"gold_fqns": ["a.b.f", "a.c.g"], "gold_files": ["a/b.py", "a/c.py"]}
        self.assertTrue(self.g("FILE: a/c.py\nSYMBOL: a.c.g", q)["symbol_hit"])


class IsolationTest(unittest.TestCase):
    def write(self, events, debug):
        d = tempfile.mkdtemp()
        ev, dbg = os.path.join(d, "e.jsonl"), os.path.join(d, "d.log")
        with open(ev, "w") as fh:
            fh.write("\n".join(json.dumps(e) for e in events) + "\n")
        with open(dbg, "w") as fh:
            fh.write(debug)
        return cc.parse_events(ev), dbg

    CLEAN_DEBUG = "Registered 0 hooks from 0 plugins\n[claudeai-mcp] Disabled via env var\n"

    def init(self, tools, servers=()):
        return {"type": "system", "subtype": "init", "tools": tools, "plugins": [], "skills": [],
                "slash_commands": [], "memory_paths": None,
                "mcp_servers": [{"name": s, "status": "connected"} for s in servers]}

    def test_clean_arm_c(self):
        ev, dbg = self.write([self.init(["Glob", "Grep", "Read", "mcp__tuna__search"], ["tuna"])], self.CLEAN_DEBUG)
        self.assertEqual(cc.isolation_problems(ev, "C", dbg), [])

    def test_catches_leaks(self):
        bad = self.init(["Glob", "Grep", "Read", "Bash"])
        bad["plugins"] = [{"name": "x"}]
        ev, dbg = self.write([bad, {"type": "system", "subtype": "hook_started"}],
                             "hooks module cc-plugin-agents-md@builtin loaded (native)\n")
        p = " ".join(cc.isolation_problems(ev, "A", dbg))
        for needle in ("tools", "plugins", "hook events", "hooks module", "connectors"):
            self.assertIn(needle, p)

    def test_search_arm_needs_its_server(self):
        ev, dbg = self.write([self.init(["Glob", "Grep", "Read", "mcp__tuna__search"])], self.CLEAN_DEBUG)
        self.assertTrue(cc.isolation_problems(ev, "B", dbg))

    def test_flags_hash_differs_by_arm_only_in_tools(self):
        self.assertNotEqual(cc.flags_hash("A", cc.MODEL), cc.flags_hash("C", cc.MODEL))
        self.assertEqual(cc.flags_hash("B", cc.MODEL), cc.flags_hash("C", cc.MODEL))  # tag lives in the mcp json


class FailureTest(unittest.TestCase):
    INFO = {"exit": 1, "stderr": ""}

    def test_service(self):
        self.assertEqual(cc.failure_kind({"api_error_status": 529, "is_error": True}, self.INFO), "service")
        self.assertEqual(cc.failure_kind({"is_error": True, "result": "API Error: 500 Internal server error"},
                                         self.INFO), "service")
        self.assertEqual(cc.failure_kind(None, {"exit": 1, "stderr": "Error: socket hang up"}), "service")
        self.assertEqual(cc.failure_kind({"is_error": True, "result": "Claude usage limit reached"}, self.INFO),
                         "service")

    def test_agent(self):
        self.assertEqual(cc.failure_kind({"subtype": "error_max_turns", "is_error": True}, self.INFO), "agent")
        self.assertEqual(cc.failure_kind(None, {"exit": "timeout", "stderr": ""}), "agent")
        # a finished answer that merely mentions a rate limit is not a service failure
        self.assertEqual(cc.failure_kind({"subtype": "error_during_execution", "is_error": False,
                                          "result": "the rate limit middleware"}, self.INFO), "agent")

    def test_load_latest(self):
        rows = [{"qid": "q1", "arm": "A", "status": "service_fail", "wall_s": 1},
                {"qid": "q1", "arm": "A", "status": "ok", "wall_s": 2, "file_hit": True},
                {"qid": "q2", "arm": "A", "status": "agent_fail", "wall_s": 3, "file_hit": True, "cost_usd": 0.5}]
        d = tempfile.mkdtemp()
        p = os.path.join(d, "runs.jsonl")
        with open(p, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        latest, reruns = cc.load_latest(p, "r")
        self.assertEqual(reruns["A"], 1)
        self.assertEqual(latest[("r/q1", "A")]["status"], "ok")
        self.assertFalse(latest[("r/q2", "A")]["file_hit"])  # agent failure: a miss
        self.assertEqual(latest[("r/q2", "A")]["cost_usd"], 0.5)  # at its real cost


class ClusterBootstrapTest(unittest.TestCase):
    def test_point_and_interval(self):
        vals = {"r/a": 1.0, "r/b": 1.0, "r/c": 0.0, "s/d": 0.0, "s/e": 1.0}
        clusters = {"r": [["r/a", "r/b"], ["r/c"]], "s": [["s/d"], ["s/e"]]}
        m, lo, hi = cc.cluster_bootstrap(vals, clusters, n=2000)
        self.assertAlmostEqual(m, 0.6)
        self.assertLessEqual(lo, 0.6)
        self.assertGreaterEqual(hi, 0.6)

    def test_clusters_move_together(self):
        # two identical questions in one cluster: an interval as wide as one question, not two
        vals = {"r/a": 1.0, "r/b": 1.0, "r/c": 0.0}
        m, lo, hi = cc.cluster_bootstrap(vals, {"r": [["r/a", "r/b"], ["r/c"]]}, n=4000)
        self.assertEqual((lo, hi), (0.0, 1.0))


class BootstrapTest(unittest.TestCase):
    def test_interval_brackets_mean(self):
        m, lo, hi = cc.bootstrap([0, 1] * 25)
        self.assertAlmostEqual(m, 0.5)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


if __name__ == "__main__":
    unittest.main()
