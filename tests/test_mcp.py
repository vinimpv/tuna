"""tuna mcp: search over a saved index, JSON-RPC dispatch, and an end-to-end stdio run.

Run: .venv/bin/python -m unittest tests.test_mcp
The end-to-end test needs the Saleor lfm25-emb index under $TUNA_OUT (default
~/tuna-data/out) and the LFM2.5 snapshot in the local HF cache; it is skipped otherwise.
"""
import json
import os
import subprocess
import sys
import threading
import unittest

import numpy as np

from tuna import common, mcp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sym(fqn, path, a, b, body):
    return {"fqn": fqn, "file_path": path, "line_start": a, "line_end": b,
            "body": f"# FQN: {fqn}\n# File: {path}:{a}-{b}\n{body}"}


SYMS = [sym("pkg.a.f", "pkg/a.py", 1, 3, "def f():\n    return 1"),
        sym("pkg.a.C", "pkg/a.py", 5, 20, "class C:\n    '''Long class.'''\n    x = 1"),
        sym("pkg.b.g", "pkg/b.py", 1, 2, "def g():\n    pass")]
# symbol 1 is split into two chunks; unit vectors on axes
M = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0], [0, 0, 1]], dtype=np.float32)
OWNER = np.array([0, 1, 1, 2])
QUERIES = {"x": np.array([1, 0, 0], np.float32), "y": np.array([0, 1, 0], np.float32),
           "z": np.array([0, 0, 1], np.float32)}


def index():
    return mcp.Index(M, OWNER, SYMS, lambda q: QUERIES[q])


class IndexTest(unittest.TestCase):
    def test_ranks_by_best_chunk(self):
        hits = index().search("x", k=3)
        self.assertEqual([h["fqn"] for h in hits], ["pkg.a.f", "pkg.a.C", "pkg.b.g"])  # C scores 0.6 via chunk 2
        self.assertEqual(hits[0]["rank"], 1)
        self.assertEqual((hits[1]["file"], hits[1]["line_start"], hits[1]["line_end"]), ("pkg/a.py", 5, 20))

    def test_k_is_clamped(self):
        self.assertEqual(len(index().search("z", k=1)), 1)
        self.assertEqual(len(index().search("z", k=999)), 3)
        self.assertEqual(len(index().search("z", k=0)), 1)

    def test_snippet_drops_header_and_caps(self):
        s = mcp.snippet(SYMS[1]["body"])
        self.assertNotIn("# FQN", s)
        self.assertTrue(s.startswith("class C:"))
        long = "# FQN: x\n" + "\n".join(f"line {i}" for i in range(50))
        self.assertEqual(len(mcp.snippet(long).splitlines()), mcp.SNIPPET_LINES)
        self.assertTrue(mcp.snippet("a" * 1000).endswith(" ..."))

    def test_format(self):
        text = mcp.format_results(index().search("y", k=1))
        self.assertTrue(text.startswith("1. pkg.a.C  (pkg/a.py:5-20)\n   class C:"))
        self.assertEqual(mcp.format_results([]), "No results.")


class FakeBM25:
    def __init__(self, order):
        self.order = order

    def top(self, query, k):
        return [(i, 10.0 - r) for r, i in enumerate(self.order[:k])]


class FusionTest(unittest.TestCase):
    def test_rrf_matches_scripts_rrf(self):
        # dense for "x": f(1.0) > C(0.6) > g(0); BM25: g > C > f. RRF: g = f = 1/61 + 1/63 > C = 2/62.
        idx = mcp.Index(M, OWNER, SYMS, lambda q: QUERIES[q], bm25=FakeBM25([2, 1, 0]))
        got = [h["fqn"] for h in idx.search("x", k=3)]
        self.assertEqual(got, ["pkg.b.g", "pkg.a.f", "pkg.a.C"])  # g ties f; BM25's order first, as rrf.py
        self.assertAlmostEqual(idx.search("x", k=1)[0]["score"], round(1 / 61 + 1 / 63, 4))

    def test_bm25_only_symbols_still_rank(self):
        idx = mcp.Index(M, OWNER, SYMS, lambda q: QUERIES[q], bm25=FakeBM25([2]))
        self.assertEqual(idx.search("x", k=1)[0]["fqn"], "pkg.b.g")  # 1/61 + 1/63 > f's 1/61


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.out = []
        self.server = mcp.Server(index, self.out.append)
        self.server.ready.wait(5)

    def msgs(self):
        return [json.loads(m) for m in self.out]

    def test_initialize_and_list(self):
        self.server.serve([json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                       "params": {"protocolVersion": "2025-06-18"}}),
                           json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                           json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})])
        init, tools = self.msgs()
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", init["result"]["capabilities"])
        self.assertEqual([t["name"] for t in tools["result"]["tools"]], ["search"])
        self.assertEqual(tools["id"], 2)

    def test_unknown_protocol_version_gets_latest(self):
        self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999"}})
        self.assertEqual(self.msgs()[0]["result"]["protocolVersion"], mcp.PROTOCOL_VERSIONS[0])

    def test_call(self):
        self.server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "search", "arguments": {"query": "z", "k": 2}}})
        res = self.msgs()[0]["result"]
        self.assertFalse(res.get("isError"))
        self.assertTrue(res["content"][0]["text"].startswith("1. pkg.b.g  (pkg/b.py:1-2)"))

    def test_bad_calls(self):
        for params in ({"name": "nope"}, {"name": "search", "arguments": {}},
                       {"name": "search", "arguments": {"query": "x", "k": "many"}}):
            self.out.clear()
            self.server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": params})
            self.assertTrue(self.msgs()[0]["result"]["isError"])

    def test_unknown_method_and_parse_error(self):
        self.server.serve(['{"jsonrpc":"2.0","id":5,"method":"resources/list"}', "{not json",
                           '{"jsonrpc":"2.0","method":"notifications/cancelled"}'])
        a, b = self.msgs()
        self.assertEqual(a["error"]["code"], -32601)
        self.assertEqual(b["error"]["code"], -32700)

    def test_load_failure_is_reported(self):
        out = []

        def boom():
            raise FileNotFoundError("no vectors")
        s = mcp.Server(boom, out.append)
        s.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "search", "arguments": {"query": "x"}}})
        res = json.loads(out[0])["result"]
        self.assertTrue(res["isError"])
        self.assertIn("no vectors", res["content"][0]["text"])

    def test_search_waits_for_load(self):
        gate = threading.Event()

        def slow():
            gate.wait(5)
            return index()
        out = []
        s = mcp.Server(slow, out.append)
        t = threading.Thread(target=s.handle, args=({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                     "params": {"name": "search", "arguments": {"query": "x"}}},))
        t.start()
        s.handle({"jsonrpc": "2.0", "id": 0, "method": "ping"})  # answered while the index loads
        self.assertEqual(json.loads(out[0])["id"], 0)
        gate.set()
        t.join(5)
        self.assertIn("pkg.a.f", json.loads(out[1])["result"]["content"][0]["text"])


def e2e_ready():
    vec = os.path.join(common.default_out(), "saleor", "vectors", "lfm25-emb.npy")
    return os.path.exists(vec)


@unittest.skipUnless(e2e_ready(), "Saleor lfm25-emb index not on this machine")
class EndToEndTest(unittest.TestCase):
    """Drive scripts/tuna-mcp over stdio with a real question, from a directory that has a `saleor/`
    folder (as a checkout does), and check the CPU ranking against the saved GPU rank file."""
    QID = "saleor-claude-m3-001"

    def run_server(self, tag, question, extra=()):
        reqs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "search", "arguments": {"query": question, "k": 10}}}]
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "saleor"))
            log = os.path.join(tmp, "search.jsonl")
            p = subprocess.run([os.path.join(ROOT, "scripts", "tuna-mcp"), "--repo", "saleor", "--tag", tag,
                                "--log", log, *extra], input="".join(json.dumps(r) + "\n" for r in reqs), cwd=tmp,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            msgs = [json.loads(l) for l in p.stdout.splitlines()]  # stdout carries nothing but JSON-RPC
            self.assertEqual([m["id"] for m in msgs], [1, 2, 3])
            text = msgs[2]["result"]["content"][0]["text"]
            self.assertFalse(msgs[2]["result"].get("isError"), text)
            self.assertEqual(len([l for l in text.splitlines() if l[:1].isdigit()]), 10)
            with open(log) as fh:
                return json.loads(fh.readline())["ranked"]

    def check(self, tag, saved_row=None, qset="claude-m3", qid=None, extra=()):
        ranks = os.path.join(common.default_out(), "saleor", "ranks", f"{saved_row or tag}_{qset}.jsonl")
        if not os.path.exists(ranks):
            self.skipTest(f"{ranks} missing")
        saved = next(r for r in common.read_jsonl(ranks) if r["id"] == (qid or self.QID))
        got = self.run_server(tag, saved["question"], extra)
        # CPU bf16 vs the GPU run that wrote the rank file: same top 3, near-identical top 10
        self.assertEqual(got[:3], saved["ranked"][:3])
        self.assertGreaterEqual(len(set(got) & set(saved["ranked"][:10])), 8)

    def test_stock(self):
        self.check("lfm25-emb")

    def test_tuned_lora(self):
        self.check("lfm25-ft-lora")

    def test_bm25_alone(self):
        """BM25 alone (tag bm25, used by the H-bm25 card arm), against `tuna bm25`'s saved issue ranks."""
        self.check("bm25", "bm25", "issues-v1", "saleor-issues-v1-001")

    def test_bm25_fused_qwen(self):
        """Arm D: BM25 + tuned Qwen3 LoRA, RRF, against scripts/rrf.py's saved fusion for an issue."""
        self.check("qwen3-0.6b-ft-lora", "rrf-bm25+qwen3-0.6b-ft-lora", "issues-v1", "saleor-issues-v1-001",
                   ["--fuse-bm25"])


if __name__ == "__main__":
    unittest.main()
