"""Hard-negative mining rules (tuna.train.mine).  Run: .venv/bin/python -m unittest discover -s tests"""
import unittest

import numpy as np

from tuna import train

# five symbols, one chunk each; symbol 0 is the gold, 1 is its method (nested), 4 is in a held-out file
SYMS = [{"fqn": "pkg.m.Cart", "file_path": "pkg/m.py"},
        {"fqn": "pkg.m.Cart.total", "file_path": "pkg/m.py"},
        {"fqn": "pkg.n.price", "file_path": "pkg/n.py"},
        {"fqn": "pkg.n.tax", "file_path": "pkg/n.py"},
        {"fqn": "pkg.h.secret", "file_path": "pkg/h.py"}]
BY_FQN = {s["fqn"]: i for i, s in enumerate(SYMS)}
OWNER = [0, 1, 2, 3, 4]
ROWS = [{"id": "q1", "gold_fqn": "pkg.m.Cart"}]
HELD = {"pkg/h.py"}
Q = np.array([[1.0, 0.0]])
# stock scores: gold 0.50, nested method 0.90, price 0.49 (near tie), tax 0.20, held-out 0.95
D = np.array([[0.50, 0], [0.90, 0], [0.49, 0], [0.20, 0], [0.95, 0]])


class Mine(unittest.TestCase):
    def test_pool_and_structure(self):
        (pos, cand), = train.mine(Q, D, OWNER, SYMS, ROWS, BY_FQN, HELD, depth=10)
        self.assertEqual(pos, 0)
        self.assertNotIn(4, cand)  # held-out file: never a negative
        self.assertNotIn(1, cand)  # nested in the gold: never a negative
        self.assertEqual(cand, [2, 3])

    def test_own_score_rule_drops_near_ties(self):
        stats = {}
        (_, cand), = train.mine(Q, D, OWNER, SYMS, ROWS, BY_FQN, HELD, 10, train.own_score_rule(0.95), stats)
        self.assertEqual(cand, [3])
        self.assertEqual(stats["fn_dropped"], 1)
        self.assertAlmostEqual(stats["fn_dropped_share"], 0.5)

    def test_judge_rule_is_shared(self):
        # the judge thinks "tax" also answers the question (0.97 x gold), "price" does not
        judge = np.array([[0.80, 0.70, 0.30, 0.78, 0.10]])
        rule = train.judge_rule(judge, OWNER, [0], 0.95)
        (_, cand), = train.mine(Q, D, OWNER, SYMS, ROWS, BY_FQN, HELD, 10, rule)
        self.assertEqual(cand, [2])
        # another model with different stock scores: the same symbols are excluded
        D2 = np.array([[0.10, 0], [0.20, 0], [0.90, 0], [0.80, 0], [0.30, 0]])
        (_, cand2), = train.mine(Q, D2, OWNER, SYMS, ROWS, BY_FQN, HELD, 10, rule)
        self.assertEqual(cand2, [2])


if __name__ == "__main__":
    unittest.main()
