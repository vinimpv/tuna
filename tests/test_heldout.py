"""Held-out files keep test sets out of training.  Run: .venv/bin/python -m unittest discover -s tests"""
import collections
import unittest

from tuna import heldout
from tuna import questions as Q

FILE_OF = {f"pkg.m{i}.f{j}": f"pkg/m{i}.py" for i in range(30) for j in range(4)}
SYMS = [{"fqn": fqn, "file_path": f, "kind": "Function"} for fqn, f in FILE_OF.items()]
TEST_ROWS = [
    {"id": "t1", "question": "Where are carts cleaned up?", "file": "pkg/m1.py", "gold_fqn": "pkg.m1.f0",
     "gold_fqns": ["pkg.m1.f0"]},
    # an issue-style row: no target file, several golds in other files
    {"id": "t2", "question": "Crash when saving a draft", "gold_fqns": ["pkg.m2.f1", "pkg.m3.f2"]},
]


class Build(unittest.TestCase):
    def test_targets_and_every_gold_file(self):
        self.assertEqual(heldout.heldout_files(TEST_ROWS, FILE_OF), {"pkg/m1.py", "pkg/m2.py", "pkg/m3.py"})

    def test_unknown_gold_is_an_error(self):
        with self.assertRaises(ValueError):
            heldout.heldout_files([{"question": "x", "gold_fqns": ["pkg.gone.f"]}], FILE_OF)


class Sampling(unittest.TestCase):
    def test_sampler_never_returns_heldout_files(self):
        held = heldout.heldout_files(TEST_ROWS, FILE_OF)
        for seed in range(50):
            order = Q.file_order(SYMS, seed, exclude=held)
            self.assertFalse(held & set(order))
            self.assertEqual(len(order), 30 - len(held))


class Check(unittest.TestCase):
    held = {"pkg/m1.py", "pkg/m2.py", "pkg/m3.py"}

    def train(self, **kw):
        row = {"id": "x1", "question": "How are refunds rounded?", "file": "pkg/m9.py", "gold_fqn": "pkg.m9.f0",
               "gold_fqns": ["pkg.m9.f0"]}
        row.update(kw)
        return heldout.check_training([row], self.held, TEST_ROWS, FILE_OF)

    def test_clean_row(self):
        self.assertEqual(self.train(), [])

    def test_each_leak_is_caught(self):
        self.assertTrue(any("targets held-out file" in p for p in self.train(file="pkg/m2.py")))
        self.assertTrue(any("lives in held-out file" in p
                            for p in self.train(gold_fqn="pkg.m3.f0", gold_fqns=["pkg.m3.f0"])))
        self.assertTrue(any("is a test gold" in p for p in self.train(gold_fqn="pkg.m1.f0", gold_fqns=["pkg.m1.f0"])))
        self.assertTrue(any("equals a test question" in p for p in self.train(question="Where are carts cleaned up? ")))


class Split(unittest.TestCase):
    def test_grouped_by_file_and_sized(self):
        qs = [{"id": f"q{i}", "file": f"f{i % 37}.py"} for i in range(500)]
        val = Q.split_by_file(qs, 0.1, 7)
        self.assertEqual(val, Q.split_by_file(qs, 0.1, 7))
        n_val = sum(q["file"] in val for q in qs)
        self.assertTrue(50 <= n_val <= 50 + max(collections.Counter(q["file"] for q in qs).values()))
        # a file is wholly in one split
        self.assertEqual({q["file"] for q in qs if q["file"] in val}, val)


if __name__ == "__main__":
    unittest.main()
