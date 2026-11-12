"""Long-symbol chunking and best-chunk scoring.  Run: .venv/bin/python -m unittest discover -s tests"""
import unittest

import numpy as np

from tuna import chunking


def words(text):
    """Fake tokenizer: one token per whitespace-separated word, plus one special token."""
    return len(text.split()) + 1


def doc(fqn, n_lines, words_per_line=5):
    head = f"# FQN: {fqn}\n# File: pkg/mod.py:1-{n_lines}"
    body = [" ".join(f"w{i}_{j}" for j in range(words_per_line)) for i in range(n_lines)]
    return head + "\n" + "\n".join(body)


HEADER_TOKENS = 6  # "# FQN: x" + "# File: y" = 3 + 3 words


class SplitDocument(unittest.TestCase):
    def test_fits_is_unchanged(self):
        d = doc("pkg.mod.small", 3)
        chunks, hard = chunking.split_document(d, words, limit=100)
        self.assertEqual(chunks, [d])
        self.assertEqual(hard, 0)

    def test_exactly_at_limit_is_not_split(self):
        d = doc("pkg.mod.exact", 4)
        chunks, _ = chunking.split_document(d, words, limit=words(d))
        self.assertEqual(chunks, [d])

    def test_long_symbol_splits_on_lines_with_header(self):
        d = doc("pkg.mod.big", 40)
        limit = HEADER_TOKENS + 1 + 5 * 6  # header, special token, six 5-word lines
        chunks, hard = chunking.split_document(d, words, limit)
        self.assertEqual(hard, 0)
        self.assertEqual(len(chunks), 7)  # 40 lines, 6 per chunk
        head = "\n".join(d.split("\n")[:2])
        for c in chunks:
            self.assertLessEqual(words(c), limit)
            self.assertTrue(c.startswith(head + "\n"))
        # Lossless: the chunk bodies, in order, are the original body.
        body = "\n".join("\n".join(c.split("\n")[2:]) for c in chunks)
        self.assertEqual(body, "\n".join(d.split("\n")[2:]))

    def test_single_line_over_limit_is_cut_and_counted(self):
        head = "# FQN: pkg.mod.blob\n# File: pkg/mod.py:1-3"
        long_line = " ".join(f"x{i}" for i in range(50))
        d = f"{head}\nshort line\n{long_line}\nend"
        limit = HEADER_TOKENS + 1 + 10
        chunks, hard = chunking.split_document(d, words, limit)
        self.assertEqual(hard, 1)
        for c in chunks:
            self.assertLessEqual(words(c), limit)
            self.assertTrue(c.startswith(head + "\n"))
        pieces = [c.split("\n", 2)[2] for c in chunks]
        self.assertEqual(pieces[0], "short line")
        self.assertEqual(pieces[-1], "end")
        self.assertEqual("".join(pieces[1:-1]), long_line)

    def test_header_alone_over_limit_raises(self):
        with self.assertRaises(ValueError):
            chunking.split_document(doc("pkg.mod.x", 50), words, limit=HEADER_TOKENS)


class SplitSymbols(unittest.TestCase):
    def test_owner_and_stats(self):
        docs = [doc("a", 2), doc("b", 40), doc("c", 1), doc("d", 13)]
        limit = HEADER_TOKENS + 1 + 5 * 6
        texts, owner, st = chunking.split_symbols(docs, words, limit)
        self.assertEqual(owner, [0] + [1] * 7 + [2] + [3] * 3)
        self.assertEqual(len(texts), len(owner))
        self.assertEqual((st.symbols, st.split, st.chunks, st.max_chunks, st.hard_split_lines, st.limit),
                         (4, 2, 12, 7, 0, limit))
        self.assertIn("2 of 4 symbols over", chunking.describe(st, "repo", "model"))


class BestChunk(unittest.TestCase):
    def test_max_per_symbol(self):
        owner = [0, 1, 1, 1, 2]
        scores = np.array([[0.1, 0.5, 0.9, 0.2, 0.3],
                           [0.4, 0.1, 0.0, 0.2, 0.8]])
        np.testing.assert_allclose(chunking.max_per_symbol(scores, owner, 3), [[0.1, 0.9, 0.3], [0.4, 0.2, 0.8]])
        np.testing.assert_allclose(chunking.max_per_symbol(scores[0], owner, 3), [0.1, 0.9, 0.3])

    def test_owner_must_cover_every_symbol_in_order(self):
        with self.assertRaises(ValueError):
            chunking.max_per_symbol(np.zeros(3), [0, 2, 2], 3)
        with self.assertRaises(ValueError):
            chunking.max_per_symbol(np.zeros(3), [1, 0, 2], 3)


class MaxSim(unittest.TestCase):
    def test_matches_brute_force_across_blocks(self):
        from tuna.colbert import maxsim
        rng = np.random.default_rng(0)
        lengths = [3, 1, 7, 2, 5]
        tokens = rng.standard_normal((sum(lengths), 8)).astype(np.float16)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        q = rng.standard_normal((4, 8)).astype(np.float32)
        expect = [(q @ tokens[offsets[i]:offsets[i + 1]].astype(np.float32).T).max(1).sum()
                  for i in range(len(lengths))]
        for block in (1, 4, 1 << 20):  # tiny blocks force chunk-boundary handling
            np.testing.assert_allclose(maxsim(q, tokens, offsets, "cpu", block_tokens=block), expect, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
