"""tuna's prefix + encode path gives the model's own prompt_name encoding.

Loads LFM2.5-Embedding-350M at its pinned revision, so it only runs with
TUNA_MODEL_TESTS=1 (and uses CUDA when available):

  TUNA_MODEL_TESTS=1 .venv/bin/python -m unittest tests.test_prompts
"""
import os
import unittest

MODEL = "LiquidAI/LFM2.5-Embedding-350M"


@unittest.skipUnless(os.environ.get("TUNA_MODEL_TESTS") == "1", "needs the model: set TUNA_MODEL_TESTS=1")
class PrefixEqualsPromptName(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from tuna import encoders
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.enc = encoders.LocalEncoder(MODEL, device=cls.device)
        cls.st = cls.enc._st

    def test_prefixes_are_the_models_prompts(self):
        self.assertEqual(self.enc.prefix, self.st.prompts["document"])
        self.assertEqual(self.enc.query_prefix, self.st.prompts["query"])

    def test_documents_and_queries(self):
        import numpy as np
        docs = ["def add(a, b):\n    return a + b", "class Cart:\n    \"\"\"Items a shopper is buying.\"\"\""]
        qs = ["Where are two numbers summed?", "What holds the items a customer will pay for?"]
        ours_d = self.enc.encode_documents(docs)
        ours_q = self.enc.encode_queries(qs)
        ref_d = self.st.encode(docs, prompt_name="document", normalize_embeddings=True, convert_to_numpy=True)
        ref_q = self.st.encode(qs, prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)
        np.testing.assert_allclose(ours_d, ref_d, atol=2e-3)
        np.testing.assert_allclose(ours_q, ref_q, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
