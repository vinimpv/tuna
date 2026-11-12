"""The v2 identifier-token filter.  Run: .venv/bin/python -m unittest discover -s tests"""
import unittest

from tuna import question_filter as F


def rules(q, file_words=frozenset()):
    return sorted({r for r, _ in F.identifier_tokens(q, file_words)})


class Tokens(unittest.TestCase):
    def test_clean_questions_pass(self):
        for q in ("What happens to a shopper's cart when an item in it sells out?",
                  "Where does the GraphQL API decide whether a staff user may refund an order?",
                  "How are Stripe webhooks for failed payments handled, e.g. retries?",
                  "Which URLs and APIs does the Node.js client call when OAuth tokens expire?",
                  "What sets the Retry-After header when the server rate-limits an X-Forwarded-For client?",
                  "Where are UUIDs for new PostgreSQL rows generated in Django?"):
            self.assertEqual(rules(q), [], q)

    def test_each_rule(self):
        self.assertEqual(rules("When is page_type_deleted sent to apps?"), ["underscore"])
        self.assertEqual(rules("What does the PromoCodeRejectionReason enum cover?"), ["mixed_case"])
        self.assertEqual(rules("How is mergeStatus computed for a pull request?"), ["mixed_case"])
        self.assertEqual(rules("Where is order.lines filtered before invoicing?"), ["dotted"])
        self.assertEqual(rules("Which code reads settings.py on startup?"), ["dotted"])
        self.assertEqual(rules("What does `save` do for new products?"), ["code_syntax"])
        self.assertEqual(rules("When is save() called on the checkout?"), ["code_syntax"])

    def test_file_word_needs_file_and_non_english(self):
        q = "How are the attrs of a variant merged when importing products?"
        self.assertEqual(rules(q), [])                                  # not an identifier of the file
        self.assertEqual(rules(q, {"attrs", "variant"}), ["file_word"])  # identifier word, not English
        self.assertEqual(rules("How are variant prices rounded?", {"variant", "prices"}), [])  # English
        self.assertEqual(rules("Which webhook fires on refunds for each backend?", {"webhook", "backend"}), [])

    def test_allowed_tokens_are_not_split_into_file_words(self):
        self.assertEqual(rules("How does the IPv6 check treat Auth0 callbacks?", {"ipv", "auth"}), [])

    def test_identifier_words_from_code_only(self):
        src = b'''
def get_attrs(qtyLeft):
    """Docstring words like zzqx are not identifiers."""
    # nor are comment words like yyqw
    return obj.fooBar
'''
        words = F.identifier_words(src)
        self.assertLessEqual({"attrs", "qty", "left", "obj", "foo", "bar"}, words)
        self.assertNotIn("zzqx", words)
        self.assertNotIn("yyqw", words)


if __name__ == "__main__":
    unittest.main()
