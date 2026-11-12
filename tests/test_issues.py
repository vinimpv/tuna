"""Issue test set: diff-to-symbol mapping, fix links, text cleaning.
Run: .venv/bin/python -m unittest discover -s tests"""
import os
import subprocess
import tempfile
import textwrap
import unittest

import tree_sitter

from tuna import issues as I
from tuna import langs

OLD = textwrap.dedent('''\
    import os


    def untouched(x):
        return x


    class Cart:
        LIMIT = 3

        def add(self, item):
            self.items.append(item)
            return self

        @property
        def total(self):
            def price(i):
                return i.price
            return sum(price(i) for i in self.items)

        def remove(self, item):
            self.items.remove(item)
            # trailing comment
            return self
    ''')

NEW = textwrap.dedent('''\
    import os
    import sys


    def untouched(x):
        return x


    class Cart:
        LIMIT = 5

        def add(self, item):
            self.items.append(item)
            return self

        @property
        def total(self):
            def price(i):
                return i.price * i.quantity
            return sum(price(i) for i in self.items)

        def remove(self, item):
            self.items.remove(item)
            return self
    ''')

PARSER = tree_sitter.Parser(langs.PYTHON.language())


def git(cwd, *args):
    return subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True, text=True).stdout


class DiffToSymbols(unittest.TestCase):
    def test_parse_unified_zero(self):
        diff = textwrap.dedent('''\
            diff --git a/m.py b/m.py
            --- a/m.py
            +++ b/m.py
            @@ -1,0 +2 @@ import os
            +import sys
            @@ -9 +10 @@ class Cart:
            -    LIMIT = 3
            +    LIMIT = 5
            @@ -23 +23,0 @@ def remove
            -        # trailing comment
            diff --git a/gone.py b/gone.py
            --- a/gone.py
            +++ /dev/null
            @@ -1,2 +0,0 @@
            -def f():
            -    pass
            ''')
        files = I.parse_unified_zero(diff)
        self.assertEqual(files["m.py"]["plus"], {2, 10})
        self.assertEqual(files["m.py"]["minus"], {9, 23})
        self.assertEqual(files["gone.py"]["minus"], {1, 2})
        self.assertIsNone(files["gone.py"]["new"])

    def test_innermost_function_and_outside_lines(self):
        _, syms = I.extract.extract_file(langs.PYTHON, PARSER, "shop/cart.py", NEW.encode())
        hit, outside = I.innermost_functions(syms, [2, 10, 19, 17])
        # line 19 is in the nested price(); line 17 is the decorated def line of total
        self.assertEqual(hit, {"shop.cart.Cart.total.price": "Function", "shop.cart.Cart.total": "Method"})
        self.assertEqual(outside, [2, 10])  # the import and the class attribute

    def test_changed_functions_both_sides(self):
        # plus side: price() body; minus side: the comment deleted from remove(),
        # which leaves no line on the new side inside remove()
        fqns, outside = I.changed_functions(langs.PYTHON, PARSER, "shop/cart.py", OLD.encode(), "shop/cart.py",
                                            NEW.encode(), minus={9, 18, 23}, plus={2, 10, 19})
        self.assertEqual(fqns, {"shop.cart.Cart.total.price", "shop.cart.Cart.remove"})
        self.assertEqual(outside, 2)  # LIMIT on both sides; the import does not count

    def test_commit_gold_on_a_real_repo(self):
        with tempfile.TemporaryDirectory() as d:
            git(d, "init", "-q")
            git(d, "config", "user.email", "t@example.com")
            git(d, "config", "user.name", "t")
            os.makedirs(os.path.join(d, "shop", "tests"))
            os.makedirs(os.path.join(d, "web"))
            for path, text in (("shop/cart.py", OLD), ("shop/tests/test_cart.py", "def test_x():\n    pass\n"),
                               ("web/cart.ts", "export const a = 1;\n")):
                with open(os.path.join(d, path), "w") as fh:
                    fh.write(text)
            git(d, "add", "-A")
            git(d, "commit", "-qm", "init")
            with open(os.path.join(d, "shop", "cart.py"), "w") as fh:
                fh.write(NEW)
            with open(os.path.join(d, "shop", "tests", "test_cart.py"), "w") as fh:
                fh.write("def test_x():\n    assert True\n")
            with open(os.path.join(d, "web", "cart.ts"), "w") as fh:
                fh.write("export const a = 2;\n")
            git(d, "commit", "-qam", "Fix the cart total.\n\nFixes #12.")
            cfg = {"include": ["shop/*"], "exclude": ["*/tests/*"]}
            commit = I.read_commits(d, "HEAD")[0]
            self.assertEqual(I.closing_refs(commit["body"], "o", "r"), [12])
            g = I.commit_gold(d, cfg, commit, langs.PYTHON, PARSER)
            self.assertEqual(g["fqns"], {"shop.cart.Cart.total.price", "shop.cart.Cart.remove"})
            self.assertEqual(g["in_scope_files"], ["shop/cart.py"])
            self.assertEqual(g["other_code_files"], ["web/cart.ts"])  # tests are neither


class Links(unittest.TestCase):
    def test_closing_refs(self):
        f = lambda t: I.closing_refs(t, "zulip", "zulip")  # noqa: E731
        self.assertEqual(f("Fixes #12."), [12])
        self.assertEqual(f("fixes: #12, #13 and #14"), [12, 13, 14])
        self.assertEqual(f("Closes https://github.com/zulip/zulip/issues/77"), [77])
        self.assertEqual(f("Resolved #5\nFixes #5"), [5])
        self.assertEqual(f("Fixes part of #12."), [])
        self.assertEqual(f("Partially fixes #12."), [])
        self.assertEqual(f("Fixes other/repo#12 and https://github.com/x/y/issues/3"), [])
        self.assertEqual(f("Fix #12 crash"), [12])
        self.assertEqual(f("prefix-fixes #12"), [])

    def test_pr_numbers(self):
        self.assertEqual(I.pr_numbers("Fix checkout (#123)", ""), [123])
        self.assertEqual(I.pr_numbers("Merge pull request #45 from a/b", ""), [45])
        self.assertEqual(I.pr_numbers("Fix auth (#19748)", "Backport of #19747."), [19748, 19747])
        self.assertEqual(I.pr_numbers("Fix #12 in checkout", ""), [])


class Text(unittest.TestCase):
    def test_clean_body(self):
        body = textwrap.dedent('''\
            <!-- Please describe the bug -->
            ### What are you trying to achieve?
            Vouchers apply twice when the cart has two lines.

            ### Steps to reproduce the problem
            1. Add a voucher
            - [ ] I searched existing issues
            ![shot](https://user-images.githubusercontent.com/1/2.png)

            ```
            ''') + "\n".join(f"log line {i}" for i in range(30)) + textwrap.dedent('''
            ```

            ```python
            total = cart.total()
            ```

            ### Environment
            Saleor version: 3.1
            OS and version: Ubuntu
            ''')
        out = I.clean_body(body)
        self.assertEqual(out, "Vouchers apply twice when the cart has two lines.\n\n1. Add a voucher\n\n"
                              "```python\ntotal = cart.total()\n```")

    def test_clean_feature_templates(self):
        body = ("### What I'm trying to achieve\nSort orders by the newest first.\n\n"
                "### Other solutions I've tried and won't work\nN/A\n\n### Screenshots or mockups\nsee below\n\n"
                "**Describe the solution you'd like**\nA sort option.\n\n"
                "#### *Thank you* for your feature request – we love each and every one!\n")
        self.assertEqual(I.clean_body(body), "Sort orders by the newest first.\n\nA sort option.")

    def test_language_and_length(self):
        self.assertTrue(I.is_english("The checkout total is wrong when a voucher is applied to a cart "
                                     "with several lines and a shipping method."))
        self.assertFalse(I.is_english("O total do carrinho fica errado quando um cupom é aplicado com várias "
                                      "linhas e um método de envio."))
        self.assertFalse(I.is_english("当购物车中有多个商品时，优惠券会被重复使用。"))
        self.assertEqual(I.word_count("Two words\n```\nmany code words here\n```\nhttps://x.y/z"), 2)

    def test_leaks(self):
        gold = [("zerver.actions.users.do_change_user_role", "zerver/actions/users.py")]
        self.assertEqual(I.leaks("Calling do_change_user_role fails", gold), (["do_change_user_role"], []))
        self.assertEqual(I.leaks("See doChangeUserRole", gold), (["do_change_user_role"], []))
        self.assertEqual(I.leaks("The bug is in actions/users.py", gold), ([], ["actions/users.py"]))
        self.assertEqual(I.leaks("in zerver.actions.users somewhere", gold), ([], ["zerver.actions.users"]))
        self.assertEqual(I.leaks("Changing a user's role does nothing", gold), ([], []))
        one = [("saleor.order.models.Order.save", "saleor/order/models.py")]
        self.assertEqual(I.leaks("I cannot save the order", one), ([], []))
        self.assertEqual(I.leaks("order.save() raises", one), (["save"], []))
        self.assertEqual(I.leaks("models.py in order", one), ([], ["models.py"]))


if __name__ == "__main__":
    unittest.main()
