"""Find identifier-like tokens in a search question (the v2 question filter).

A test question should read like a developer describing behaviour, not like
code. `identifier_tokens(question, file_words)` returns every offending token
as (rule, token):

  underscore   a token containing `_`                       page_type_deleted
  mixed_case   a capital after the first letter, with lower   fooBar, FooBar
               case present, in any hyphen-separated part
               (acronyms and their plurals, like API or URLs,
               and Title-Case header names like Retry-After
               are fine)
  dotted       a dotted name                                 order.lines, utils.py
  code_syntax  backticks, or parentheses right after a word  `x`, save()
  file_word    a word of 3+ letters that is one of the file's  "attrs" when the file
               identifier words but not an English word       has get_attrs()

Product and framework names in data/allowed-terms.txt (GraphQL, Node.js,
webhook, ...) are never flagged. English words come from
data/english-words.txt.gz (SCOWL, see data/NOTICE-english-words).
"""
import functools
import gzip
import os
import re

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
ABBREVIATIONS = {"e.g", "i.e", "a.m", "p.m", "u.s", "etc", "vs"}
EDGE = "\"'“”‘’.,;:!?()[]{}<>*"


@functools.lru_cache(maxsize=1)
def english_words():
    with gzip.open(os.path.join(DATA, "english-words.txt.gz"), "rt", encoding="ascii") as fh:
        return frozenset(fh.read().split())


@functools.lru_cache(maxsize=1)
def allowed_terms():
    with open(os.path.join(DATA, "allowed-terms.txt"), encoding="utf-8") as fh:
        return frozenset(l.strip().lower() for l in fh if l.strip() and not l.lstrip().startswith("#"))


def identifier_words(source_bytes):
    """Lower-case words (3+ letters) of every identifier in a Python file: names,
    attributes, parameters. Comments and strings are not identifiers."""
    from . import langs
    import tree_sitter
    tree = tree_sitter.Parser(langs.PYTHON.language()).parse(source_bytes)
    out, stack = set(), [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "identifier":
            for piece in node.text.decode("utf-8", "replace").split("_"):
                out.update(p.lower() for p in CAMEL.findall(piece) if len(p) >= 3 and p.isalpha())
        stack.extend(node.children)
    return out


def _allowed(token):
    t = token.lower()
    return t in allowed_terms() or (t.endswith("s") and t[:-1] in allowed_terms())


def _bare(token):
    t = token.strip(EDGE)
    return re.sub(r"['’]s$", "", t)


def identifier_tokens(question, file_words=frozenset()):
    found = []
    if "`" in question:
        found.append(("code_syntax", "`"))
    for m in re.finditer(r"[A-Za-z_][\w.]*\(", question):
        found.append(("code_syntax", m.group(0)))
    for raw in question.split():
        tok = _bare(raw)
        if not tok or _allowed(tok):
            continue
        if "_" in tok:
            found.append(("underscore", tok))
            continue
        if re.search(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", tok) and tok.lower().rstrip(".") not in ABBREVIATIONS:
            found.append(("dotted", tok))
            continue
        # Per hyphen part, so protocol names like Retry-After or X-Forwarded-For pass.
        for part in tok.split("-"):
            letters = re.sub(r"[^A-Za-z]", "", part)
            acronym = re.fullmatch(r"[A-Z0-9]+s?", part) is not None
            if (not acronym and not _allowed(part) and any(c.isupper() for c in letters[1:])
                    and any(c.islower() for c in letters)):
                found.append(("mixed_case", tok))
                break
    english, allowed = english_words(), allowed_terms()
    for raw in question.split():
        if _allowed(_bare(raw)):  # IPv6 or Auth0 must not yield "IPv" or "Auth"
            continue
        for w in re.findall(r"[A-Za-z]{3,}", raw):
            lw = w.lower()
            if lw in file_words and lw not in english and lw not in allowed:
                found.append(("file_word", w))
    return found
