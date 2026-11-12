"""Split symbol documents that exceed a model's input limit.

Every model reads a symbol at its own maximum length. A document that fits is
used whole. A longer one is split on line boundaries into chunks that each fit,
and every chunk repeats the document's two-line header (`# FQN:` / `# File:`),
so each chunk still says which symbol it belongs to. A symbol's score is its
best chunk's score (`max_per_symbol`). Nothing is truncated: the only cut below
line level is a single line that cannot fit on its own (a huge literal), which
is split between tokens and counted as `hard_split_lines`.

`count_tokens(text)` must return what the model will see for `text`, prompt or
prefix and special tokens included, so a chunk is never cut by the model.
"""
import collections

HEADER_LINES = 2

Stats = collections.namedtuple("Stats", "symbols split chunks max_chunks hard_split_lines limit")


def _header_and_lines(doc):
    lines = doc.split("\n")
    head, body = lines[:HEADER_LINES], lines[HEADER_LINES:]
    if len(head) < HEADER_LINES or not head[0].startswith("# FQN:"):
        head, body = [], lines  # not a tuna symbol document: split without a header
    return head, body


def split_document(doc, count_tokens, limit):
    """[doc] if it fits in `limit` tokens, else header-prefixed chunks that do.
    Returns (chunks, hard_split_lines)."""
    if count_tokens(doc) <= limit:
        return [doc], 0
    head, body = _header_and_lines(doc)
    prefix = "\n".join(head) + "\n" if head else ""
    if count_tokens(prefix) >= limit:
        raise ValueError(f"limit {limit} leaves no room after the header of {head[0] if head else doc[:60]!r}")

    def fits(lines):
        return count_tokens(prefix + "\n".join(lines)) <= limit

    chunks, current, hard = [], [], 0
    for line in body:
        if fits(current + [line]):
            current.append(line)
            continue
        if current:
            chunks.append(current)
            current = []
        if fits([line]):
            current = [line]
            continue
        # One line longer than the whole budget: cut it into the longest pieces that fit.
        hard += 1
        rest = line
        while rest:
            lo, hi = 1, len(rest)
            while lo < hi:  # longest prefix of `rest` that fits
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if fits([rest[:mid]]) else (lo, mid - 1)
            chunks.append([rest[:lo]])
            rest = rest[lo:]
    if current:
        chunks.append(current)
    return [prefix + "\n".join(c) for c in chunks], hard


def split_symbols(docs, count_tokens, limit):
    """Chunk a list of documents. Returns (texts, owner, stats): texts[i] belongs
    to docs[owner[i]]; every document has at least one chunk."""
    texts, owner = [], []
    split = hard = max_chunks = 0
    for i, doc in enumerate(docs):
        chunks, h = split_document(doc, count_tokens, limit)
        texts.extend(chunks)
        owner.extend([i] * len(chunks))
        if len(chunks) > 1 or h:
            split += 1
        hard += h
        max_chunks = max(max_chunks, len(chunks))
    return texts, owner, Stats(symbols=len(docs), split=split, chunks=len(texts), max_chunks=max_chunks,
                               hard_split_lines=hard, limit=limit)


def describe(stats, repo, model):
    return (f"[chunk] {repo} / {model}: {stats.split} of {stats.symbols} symbols over {stats.limit} tokens, "
            f"split into {stats.chunks - (stats.symbols - stats.split)} chunks (max {stats.max_chunks} per symbol, "
            f"{stats.hard_split_lines} single lines cut); {stats.chunks} documents in total")


def max_per_symbol(chunk_scores, owner, n_symbols):
    """Per-symbol best chunk: chunk_scores is (..., n_chunks), result is (..., n_symbols).
    `owner` must be as split_symbols returns it: non-decreasing, every symbol present."""
    import numpy as np
    owner = np.asarray(owner)
    starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    if len(starts) != n_symbols or owner[0] != 0 or np.any(np.diff(owner) < 0) or owner[-1] != n_symbols - 1:
        raise ValueError("owner must list every symbol, in order")
    return np.maximum.reduceat(np.asarray(chunk_scores), starts, axis=-1)


def token_counter(tokenizer, prefix=""):
    """count_tokens for a Hugging Face tokenizer: prefix + text with special tokens."""
    def count(text):
        return len(tokenizer(prefix + text, add_special_tokens=True)["input_ids"])
    return count
