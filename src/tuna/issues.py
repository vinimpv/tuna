"""A test set from real GitHub issues: the issue text is the question, the
functions and methods its fix changed are the gold.

  tuna issues --repo saleor [--n 150] [--seed 20261111] [--name issues-v1]

Steps, each with its yield in the manifest:

 1. links    Fix commits are found offline in a full-history clone (blob-less,
             <data root>/history/<repo>.git, made on first use) among commits
             reachable from the pin: a closing keyword in the commit message
             (`Fixes #N`, `closes #N`, `resolves #N`, or the full issue URL).
             Squash-merged PRs whose body closes an issue are found through the
             search API (merged PRs with a linked issue) and mapped to their
             commit by the `(#PR)` subject suffix, `Merge pull request #PR` or
             `Backport of #PR`. "Fixes part of #N" and reverts do not count; a
             commit that closes several issues is dropped.
 2. gold     Each fix commit is diffed against its first parent (-U0). Changed
             lines in in-scope .py files (the config's include/exclude, so no
             tests or migrations) are mapped, on both sides of the diff, to the
             innermost function or method around them, with the same tree-sitter
             extractor and FQN rules as `tuna extract`. All commits of an issue
             are pooled. Kept: 1 to 3 changed functions or methods, at least
             one of which is still in symbols.jsonl at the pin (those are the
             gold); and no more changed out-of-scope code files than in-scope ones.
 3. issue    Title and body come from the search API (closed issues, swept by
             creation date, 100 per request, cached). The number must be a
             closed issue, not a pull request, and not opened by a bot.
 4. text     The body is cleaned (see clean_body) and the question is the title,
             a blank line, and the cleaned body. Dropped: non-English, fewer
             than --min-words words. Flagged, not dropped: the text names a gold
             symbol (`leak_name`) or its file (`leak_path`).
 5. sample   Up to --n items, in a seeded random order.

Outputs under <out>/<repo>/questions/: <name>.jsonl (all kept items),
<name>-noleak.jsonl (the unflagged subset), <name>.manifest.json and
<name>.rejections.jsonl (one row per dropped issue with the stage and reason).

GitHub access is read-only and unauthenticated unless --token-env names an
environment variable holding a token. Every response is cached under
<data root>/cache/github/ and never fetched twice.
"""
import collections
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys

from . import common, extract, github, langs, question_filter

CLOSING = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
FUNC_KINDS = {"Function", "Method"}
CODE_EXT = (".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".hbs", ".html", ".css", ".scss",
            ".rs", ".go", ".sql", ".vue", ".svelte")
NON_CODE_PATTERNS = ["*/tests/*", "*/test/*", "*/__tests__/*", "*/test_*", "*_test.*", "*.test.*", "*.spec.*",
                     "*/tests.py", "*/conftest.py", "*/migrations/*", "*/__snapshots__/*", "*/fixtures/*",
                     "docs/*", "*/docs/*"]
MAX_LOG_LINES = 20
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


# ---------------------------------------------------------------- 1. links

def closing_refs(text, owner, repo):
    """Issue numbers a commit or PR text closes, in order, without duplicates.

    Matches `Fixes #12`, `closes: #12, #13`, `Resolves https://github.com/o/r/issues/12`
    (case-insensitive). Skips `Fixes part of #12`, `partially fixes #12` and
    references to another repository (`fixes other/repo#12`)."""
    url = rf"https?://github\.com/{re.escape(owner)}/{re.escape(repo)}/issues/"
    ref = rf"(?:{url}|(?<![\w/])#)(\d+)\b"
    pat = re.compile(rf"(?<![\w-])(partially\s+|partly\s+)?{CLOSING}\b:?\s+((?:{ref})(?:\s*(?:,|and|&)\s*(?:{ref}))*)",
                     re.IGNORECASE)
    out = []
    for m in pat.finditer(text):
        if m.group(1):
            continue
        for n in re.findall(rf"(?:{url}|#)(\d+)", m.group(2), re.IGNORECASE):
            if int(n) not in out:
                out.append(int(n))
    return out


def pr_numbers(subject, body):
    """PR numbers a commit stands for: `Subject (#123)`, `Merge pull request #123`,
    and `Backport of #123` / `(cherry picked from ... #123)` in the body."""
    nums = []
    m = re.search(r"\(#(\d+)\)\s*$", subject) or re.match(r"Merge pull request #(\d+)", subject)
    if m:
        nums.append(int(m.group(1)))
    nums += [int(n) for n in re.findall(r"\bBackport(?:ed)? (?:of|from) #(\d+)", body, re.IGNORECASE)]
    return nums


def git(repo, *args, input=None, lazy=False):
    env = dict(os.environ)
    if not lazy:
        env["GIT_NO_LAZY_FETCH"] = "1"   # fail rather than fetch blobs one by one
    # attr.tree = the empty tree: a bare repo would read .gitattributes from HEAD, a blob we do not have.
    r = subprocess.run(["git", "-C", repo, "-c", f"attr.tree={EMPTY_TREE}", *args], input=input,
                       capture_output=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.decode(errors='replace')[:500]}")
    return r.stdout


def ensure_history(cfg, path, log):
    """A blob-less bare clone holding the pin; made on first use."""
    if not os.path.isdir(path):
        log(f"[issues] cloning full history (no blobs) into {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--bare", "--filter=blob:none", cfg["url"], path], check=True)
    r = subprocess.run(["git", "-C", path, "cat-file", "-e", cfg["commit"] + "^{commit}"])
    if r.returncode != 0:
        subprocess.run(["git", "-C", path, "fetch", "--quiet", "--filter=blob:none", "origin", cfg["commit"]], check=True)


def read_commits(hist, pin):
    raw = git(hist, "log", "--format=%H%x1f%P%x1f%ct%x1f%s%x1f%B%x1e", pin).decode("utf-8", "replace")
    out = []
    for rec in raw.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, parents, ct, subject, body = rec.split("\x1f", 4)
        out.append({"sha": sha, "parents": parents.split(), "time": int(ct), "subject": subject, "body": body})
    return out


# ---------------------------------------------------------------- 2. gold

def parse_unified_zero(diff_text):
    """`git diff -U0 -M` output -> {new_path: {"old": old_path, "plus": set, "minus": set}}
    with 1-based changed line numbers on the new (plus) and old (minus) side."""
    files, cur = {}, None
    old_path = None
    for line in diff_text.split("\n"):
        if line.startswith("diff --git "):
            cur, old_path = None, None
        elif line.startswith("--- "):
            old_path = None if line[4:] == "/dev/null" else line[6:]
        elif line.startswith("+++ "):
            new_path = None if line[4:] == "/dev/null" else line[6:]
            key = new_path or old_path
            cur = files.setdefault(key, {"old": old_path, "new": new_path, "plus": set(), "minus": set()})
        elif line.startswith("@@") and cur is not None:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            c, d = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
            cur["minus"].update(range(a, a + b))
            cur["plus"].update(range(c, c + d))
    return files


def innermost_functions(symbols, lines):
    """symbols: [(fqn, kind, start, end, ...)]; lines: 1-based line numbers.
    -> ({fqn: kind} of the innermost function/method around each line,
        [lines that lie in no function or method])."""
    funcs = sorted(((s[2], s[3], s[0], s[1]) for s in symbols if s[1] in FUNC_KINDS), key=lambda t: t[1] - t[0])
    hit, outside = {}, []
    for ln in sorted(lines):
        f = next((t for t in funcs if t[0] <= ln <= t[1]), None)
        if f:
            hit[f[2]] = f[3]
        else:
            outside.append(ln)
    return hit, outside


TRIVIAL = re.compile(r"^\s*(#.*|from\s+\S+\s+import\b.*|import\s+.*|\)|\(|)\s*$")


def changed_functions(spec, parser, old_path, old_src, new_path, new_src, minus, plus):
    """Functions/methods touched by one file's diff, on either side.
    -> (set of fqns, number of non-trivial changed lines outside any function)."""
    fqns, outside = set(), 0
    for path, src, lines in ((old_path, old_src, minus), (new_path, new_src, plus)):
        if path is None or src is None or not lines:
            continue
        _, syms = extract.extract_file(spec, parser, path, src)
        hit, out = innermost_functions(syms, lines)
        fqns.update(hit)
        text = src.decode("utf-8", "replace").split("\n")
        outside += sum(1 for ln in out if ln <= len(text) and not TRIVIAL.match(text[ln - 1]))
    return fqns, outside


def is_code(path):
    return path.endswith(CODE_EXT) and not common._match(path, NON_CODE_PATTERNS)


def tree_changes(hist, parent, sha):
    """[(status, old_path, new_path, old_oid, new_oid)] from diff-tree; needs no blobs."""
    out = git(hist, "diff-tree", "-r", "--no-renames", "-z", "--no-commit-id", parent, sha).decode("utf-8", "surrogateescape")
    parts, rows, i = out.split("\0"), [], 0
    while i < len(parts) - 1:
        meta = parts[i].split()
        if not meta:
            break
        status = meta[4]
        if status[0] in "RC":
            rows.append((status[0], parts[i + 1], parts[i + 2], meta[2], meta[3]))
            i += 3
        else:
            p = parts[i + 1]
            rows.append((status[0], p, p, meta[2], meta[3]))
            i += 2
    return rows


def prefetch_blobs(hist, oids, log):
    oids = sorted({o for o in oids if o and set(o) != {"0"}})
    if not oids:
        return
    have = set(git(hist, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").decode().split())
    missing = [o for o in oids if o not in have]
    log(f"[issues] blobs needed {len(oids)}, fetching {len(missing)}")
    for i in range(0, len(missing), 2000):
        subprocess.run(["git", "-C", hist, "-c", "fetch.negotiationAlgorithm=noop", "fetch", "--quiet", "--no-tags",
                        "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none", "--stdin", "origin"],
                       input="\n".join(missing[i:i + 2000]).encode(), check=True)


def commit_gold(hist, cfg, commit, spec, parser):
    """Changed functions/methods of one commit against its first parent, plus scope counts."""
    if not commit["parents"]:
        return None
    parent = commit["parents"][0]
    changes = tree_changes(hist, parent, commit["sha"])
    in_scope = [c for c in changes if c[2].endswith(".py") and (common.in_scope(c[2], cfg) or common.in_scope(c[1], cfg))]
    other_code = [c for c in changes if c not in in_scope and is_code(c[2])]
    res = {"in_scope_files": [c[2] for c in in_scope], "other_code_files": [c[2] for c in other_code],
           "fqns": set(), "outside_lines": 0}
    if not in_scope:
        return res
    paths = sorted({p for c in in_scope for p in (c[1], c[2])})
    diff = git(hist, "diff", "-U0", "-M", "--no-color", "--no-ext-diff", parent, commit["sha"], "--", *paths)
    parsed = parse_unified_zero(diff.decode("utf-8", "replace"))
    oids = {(c[1], c[2]): (c[3], c[4]) for c in in_scope}
    for key, f in parsed.items():
        old_oid, new_oid = next((v for (o, n), v in oids.items() if n == key or o == key), (None, None))
        old_src = blob(hist, old_oid) if f["old"] else None
        new_src = blob(hist, new_oid) if f["new"] else None
        fq, outside = changed_functions(spec, parser, f["old"], old_src, f["new"], new_src, f["minus"], f["plus"])
        res["fqns"] |= fq
        res["outside_lines"] += outside
    return res


def blob(hist, oid):
    if not oid or set(oid) == {"0"}:
        return None
    return git(hist, "cat-file", "blob", oid)


# ---------------------------------------------------------------- 4. text

TEMPLATE_HEADING = re.compile(
    r"^\s*(\*\*|__)?\s*(what (are you trying to achieve|did you expect( to happen)?|happened|actually happened|"
    r"is the (current|expected) behaviou?r)|steps? to reproduce( the problem)?|expected (behaviou?r|result)|"
    r"actual (behaviou?r|result)|current behaviou?r|(bug|issue|problem) description|describe the (bug|issue|problem)|"
    r"how to reproduce|reproduction( steps)?|to reproduce|additional (context|information|details)|"
    r"context|description|summary|details|possible (solution|fix)|proposed (solution|fix)|suggested fix|notes?|"
    r"what i['’]?m trying to achieve|what i expected to happen|what happened instead(/how it failed)?|"
    r"describe a proposed solution|other solutions i['’]?ve tried( and won['’]?t work)?|feature request|problem|"
    r"is your feature request related to a problem\??( please describe\.?)?|describe the solution you['’]?d like|"
    r"describe alternatives you['’]?ve considered|in what situation are you experiencing subpar performance\??)"
    r"\s*[:?]?\s*(\*\*|__)?\s*$",
    re.IGNORECASE)
DROP_SECTION = re.compile(
    r"^\s*(\*\*|__)?\s*(environment|system information|system info|versions?|debug info(rmation)?|logs?|"
    r"error logs?|screenshots?( or videos?| or mockups| and logs)?|screen ?recordings?|videos?|attachments?|"
    r"desktop \(please complete the following information\):?|(zulip )?server and web app version:?|"
    r"(your )?(saleor|zulip|posthog) (version|details)|browser( information)?|setup|configuration)\s*[:?]?\s*(\*\*|__)?\s*$",
    re.IGNORECASE)
ENV_LINE = re.compile(
    r"^\s*([-*]\s*)?(\*\*|__)?\s*(saleor|zulip( server)?|posthog|python|django|node|postgres(ql)?|redis|celery|"
    r"os|operating system|browser|device|platform|deployment|version|version/commit|commit)( and)?( version)?"
    r"\s*(\*\*|__)?\s*:", re.IGNORECASE)
THANKS = re.compile(r"thank you.{0,40}(report|request|contribut|squash)", re.IGNORECASE)
NO_ANSWER = re.compile(r"^\s*[-*]?\s*(_?no response_?|n/?a|none|-+|\.\.\.)\s*\.?\s*$", re.IGNORECASE)
CHECKBOX = re.compile(r"^\s*[-*+]\s*\[[ xX]\]")
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>|<video\b.*?</video>|<picture\b.*?</picture>",
                   re.IGNORECASE | re.DOTALL)
ATTACHMENT_URL = re.compile(r"https?://(user-images\.githubusercontent\.com|private-user-images\.githubusercontent\.com|"
                            r"github\.com/user-attachments|github\.com/[^/\s]+/[^/\s]+/assets)/\S+", re.IGNORECASE)
TAG = re.compile(r"</?(br|p|sub|sup|kbd|b|i|em|strong|summary|details|div|span|hr|center|code|pre)\b[^>]*>", re.IGNORECASE)
LOGLIKE = re.compile(r"^(\s{4,}|\t|\s*(Traceback|File \"|at |\[?\d{4}-\d\d-\d\d|\d\d:\d\d:\d\d|[A-Z]\w*(Error|Exception)\b|"
                     r"(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b))")


def _drop_long_runs(lines, pred, limit):
    out, run = [], []
    for l in lines + [None]:
        if l is not None and pred(l):
            run.append(l)
            continue
        if run:
            if len(run) <= limit:
                out += run
            run = []
        if l is not None:
            out.append(l)
    return out


def clean_body(body):
    """Issue body -> question text: no HTML comments, images, attachments,
    checkboxes, template headings, environment/log/screenshot sections,
    environment key: value lines, code blocks or logs longer than MAX_LOG_LINES
    lines (short ones stay), or thank-you boilerplate."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?(-->|$)", "", text, flags=re.DOTALL)
    # <details> blocks: keep short ones, without the tags
    def details(m):
        inner = m.group(1)
        return "" if inner.count("\n") > MAX_LOG_LINES else inner
    text = re.sub(r"<details\b[^>]*>(.*?)</details>", details, text, flags=re.DOTALL | re.IGNORECASE)
    text = IMAGE.sub("", text)
    text = ATTACHMENT_URL.sub("", text)
    text = TAG.sub(" ", text)
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):  # fenced code blocks
        m = re.match(r"^\s*(`{3,}|~{3,})", lines[i])
        if m:
            fence = m.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(fence):
                j += 1
            if j - i - 1 <= MAX_LOG_LINES:
                out += lines[i:j + 1]
            i = j + 1
            continue
        out.append(lines[i])
        i += 1
    lines, out, dropping = out, [], False
    for l in lines:  # sections and boilerplate lines
        heading = re.match(r"^\s*#{1,6}\s*(.*?)\s*#*\s*$", l)
        title = heading.group(1) if heading else (l if re.fullmatch(r"\s*(\*\*|__)[^*_]+(\*\*|__)\s*:?\s*", l) else None)
        if title is not None:
            if DROP_SECTION.match(title):
                dropping = True
                continue
            dropping = False
            if TEMPLATE_HEADING.match(title) or THANKS.search(title):
                continue
            if heading:
                out.append(title)
                continue
        if dropping:
            continue
        if CHECKBOX.match(l) or ENV_LINE.match(l) or THANKS.search(l) or NO_ANSWER.match(l):
            continue
        out.append(l.rstrip())
    out = _drop_long_runs(out, lambda l: bool(l.strip()) and bool(LOGLIKE.match(l)), MAX_LOG_LINES)
    text = "\n".join(out)
    text = re.sub(r"\n[ \t]*\n(\s*\n)+", "\n\n", text)
    return text.strip()


def prose(text):
    """Text without fenced code and URLs, for word counts and language checks."""
    text = re.sub(r"(`{3,}|~{3,}).*?(\1|$)", " ", text, flags=re.DOTALL)
    return re.sub(r"https?://\S+", " ", text)


def word_count(text):
    return len(re.findall(r"[^\W\d_][\w'’-]*", prose(text)))


def is_english(text):
    """False when over 20% of letters are outside Latin script, or under half
    of the Latin words (2+ letters, no digits) are English or allowed terms."""
    p = prose(text)
    letters = [c for c in p if c.isalpha()]
    if not letters:
        return False
    if sum(1 for c in letters if not ("a" <= c.lower() <= "z")) / len(letters) > 0.2:
        return False
    toks = [t.lower() for t in re.findall(r"[A-Za-z]{2,}", p)]
    toks = [t for t in toks if not any(ch.isupper() for ch in t)]
    if len(toks) < 8:
        return True
    eng, allowed = question_filter.english_words(), question_filter.allowed_terms()
    return sum(1 for t in toks if t in eng or t in allowed) / len(toks) >= 0.5


def _norm(s):
    return s.replace("_", "").lower()


def leaks(text, gold):
    """(names, paths): gold symbol names and file paths that the text gives away.

    A name with two or more words (`get_user`, `getUser`, `UserProfile`) leaks
    when it appears as a token, ignoring case and underscores. A one-word name
    (`save`) leaks only as code: `save(`, `.save` or `` `save` ``. A path leaks
    as the full path, any tail of two or more segments, the dotted module (two
    or more trailing parts), or the file name (`users.py`, except `__init__.py`)."""
    names, paths = [], []
    tokens = {_norm(t) for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)}
    low = text.lower()
    for fqn, path in gold:
        name = fqn.rsplit(".", 1)[-1]
        parts = [p for p in name.split("_") for p in langs_split(p)]
        if len([p for p in parts if p]) >= 2:
            if _norm(name) in tokens:
                names.append(name)
        elif re.search(rf"(\.{re.escape(name)}\b|\b{re.escape(name)}\(|`{re.escape(name)}`)", text):
            names.append(name)
        segs = path.split("/")
        cands = ["/".join(segs[i:]) for i in range(len(segs) - 1)]
        mod = langs.python_module_name(path).split(".")
        if mod[-1] == "__init__":
            mod = mod[:-1]
        cands += [".".join(mod[i:]) for i in range(len(mod) - 1)]
        if segs[-1] != "__init__.py":
            cands.append(segs[-1])
        for c in cands:
            if re.search(rf"(?<![\w/.-]){re.escape(c.lower())}(?![\w-])", low):
                paths.append(c)
                break
    return sorted(set(names)), sorted(set(paths))


def langs_split(word):
    return question_filter.CAMEL.findall(word) if word else []


# ---------------------------------------------------------------- run

def owner_repo(cfg):
    m = re.match(r"https?://github\.com/([^/]+)/([^/.]+)", cfg["url"])
    if not m:
        sys.exit(f"{cfg['_file']}: url is not a github.com repository")
    return m.group(1), m.group(2)


def main(args):
    import tree_sitter
    log = lambda m: print(m, flush=True)  # noqa: E731
    cfg = common.load_config(args.repo)
    owner, repo = owner_repo(cfg)
    work = common.workdir(args.out, cfg)
    pin_syms = {s["fqn"]: s for s in common.load_symbols(work)}
    hist = args.history or os.path.join(common.data_root(), "history", f"{cfg['name']}.git")
    ensure_history(cfg, hist, log)
    token = os.environ.get(args.token_env) if args.token_env else None
    if args.token_env and not token:
        sys.exit(f"${args.token_env} is empty")
    gh = github.GitHub(args.cache, token=token, max_wait=args.max_wait, log=log)
    spec = langs.PYTHON
    parser = tree_sitter.Parser(spec.language())
    stages, rejections = collections.OrderedDict(), []

    def reject(issue, stage, reason, **extra):
        rejections.append({"issue": issue, "stage": stage, "reason": reason, **extra})

    # 1. links
    commits = read_commits(hist, cfg["commit"])
    by_sha = {c["sha"]: c for c in commits}
    pin_time = by_sha[cfg["commit"]]["time"]
    pin_day = datetime.datetime.fromtimestamp(pin_time, datetime.timezone.utc).strftime("%Y-%m-%d")
    links = collections.defaultdict(lambda: {"commits": set(), "sources": set(), "prs": set()})
    commit_issues = collections.defaultdict(set)
    reverts = 0
    for c in commits:
        if c["subject"].lower().startswith("revert"):
            reverts += 1
            continue
        for n in closing_refs(c["body"], owner, repo):
            links[n]["commits"].add(c["sha"])
            links[n]["sources"].add("commit_message")
            commit_issues[c["sha"]].add(n)
    pr_commit = {}
    for c in commits:
        for pr in pr_numbers(c["subject"], c["body"]):
            pr_commit.setdefault(pr, c["sha"])
    prs = gh.search_issues(f"repo:{owner}/{repo} is:pr is:merged linked:issue", created=("2008-01-01", pin_day)) \
        if not args.no_pr_search else []
    pr_stats = collections.Counter()
    for pr in prs:
        refs = closing_refs(f"{pr['title']}\n{pr.get('body') or ''}", owner, repo)
        if not refs:
            pr_stats["no_closing_keyword_in_body"] += 1
            continue
        sha = pr_commit.get(pr["number"])
        if sha is None:
            pr_stats["no_commit_at_pin"] += 1
            continue
        pr_stats["mapped"] += 1
        for n in refs:
            links[n]["commits"].add(sha)
            links[n]["sources"].add("pr_body")
            links[n]["prs"].add(pr["number"])
            commit_issues[sha].add(n)
    stages["issues_linked"] = len(links)
    multi = {sha for sha, ns in commit_issues.items() if len(ns) > 1}
    for n in sorted(links):
        if links[n]["commits"] & multi:
            reject(n, "links", "commit_closes_several_issues")
    cands = {n: v for n, v in links.items() if not (v["commits"] & multi)}
    stages["single_issue_commits"] = len(cands)
    log(f"[issues] {cfg['name']}: {len(commits)} commits to the pin, {len(links)} issues linked "
        f"({len(prs)} merged PRs with a linked issue searched: {dict(pr_stats)}), {len(cands)} after "
        f"dropping commits that close several issues")

    # 2. gold
    need = []
    for v in cands.values():
        for sha in v["commits"]:
            c = by_sha[sha]
            if c["parents"]:
                for ch in tree_changes(hist, c["parents"][0], sha):
                    if ch[2].endswith(".py") and (common.in_scope(ch[2], cfg) or common.in_scope(ch[1], cfg)):
                        need += [ch[3], ch[4]]
    prefetch_blobs(hist, need, log)
    gold_cache, items = {}, []
    reasons = collections.Counter()
    for n in sorted(cands):
        v = cands[n]
        fqns, outside, in_files, other = set(), 0, set(), set()
        for sha in sorted(v["commits"]):
            if sha not in gold_cache:
                gold_cache[sha] = commit_gold(hist, cfg, by_sha[sha], spec, parser)
            g = gold_cache[sha]
            if g is None:
                continue
            fqns |= g["fqns"]
            outside += g["outside_lines"]
            in_files |= set(g["in_scope_files"])
            other |= set(g["other_code_files"])
        reason = None
        if not in_files:
            reason = "no_in_scope_change"
        elif len(other) > len(in_files):
            reason = "mostly_out_of_scope"
        elif not fqns:
            reason = "no_function_changed"
        elif len(fqns) > args.max_gold:
            reason = "too_many_functions"
        else:
            gold = sorted(f for f in fqns if f in pin_syms)
            if not gold:
                reason = "gold_gone_at_pin"
        if reason:
            reasons[reason] += 1
            reject(n, "gold", reason, changed=sorted(fqns)[:20])
            continue
        items.append({"issue_number": n, "commits": sorted(v["commits"], key=lambda s: by_sha[s]["time"]),
                      "sources": sorted(v["sources"]), "prs": sorted(v["prs"]), "changed": sorted(fqns),
                      "gold": gold, "outside_lines": outside, "in_scope_files": sorted(in_files),
                      "other_code_files": sorted(other)})
    for r in ("no_in_scope_change", "mostly_out_of_scope", "no_function_changed", "too_many_functions",
              "gold_gone_at_pin"):
        stages[f"after_{r}"] = stages.get(list(stages)[-1]) - reasons[r] if stages else 0
    stages["with_gold"] = len(items)
    log(f"[issues] gold: {len(items)} issues with 1-{args.max_gold} changed functions, some at the pin; "
        f"dropped {dict(reasons)}")

    # 3. issue text
    lo = "2008-01-01"
    issues = {}
    try:
        for it in gh.search_issues(f"repo:{owner}/{repo} is:issue is:closed", created=(lo, pin_day)):
            issues[it["number"]] = it
    except github.RateLimited as e:
        sys.exit(f"[issues] stopped: {e}; {gh.requests} requests made, {len(issues)} issues read. "
                 "Rerun later: cached responses are reused.")
    kept = []
    reasons = collections.Counter()
    for it in items:
        iss = issues.get(it["issue_number"])
        reason = None
        if iss is None:
            reason = "not_a_closed_issue"   # a PR, still open, deleted, transferred
        elif (iss.get("user") or {}).get("type") == "Bot" or (iss.get("user") or {}).get("login", "").endswith("[bot]"):
            reason = "bot_author"
        if reason:
            reasons[reason] += 1
            reject(it["issue_number"], "issue", reason)
            continue
        it["issue"] = iss
        kept.append(it)
    stages["closed_issue_by_human"] = len(kept)
    log(f"[issues] issue text: {len(kept)} closed human-written issues; dropped {dict(reasons)}")

    # 4. clean and filter
    final = []
    reasons = collections.Counter()
    for it in kept:
        iss = it["issue"]
        body = clean_body(iss.get("body"))
        q = iss["title"].strip() + ("\n\n" + body if body else "")
        wc = word_count(q)
        reason = None
        if not is_english(q):
            reason = "non_english"
        elif wc < args.min_words:
            reason = "too_short"
        if reason:
            reasons[reason] += 1
            reject(it["issue_number"], "text", reason, words=wc, title=iss["title"])
            continue
        gold_paths = [(f, pin_syms[f]["file_path"]) for f in it["gold"]]
        names, paths = leaks(q, gold_paths)
        first = by_sha[it["commits"][0]]
        final.append({
            "id": None, "question": q, "gold_fqn": it["gold"][0], "gold_fqns": it["gold"],
            "files": sorted({pin_syms[f]["file_path"] for f in it["gold"]}),
            "kinds": [pin_syms[f]["kind"] for f in it["gold"]],
            "leak": bool(names or paths), "leak_name": names, "leak_path": paths,
            "issue_url": iss["html_url"], "issue_number": iss["number"], "issue_title": iss["title"],
            "issue_created": iss["created_at"], "issue_closed": iss.get("closed_at"),
            "issue_author": (iss.get("user") or {}).get("login"),
            "fix_commits": it["commits"], "fix_commit": it["commits"][-1], "fix_prs": it["prs"],
            "fix_date": datetime.datetime.fromtimestamp(first["time"], datetime.timezone.utc).isoformat(),
            "link_source": it["sources"], "changed_functions": it["changed"],
            "changed_not_at_pin": sorted(set(it["changed"]) - set(it["gold"])),
            "changed_lines_outside_functions": it["outside_lines"], "other_code_files": it["other_code_files"],
            "words": wc, "body_sha256": hashlib.sha256((iss.get("body") or "").encode()).hexdigest()})
    stages["english"] = len(kept) - reasons["non_english"]
    stages[f"min_{args.min_words}_words"] = len(final)
    log(f"[issues] text: {len(final)} kept; dropped {dict(reasons)}; "
        f"{sum(r['leak'] for r in final)} flagged as naming a gold symbol or its file")

    # 5. sample
    order = sorted(final, key=lambda r: r["issue_number"])
    random.Random(args.seed).shuffle(order)
    out = order[:args.n]
    for i, r in enumerate(out, 1):
        r["id"] = f"{cfg['name']}-{args.name}-{i:03d}"
    stages["sampled"] = len(out)
    qdir = os.path.join(work, "questions")
    base = os.path.join(qdir, args.name)
    common.write_jsonl(base + ".jsonl", out)
    common.write_jsonl(base + "-noleak.jsonl", [r for r in out if not r["leak"]])
    common.write_jsonl(base + ".rejections.jsonl", rejections)
    manifest = {
        "repo": cfg["name"], "name": args.name, "kind": "github-issues", "commit": cfg["commit"],
        "tag": cfg.get("tag"), "github": f"{owner}/{repo}", "seed": args.seed, "n": args.n,
        "max_gold": args.max_gold, "min_words": args.min_words, "max_log_lines": MAX_LOG_LINES,
        "stages": stages, "pr_search": dict(pr_stats, searched=len(prs)),
        "commits_to_pin": len(commits), "reverts_skipped": reverts,
        "accepted": len(out), "leak_flagged": sum(r["leak"] for r in out),
        "noleak": sum(not r["leak"] for r in out),
        "gold_per_item": dict(collections.Counter(len(r["gold_fqns"]) for r in out)),
        "link_source": dict(collections.Counter("+".join(r["link_source"]) for r in out)),
        "github_requests_this_run": gh.requests, "github_responses_used": gh.cache_hits + sum(gh.requests.values()),
        "symbols_sha256": common.file_sha256(os.path.join(work, "symbols.jsonl")),
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(base + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"[issues] {cfg['name']}: stages {json.dumps(stages)}")
    log(f"[issues] wrote {base}.jsonl ({len(out)}; {manifest['noleak']} without a leak flag); "
        f"GitHub requests {gh.requests}, cache hits {gh.cache_hits}")


def add_parser(sub):
    p = sub.add_parser("issues", help="test set from GitHub issues and the commits that fixed them")
    p.add_argument("--n", type=int, default=150, help="most items to keep")
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--name", default="issues-v1", help="output name under questions/")
    p.add_argument("--max-gold", type=int, default=3, help="most changed functions/methods per fix")
    p.add_argument("--min-words", type=int, default=15)
    p.add_argument("--history", help="full-history clone (default <data root>/history/<repo>.git)")
    p.add_argument("--cache", default=os.path.join(common.data_root(), "cache", "github"))
    p.add_argument("--token-env", help="name of an environment variable holding a read-only GitHub token "
                                       "(default: unauthenticated)")
    p.add_argument("--max-wait", type=int, default=900, help="longest sleep for a rate-limit reset, seconds")
    p.add_argument("--no-pr-search", action="store_true", help="only use commit messages to find fixes")
    p.set_defaults(func=main)
    return p
