"""Extract symbols from one repo into <out>/<repo>/symbols.jsonl with tree-sitter.

Symbols are what the language spec (langs.py) marks as such; for Python that is
every class and function at any depth: module-level functions and classes,
methods, and nested functions and classes. Each row:

  fqn         dotted module path + enclosing symbols + name,
              e.g. saleor.checkout.tasks.delete_expired_checkouts
  kind        Class | Function | Method (a function defined directly in a class body)
  file_path   repo-relative, `/`-separated
  line_start  1-based, first decorator line included
  line_end    1-based, inclusive
  body        the symbol document, see symbol_text()
  body_hash   sha256 of `body`

Files come from `git ls-files` (tracked files only) filtered by the config's
include/exclude globs. When two symbols share an FQN (a conditional
redefinition, a property setter) the first one in the file wins.
"""
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

from . import common, langs


def symbol_text(fqn, path, start, end, source):
    """Header plus source: two comment lines naming the symbol, then its
    decorators, signature, docstring and body exactly as written, dedented to
    column 0. No call graph or other derived context."""
    return f"# FQN: {fqn}\n# File: {path}:{start}-{end}\n{source}"


def list_files(root):
    if os.path.isdir(os.path.join(root, ".git")):
        out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True).stdout
        return sorted(p for p in out.decode("utf-8", "surrogateescape").split("\0") if p)
    files = []
    for d, dirs, names in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        files += [os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/") for n in names]
    return sorted(files)


def _source_lines(lines, start_row, end_row, indent):
    out = []
    for line in lines[start_row:end_row + 1]:
        out.append(line[indent:] if line[:indent].isspace() else line)
    return "\n".join(out).rstrip()


def extract_file(spec, parser, path, data):
    """Parse one file: (has_syntax_errors, [(fqn, kind, line_start, line_end, text), ...])."""
    tree = parser.parse(data)
    return tree.root_node.has_error, list(_symbols(spec, tree, path, data))


def _symbols(spec, tree, path, data):
    lines = data.decode("utf-8", "replace").split("\n")
    module = spec.module_name(path)
    # explicit stack, pre-order: (node, scope fqn, scope node type, wrapper node)
    stack = [(c, module, None, None) for c in reversed(tree.root_node.children)]
    while stack:
        node, scope, scope_type, wrapper = stack.pop()
        if node.type in spec.wrappers:
            stack.extend((c, scope, scope_type, node) for c in reversed(node.children) if c.type in spec.symbols)
            continue
        if node.type in spec.symbols:
            name_node = node.child_by_field_name(spec.name_field)
            if name_node is None:  # broken syntax
                continue
            kind = spec.symbols[node.type]
            if kind == "Function" and scope_type in spec.member_scopes:
                kind = "Method"
            fqn = f"{scope}.{name_node.text.decode('utf-8', 'replace')}"
            span = wrapper or node
            start, end = span.start_point.row, span.end_point.row
            source = _source_lines(lines, start, end, span.start_point.column)
            yield fqn, kind, start + 1, end + 1, symbol_text(fqn, path, start + 1, end + 1, source)
            stack.extend((c, fqn, node.type, None) for c in reversed(node.children))
            continue
        stack.extend((c, scope, scope_type, None) for c in reversed(node.children))


def grammar_versions(names):
    out = {"tree-sitter": importlib.metadata.version("tree-sitter")}
    for n in names:
        pkg = langs.SPECS[n].grammar_module.replace("_", "-")
        out[pkg] = importlib.metadata.version(pkg)
    return out


def main(args):
    import tree_sitter
    cfg = common.load_config(args.repo)
    head = common.check_commit(cfg, args.allow_mismatch)
    work = common.workdir(args.out, cfg)
    enabled = cfg.get("languages", ["python"])
    unknown = [n for n in enabled if n not in langs.SPECS]
    if unknown:
        sys.exit(f"{cfg['_file']}: unsupported languages {unknown}; known: {sorted(langs.SPECS)}")
    parsers = {n: tree_sitter.Parser(langs.SPECS[n].language()) for n in enabled}

    seen, rows = set(), []
    dropped = collections.Counter()
    files, parse_errors = 0, []
    for path in list_files(cfg["path"]):
        spec = langs.spec_for_path(path, enabled)
        if spec is None or not common.in_scope(path, cfg):
            continue
        with open(os.path.join(cfg["path"], path), "rb") as fh:
            data = fh.read()
        files += 1
        has_error, symbols = extract_file(spec, parsers[spec.name], path, data)
        if has_error:
            parse_errors.append(path)
        if not symbols:
            dropped["files_without_symbols"] += 1
        for fqn, kind, start, end, text in symbols:
            if fqn in seen:
                dropped["duplicate_fqn"] += 1
                continue
            seen.add(fqn)
            rows.append({"fqn": fqn, "kind": kind, "file_path": path, "line_start": start, "line_end": end,
                         "body": text, "body_hash": hashlib.sha256(text.encode()).hexdigest()})

    out = os.path.join(work, "symbols.jsonl")
    common.write_jsonl(out, rows)
    kinds = collections.Counter(r["kind"] for r in rows)
    manifest = {
        "repo": cfg["name"], "tag": cfg.get("tag"), "commit": head, "config": os.path.basename(cfg["_file"]),
        "include": cfg["include"], "exclude": cfg["exclude"], "languages": enabled,
        "extractor": "tuna tree-sitter", "grammars": grammar_versions(enabled),
        "symbols": len(rows), "files": files, "files_with_symbols": len({r["file_path"] for r in rows}),
        "kinds": dict(kinds), "dropped": dict(dropped), "parse_error_files": parse_errors,
        "symbols_sha256": common.file_sha256(out),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(work, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({k: manifest[k] for k in ("symbols", "files", "kinds", "dropped")}
                     | {"parse_error_files": len(parse_errors)}), flush=True)
    print(f"[extract] wrote {out}", flush=True)


def add_parser(sub):
    p = sub.add_parser("extract", help="symbols and their source documents via tree-sitter")
    p.set_defaults(func=main)
    return p


if __name__ == "__main__":
    sys.exit("run as: tuna extract --repo <name>")
