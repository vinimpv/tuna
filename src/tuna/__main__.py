"""tuna: tune code search for your repo.

  tuna clone   --repo saleor                          # shallow clone at the pinned commit
  tuna extract --repo saleor                          # tree-sitter symbols
  tuna bm25    --repo saleor --questions Q.jsonl
  tuna embed   --repo saleor --model M --tag T [--backend served|local]
  tuna dense   --repo saleor --questions Q.jsonl --vectors T1 [T2]
  tuna colbert --repo saleor --questions Q.jsonl      # LFM2.5-ColBERT-350M, MaxSim
  tuna rerank  --repo saleor --questions Q.jsonl --input RANKS --model R
  tuna eval    --repo saleor --questions Q.jsonl
  tuna questions --repo saleor                        # LLM-written test questions (pi + Qwen)
  tuna review    --repo saleor                        # markdown review sheet for 20 of them
  tuna issues    --repo saleor                        # test set from GitHub issues and their fix commits
  tuna mcp       --repo saleor --tag lfm25-ft-lora    # stdio MCP server: `search` over a saved index (CPU)
  tuna hint      --repo saleor --tag lfm25-ft-lora    # UserPromptSubmit hook: search first, inject a hint card

--repo is a config name under repos/ (or $TUNA_REPOS_DIR) or a path to a .toml.
Clones go to $TUNA_DATA/repos/<repo>, outputs to $TUNA_OUT/<repo>/
(defaults: ~/tuna-data/repos and ~/tuna-data/out).
"""
import argparse

from . import heldout, bm25, clone, colbert, common, dense, embed, evaluate, extract, hint, issues, mcp, questions, rerank, train


def main():
    ap = argparse.ArgumentParser(prog="tuna", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    adders = [m.add_parser for m in (clone, extract, bm25, embed, dense, colbert, rerank, evaluate, questions, issues,
                                     heldout, train, mcp, hint)]
    for add in adders + [questions.add_review_parser]:
        p = add(sub)
        p.add_argument("--repo", required=True, help="config name under repos/ or a .toml path")
        p.add_argument("--out", default=common.default_out(), help="output root (default $TUNA_OUT or ~/tuna-data/out)")
        p.add_argument("--allow-mismatch", action="store_true", help="extract: tolerate a clone off its pin")
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
