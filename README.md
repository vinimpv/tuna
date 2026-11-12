<img src="assets/icon.svg" width="64" height="64" alt="">

# tuna

Tune code search for your repo.

tuna measures how well an embedding model finds the right function or class
in *your* codebase, given the kind of question a developer asks ("where do we
decide which warehouse an order ships from?"). It is also the harness for
tuning a small embedder on questions that an LLM writes about the code, and for
showing whether that helped.

This repository is the code and the data behind the two-page summary
[`summary.pdf`](summary.pdf), "Tuning Small Embedders for Code Search with
LLM-Written Questions: How Much Is Repository-Specific?". A full paper is in preparation.

## Results in brief

| Model | Saleor | Zulip | PostHog | Mean | Issues |
|---|--:|--:|--:|--:|--:|
| BM25 | .265 | .369 | .439 | .357 | .279 |
| LFM2.5-Embedding-350M, stock | .247 | .301 | .328 | .292 | .207 |
| LFM2.5-Embedding-350M, tuned | .627 | .620 | .538 | .595 | .322 |
| Jina code 0.5B, stock | .510 | .546 | .523 | .527 | .349 |
| Jina code 0.5B, tuned | .735 | .731 | .691 | .719 | .367 |
| Qwen3-Embedding-0.6B, stock | .595 | .619 | .542 | .585 | .350 |
| Qwen3-Embedding-0.6B, tuned | .784 | .798 | .735 | .772 | .356 |
| Jina code 1.5B, stock | .617 | .658 | .570 | .615 | .349 |
| LFM2.5-ColBERT-350M, stock | .390 | .494 | .541 | .475 | .304 |

MRR@10 of the gold symbol. Saleor, Zulip, PostHog and Mean: Claude-written questions on held-out files, 400 per repository. Issues: 321 real issues, pooled.

- Tuning raises MRR@10 for every model on every repository, by 0.19 to 0.30 pooled over the three.
- Most of that gain is not specific to the repository. A model tuned on a different repository gets 79% to 89% of it, and tuning on the target adds 0.02 to 0.05.
- LFM2.5-Embedding-350M, the smallest model, starts below BM25 and ends near stock embedders of 0.6B and 1.5B parameters.
- On 321 real GitHub issues only the smallest model gains.
- A coding agent that was given search hits from the tuned index, instead of the stock one, found the right file no more often.

These are preliminary results of 12 November 2026. The summary has the intervals, the set-up and the limits.

## What is in this repository

| path | what |
|---|---|
| `src/tuna/`, `scripts/`, `tests/` | the harness: extraction, question writing, BM25, embedding, tuning, scoring, the agent experiments, and the scripts that ran every step |
| `repos/` | configs for the five benchmark repositories, each pinned to one commit (none of their code) |
| `data/out/` | the released results, in the layout the harness writes (set `TUNA_OUT` to it) |
| `data/metrics/ledger.jsonl` | the run ledger: one line per step of every run in this release, with its settings, counts, timings and machine, and notes and rules written along the way |
| `data/REDACTIONS.txt` | what was replaced inside the released data, with counts |
| `summary.pdf` | the two-page summary |

What `data/out/` holds:

| path | what |
|---|---|
| `<repo>/questions/` | each question set of this release: `<set>.jsonl`, its `<set>.manifest.json` (prompt and prompt hash, writer model, seed, filter, guards, yield), training/validation splits, rejected drafts with reasons, and `heldout-files.txt` |
| `<repo>/eval_<set>.json` | the scored rows of this release on each test set, with each question's gold rank (`per_question_gold_rank`) |
| `<repo>/models/<row>/tuna-model.json` | each tuned row's settings and training record (per-epoch validation, chosen epoch, checkpoint size); no weights |
| `<repo>/search-bench_<row>_<device>.json` | query time and index size |
| `<repo>/cc-bench/<experiment>/` | the agent experiments: issues, and one result row per run (`runs.jsonl`) |
| `bootstrap/`, `tables/`, `cc-bench/*.json`, `judge/`, `judge-train/`, `figures/` | intervals, summary tables, agent reports, the writer check, and the cost figure's data |
| `bootstrap/extras-saleor.json`, `extras-pooled.json`, `baseline-pairs.json` | the intervals of the summary's extra rows (seeds, question-count curve, Qwen3-Embedding-4B, re-mining, bug-style and pooled rows) and of the baseline pairs; written by `scripts/m2-intervals.sh final` |
| `tables/seed-variance.json`, `tables/writer-gap.json` | seed spread of the tuned rows, and the claude-m3 against qwen-m4-400 writer gap; written by `scripts/seed-variance.py` and `scripts/writer-gap.py` |
| `cc-bench/full.plan.json` | the first agent plan (`cc-bench/full`), stopped after 30 runs on Saleor and replaced by `issues`, which reuses 18 of its runs |

The hint-card benches (`hint-pilot`, `hint-full`) reuse the issues question file.
Their runs recorded the hint hook's failure notice in its old wording ("Hint lane
failed"); the code now says "Hint search failed".

Names used in the data:

| name | meaning |
|---|---|
| `m1`, `m2`, `m3`, `m3-new` | milestones of the question sets: `qwen-m1`, `qwen-m2`, `claude-m2` (the first 100 Claude questions per repository), `claude-m3` (400: `claude-m2` plus `claude-m3-new`, 300 more) |
| `m4-400` | `qwen-m4-400`: Qwen writes one question for each of `claude-m3`'s 400 golds |
| `t400` | the 400-question test sets of the test-only repositories (`claude-t400`) and the LFM3-written set (`lfm3-t400`) |
| `bug1`, `trainbug1` | `qwen-bug1`: training questions written as bug reports; `qwen-trainbug1`: `qwen-train1` and `qwen-bug1` merged |
| `lora-bug` | the LFM2.5 LoRA row trained on `qwen-trainbug1` |
| `.rerun` | a same-seed rerun of a row on another machine (for example `lfm25-ft-lora.rerun`) |
| `.from-<repo>` | a row tuned on another repository, scored here |
| `pooled` | one model tuned on all three repositories' questions |
| `pi` | the pi coding agent, the question writer's harness (`scripts/pi-ask`) |

**Not included.** The tuned weights and LoRA adapters are not in this
repository. They will follow later, each under its base model's licence.
Also left out, because the code rebuilds them: the benchmark repositories'
code and `symbols.jsonl` (`tuna clone`, `tuna extract`). The agent attempt
logs (`questions/logs/`), event logs, vectors, ranks and the GPU power log are
not shipped.

**Redactions.** Before release, home-directory paths, local network addresses
and host names were removed or rewritten in the data; paths are relative to
`$TUNA_OUT` or `$TUNA_DATA`. `data/REDACTIONS.txt` lists every kind with its
count. Machine labels: `local` is our machine (one RTX 5090), `rented-N` are
rented single-GPU machines. Commit labels `dev-NN` in the data name commits of
the unpublished development history; this repository is a single release
commit.

## Check the published numbers without a GPU

The intervals come from the per-question ranks in `data/out`. To recompute
them you need the symbols of each repository, which `tuna extract` rebuilds
from the pinned commit on a CPU:

```sh
git clone https://github.com/vinimpv/tuna && cd tuna
uv sync --locked                                   # see Install
export TUNA_OUT=$PWD/data/out TUNA_DATA=$HOME/tuna-data
for r in saleor zulip posthog; do .venv/bin/tuna clone --repo $r && .venv/bin/tuna extract --repo $r; done
python3 scripts/paired-bootstrap.py --sets claude-m3 --pool-repos --json /tmp/claude-m3.json
```

This writes the same 101 MRR@10 differences and 95% intervals as
`data/out/bootstrap/final-mrr10-claude-m3.json` (10,000 resamples, seed 0,
resampled by gold file within each repository). The bootstrap itself takes
about a minute. Before it, `uv sync` installs the full environment, which
includes a CUDA build of torch (3.1 GB of downloads), and the three
repositories are cloned and extracted. None of it needs a GPU. `tuna extract`
writes each repository's `symbols.jsonl` into `data/out` (ignored by git) and
rewrites its `manifest.json`, in which only the `created` time changes: the
`symbols_sha256` it writes should equal the released one.
Other test sets: `--sets qwen-m2 claude-m2 qwen-m4-400 issues-v1 ...`.

## Install

tuna uses [uv](https://docs.astral.sh/uv/) and Python 3.12, on Linux x86_64.
It is not published on PyPI: the PyPI package named `tuna` is an unrelated
project, so install from this repository as below, not with `pip install`.

```sh
git clone https://github.com/vinimpv/tuna
cd tuna
uv sync --locked          # creates .venv with the exact versions in uv.lock
.venv/bin/tuna --help
```

Supply-chain policy, enforced in `pyproject.toml`:

- `exclude-newer` keeps every dependency at least 30 days older than the last
  lock refresh; `uv.lock` pins the full transitive set with hashes.
- `no-build = true`: wheels only, so no package's `setup.py` or build backend
  ever runs. (tuna itself is built by uv's built-in `uv_build` backend.)
- Direct dependencies are pinned exactly, each with its upload date.

The default torch wheel is a CUDA 13 build; it runs on NVIDIA GPUs from Turing
through Blackwell (sm_75 to sm_120, including RTX 50xx). BM25, extraction and
eval need no GPU.

## Quick start on your own repo

Describe the repo in `repos/<name>.toml`:

```toml
name = "myapp"
url = "https://github.com/me/myapp"
commit = "0123456789abcdef0123456789abcdef01234567"   # full SHA, the clone is checked against it
description = "the myapp web backend"                 # used in the reranker instruction
languages = ["python"]                                 # optional, default ["python"]

# fnmatch globs on the repo-relative path; `*` crosses `/`,
# and "*/x/*" also matches a top-level x/.
include = ["myapp/*"]
exclude = ["*/tests/*", "*/test_*.py", "*/migrations/*"]
```

Then:

```sh
tuna clone   --repo myapp                    # depth-1 fetch of exactly that commit
tuna extract --repo myapp
tuna bm25    --repo myapp --questions questions.jsonl
tuna eval    --repo myapp --questions questions.jsonl
```

`questions.jsonl` is your own test set (format under [Questions](#questions);
`samples/saleor_smoke.jsonl` has five examples).

To tune a model on your own repository, follow steps 3 to 5 of
[Reproducing the results](#reproducing-the-results): write a test set and
build the held-out list, write the training questions, then train and score.
There is no single command for this yet.

To use a checkout you already have, set `path = "..."` in the config or
`TUNA_MYAPP_PATH=/path/to/checkout`. `extract` refuses a checkout that isn't at
the pinned commit or has modified tracked files, unless you pass
`--allow-mismatch`.

### Dense retrieval, ColBERT and reranking

`embed` encodes every symbol either through an OpenAI-compatible
`/v1/embeddings` endpoint (vLLM, TEI, ...) or locally with sentence-transformers;
`dense` encodes the questions the same way (the settings are stored with the
vectors) and ranks by cosine:

```sh
# served: the endpoint's model name, a local copy of its tokenizer to count tokens, and its input limit
tuna embed --repo myapp --tag jina05b --model jinaai/jina-code-embeddings-0.5b \
           --url http://127.0.0.1:8000/v1/embeddings --max-tokens 32768 \
           --tokenizer jinaai/jina-code-embeddings-0.5b --tokenizer-revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7
# local: the limit is the model's own max_seq_length
tuna embed --repo myapp --tag lfm-emb --backend local --model LiquidAI/LFM2.5-Embedding-350M
tuna dense --repo myapp --vectors lfm-emb --questions questions.jsonl

tuna colbert --repo myapp --questions questions.jsonl          # LiquidAI/LFM2.5-ColBERT-350M, MaxSim
tuna rerank  --repo myapp --questions questions.jsonl --input dense_lfm-emb \
             --backend hf --model Qwen/Qwen3-Reranker-0.6B
tuna eval    --repo myapp --questions questions.jsonl
```

Prompts default to the model's own `document`/`query` prompts, else the
jina-code-embeddings nl2code prefixes; override them with `--prefix` and
`--query-prefix`. `rerank` runs locally with transformers (`--backend hf`) or
against a vLLM `/rerank` server (`--backend served`; the launch command is in
`src/tuna/rerank.py`).

### Long symbols

Each model reads a symbol at its own maximum input length, counted with its
own tokenizer, prompt and special tokens included. A symbol that fits is read
whole. A longer one is split on line boundaries into chunks that each fit, and
every chunk repeats the symbol's two-line header; the symbol's score is its
best chunk's score (cosine for `dense`, MaxSim for `colbert`). Nothing is cut
silently: `embed` and `colbert` log, per repo and model, how many symbols were
split into how many chunks, and store the counts with the vectors. The one
cut below line level, a single line longer than the whole budget, is logged as
`single lines cut`.

Symbols split / symbols in the benchmark repos at their pinned commits (the two
PostHog symbols over 32k tokens are the `TrinoPrinter` and `ExperimentService`
classes, about 4,100 lines each):

| model | limit (tokens) | Saleor | Zulip | PostHog |
|---|---|---|---|---|
| LFM2.5-Embedding-350M | 512 | 910 / 11,678 | 800 / 6,836 | 3,831 / 29,679 |
| LFM2.5-ColBERT-350M | 512 | 905 / 11,678 | 798 / 6,836 | 3,818 / 29,679 |
| jina-code-embeddings 0.5B / 1.5B, Qwen3-Embedding-0.6B | 32,768 | 0 / 11,678 | 0 / 6,836 | 2 / 29,679 |

## Data formats

### Symbols (`symbols.jsonl`)

One row per class or function, at any nesting depth: module-level functions
and classes, methods, and functions or classes nested inside them.

| field | meaning |
|---|---|
| `fqn` | dotted module path, enclosing symbols, name: `saleor.checkout.tasks.delete_expired_checkouts`, `saleor.order.models.Order.Meta` |
| `kind` | `Class`, `Function`, or `Method` (a function defined directly in a class body) |
| `file_path` | repo-relative, `/`-separated |
| `line_start`, `line_end` | 1-based and inclusive; `line_start` is the first decorator line |
| `body` | the symbol document (below) |
| `body_hash` | sha256 of `body` |

The module path is the file path with `/` replaced by `.` and the extension
dropped. `__init__` stays in it (`saleor/core/__init__.py` gives
`saleor.core.__init__.X`), so every FQN maps to exactly one file. When two
definitions share an FQN (a conditional redefinition, a property setter), the
first one in the file is kept. Only files tracked by git are read.

### Symbol document

The text that BM25, the embedders and the reranker see is a two-line header
followed by the symbol's source exactly as written: decorators, signature,
docstring and body, dedented to column 0. It contains nothing derived, so there
is no call graph, callers or callees, and no summary.

```
# FQN: saleor.giftcard.tasks.deactivate_expired_cards_task
# File: saleor/giftcard/tasks.py:15-25
@app.task
@allow_writer()
def deactivate_expired_cards_task():
    ...
```

A class's document contains its whole source, methods included; each method
also has its own row. Documents longer than a model's input limit are split,
not cut (see [Long symbols](#long-symbols)); `bm25 --cap` and
`rerank --body-cap` are explicit character caps.

### Questions

```json
{"id": "saleor-smoke-1", "question": "How long can an anonymous shopping cart sit untouched before a background job removes it?", "gold_fqns": ["saleor.checkout.tasks.delete_expired_checkouts"]}
```

A question is a hit at k when any of its `gold_fqns` is in the top k. `eval`
warns about gold FQNs missing from `symbols.jsonl`. `samples/saleor_smoke.jsonl`
has five hand-written examples.

### Questions from GitHub issues

`tuna issues --repo <name>` builds a test set from real issues: the question
is the issue title and cleaned body, and the gold is the 1 to 3 functions or
methods that the fixing commit changed in in-scope files, as they are named at
the pin. Fixes are found offline in a blob-less full-history clone
(`<data root>/history/<repo>.git`, cloned on first use) by closing keywords in
commit messages. For squash-merged repos they are also found in the bodies of
merged PRs that have a linked issue. Issue text and PR bodies come from the
GitHub search API, 100 per request, cached under `<data root>/cache/github/`.
No token is needed: a repo costs one search request per 100 issues, at 10 per
minute. Items whose text names a gold symbol or its file are flagged
(`leak_name`, `leak_path`), and `<name>-noleak.jsonl` holds the rest. The
manifest gives the yield at every stage. The module docstring of
`src/tuna/issues.py` has the full rules.

### Outputs

Everything goes under `$TUNA_OUT/<repo>/` (default `~/tuna-data/out/<repo>/`):
`symbols.jsonl`, `manifest.json` (commit, globs, grammar versions, counts),
`vectors/<tag>.npy`, `ranks/<name>.jsonl` and `eval_<questions>.json`.

## Paths and environment

| variable | default | |
|---|---|---|
| `TUNA_DATA` | `~/tuna-data` | root for clones (`repos/<name>`) and outputs |
| `TUNA_OUT` | `$TUNA_DATA/out` | output root (`data/out` in this repository holds the released one) |
| `TUNA_METRICS` | `$TUNA_DATA/metrics` | where the run ledger is appended |
| `TUNA_<NAME>_PATH` | `$TUNA_DATA/repos/<name>` | checkout of one repo |
| `TUNA_REPOS_DIR` | | extra place to look for `<name>.toml` (also searched: `./repos`, this checkout's `repos/`) |
| `TUNA_EMB_URL` | `http://127.0.0.1:8000/v1/embeddings` | embeddings endpoint |
| `TUNA_RR_URL` | `http://127.0.0.1:8001/rerank` | served reranker |
| `TUNA_BENCH_HOME` | `$TUNA_DATA/cc-bench` | the agent experiments' checkouts and scratch directories (`scripts/cc-bench.py`) |
| `TUNA_MACHINE` | the hostname | machine label written on every ledger line |
| `TUNA_LOCAL_MACHINE` | `local` | the label whose ledger lines count as the local machine in timing and energy scripts |

## Adding a language

Symbols come from a per-language `LangSpec` in `src/tuna/langs.py`: the grammar
module, which syntax nodes are symbols and their kind, which wrapper nodes
(such as decorators) belong to a symbol's text, which scopes make a function a
method, and how a file path becomes a module name. To add a language, add its
grammar wheel (for example `tree-sitter-go`) to `pyproject.toml`, add one spec
to `SPECS`, and list it under `languages` in the repo config.

## Benchmark repos

The configs in `repos/` pin five public Python codebases. Three are tuned on
and tested; Netflix Dispatch and a subset of Home Assistant are test-only, and
`tuna train` refuses them.

| repo | licence | role |
|---|---|---|
| [Saleor](https://github.com/saleor/saleor) | BSD-3-Clause | tuning and test |
| [Zulip](https://github.com/zulip/zulip) | Apache-2.0 | tuning and test |
| [PostHog](https://github.com/PostHog/posthog) | MIT | tuning and test |
| [Netflix Dispatch](https://github.com/Netflix/dispatch) | Apache-2.0 | test only |
| [Home Assistant](https://github.com/home-assistant/core) | Apache-2.0 | test only |

Each config gives the pinned commit, the scope and the excluded paths. Tests
and migrations are excluded everywhere.

The benchmark repos keep their own licences. This repository contains none of
their source files: only the configs with their pinned commits, and questions
and metadata about them (question text, gold symbol names and file paths, issue
numbers). The issue sets quote the public title and body of GitHub issues as
their authors wrote them, so they can contain tracebacks and short code
snippets. Two of the agents' answers and one rejected draft of the question
writer quote one function signature each. Run `tuna clone` to fetch the code itself from upstream.

## Models that ship their own code

Some models only load with `trust_remote_code`, because their Hugging Face
repo carries Python (`auto_map`): the Liquid LFM2.5 encoders and LFM3. tuna
loads those only through `tuna.pinned.load()`, which:

- reads the model's row in `src/tuna/pins.toml` (full commit sha, commit date,
  and a note on what the code at that commit does and who read it), and refuses
  any model that has no row;
- passes that sha as both `revision` and `code_revision`, so the weights and
  the executed code come from the read commit;
- refuses a revision that ships pickle-format weights (`.bin`, `.pt`, ...) and
  loads with `use_safetensors=True`;
- sets `USE_HUB_KERNELS=0` so no compiled kernels are fetched from other repos,
  and keeps Triton's JIT cache under `$TUNA_DATA/cache/triton`;
- allows one model whose code patches `transformers.models.lfm2` per process
  (the two LFM2.5 encoders patch it differently).

Every other loader calls `pinned.refuse_unpinned()` first. To move a pin,
read the Python files at the new commit, then update the sha, date and `audit`.

```python
from tuna import pinned
model = pinned.load("LiquidAI/LFM2.5-Embedding-350M")          # SentenceTransformer
colbert = pinned.load("LiquidAI/LFM2.5-ColBERT-350M")          # MultiVectorEncoder (no PyLate needed)
tokenizer, lm = pinned.load("LiquidAI/LFM3-1.7B")              # causal LM
```

## Reproducing the results

These steps rebuild the retrieval numbers on one Linux x86_64 machine with one
NVIDIA GPU. Not covered: the writer check (`scripts/judge-check.py`, which
needs the Claude Code CLI) and the preference test (`scripts/pref-test.py`,
whose rankings came from a job on our rented machines that is not in this
repository). Our runs used a 32 GB card (RTX 5090), and the recipes are sized for it:
jina and Qwen3 train with GradCache, and Qwen3's validation encodes in
16k-token batches. Everything writes under `$TUNA_OUT` (default
`$TUNA_DATA/out`); point it somewhere other than `data/out` to keep the
released files apart from yours.

**1. Environment.** `uv sync --locked` builds `.venv` from `uv.lock` (see
[Install](#install)). Model weights download from Hugging Face on first use,
at fixed revisions:

| model | revision |
|---|---|
| `LiquidAI/LFM2.5-Embedding-350M` | `f35ae2c91d687658dbf1f2b449382f0b019b9808` |
| `LiquidAI/LFM2.5-ColBERT-350M` | `9772bdf797255d8693b83e84aa98e9b2d36dd0be` |
| `jinaai/jina-code-embeddings-0.5b` | `4db235132dafbe56a8b9c5f59b59795ecf58a4a7` |
| `jinaai/jina-code-embeddings-1.5b` | `39aeb4fb9b60f930934c78ae5d749a46287c248a` |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |

The Liquid pins live in `src/tuna/pins.toml`, and the others are passed as
`--revision` by the scripts. `scripts/embed-smoke.py` and
`scripts/liquid-smoke.py` check that the models load.

**2. Repos.** Clone and extract each benchmark repo at its pinned commit:

```sh
for r in saleor zulip posthog dispatch homeassistant; do
  .venv/bin/tuna clone --repo $r && .venv/bin/tuna extract --repo $r
done
```

**3. Questions.** The simplest path is to use the released question sets:
copy `data/out/<repo>/questions/` to `$TUNA_OUT/<repo>/questions/`. Each
set's manifest records its prompt and prompt hash, seed, writer model, guards
and splits, and `heldout-files.txt` is the held-out file list.
`tuna heldout --repo <name> --check qwen-train1` verifies that no training
question touches a held-out file. To write the sets again, write the test
sets first, then build the held-out list with `tuna heldout --repo <name>`,
then write the training sets, which exclude the held-out files:

- *Qwen-written sets* (`qwen-train1`, `qwen-m2`, `qwen-m4-400`, `qwen-bug1`)
  come from the [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
  coding agent 1.2.0 (`scripts/pi-ask`, read-only `read`/`grep`/`find`/`ls`
  tools), which talks to any OpenAI-compatible chat-completions server. Put
  the server in `$PI_CODING_AGENT_DIR/models.json` as provider `local` (the
  name `scripts/pi-ask` expects) (`"api": "openai-completions"`, your `baseUrl`, and a model entry). Our
  settings: model `qwen3.6-35b-a3b`, `contextWindow` 160000, `maxTokens`
  16384, and the server's defaults for everything else. The server was
  [ninfer](https://github.com/Neroued/ninfer), serving a mixed-precision quantised conversion of Qwen3.6-35B-A3B
  with thinking on, temperature 0.6, top-p 0.95, top-k 20 and a presence
  penalty of 1.0, all of them the server's defaults. We read these back from
  the server's log after the runs. The ledger entry "writer identity
  (recorded after the fact)" of 2026-11-12T16:19:29Z has the model revision,
  the file's checksum and the request counts. For the sets written on rented
  machines, only the configuration we shipped is recorded. Per question: 90 s timeout, at most 40 tool calls,
  up to 3 attempts. We ran up to 8 writers in parallel with seed 20261111
  (`qwen-bug1`: 20261113) and filter `v2`. The prompt is `question-v2` for
  `qwen-train1` and `qwen-m2`, `question-v2-gold` for the gold-targeted sets
  and `question-bug1` for `qwen-bug1`. Each manifest names its prompt and the
  prompt's hash. The training set is written by
  `scripts/m2-train-questions.sh`.
- *Claude-written sets* (`claude-m2`, `claude-m3`) come from
  Claude Code through `scripts/claude-ask` (model `claude-opus-5-5`, the same
  read-only tools and guards) via `scripts/claude-test-set-m3.sh`; `claude-t400`,
  for Dispatch and Home Assistant, via `scripts/claude-test-set.sh` (scored by
  `scripts/m2-testrepo.sh`). `scripts/merge-question-sets.py` builds `claude-m3`
  from `claude-m2` and `claude-m3-new`.
- *LFM3-written sets* (`lfm3-t400`) come from LFM3-1.7B through
  `scripts/lfm3-test-set.sh`.
- *Issue sets* (`issues-v1`, `-title`, `-noleak`) come from
  `tuna issues --repo <name>` (see
  [Questions from GitHub issues](#questions-from-github-issues)).
- The gold-targeted sets (`qwen-m4-400`, the doc2query index questions
  `index-d2q`) use `tuna questions --gold-list`. `scripts/m2-repo.sh <repo>`
  writes `qwen-train1`, `qwen-bug1` and `qwen-m4-400` (the latter two as
  below), then trains and scores the repo.

**4. Training.** `scripts/m2-train-models.sh <repo>` trains the six tuned
rows of a repo with one recipe: LFM2.5, jina 0.5B and Qwen3 0.6B, each full
and LoRA, on `qwen-train1`, with mined hard negatives and the epoch picked
on validation MRR@10. It then embeds the corpus with each tuned model. When
the set exists, the same script also trains the bug-style row
(`lfm25-ft-lora-bug`, on `qwen-train1` merged with `qwen-bug1` into
`qwen-trainbug1` by `scripts/merge-sets.py`). The re-mined row
(`lfm25-ft-lora-remine`, Saleor and LFM2.5 only) is one `tuna train` call,
`tuna train --repo saleor --name lfm25-ft-lora-remine` with the LoRA recipe
of `scripts/m2-train-models.sh` and `--remine-every 2`, then
`tuna embed --repo saleor --backend local --weights $TUNA_OUT/saleor/models/lfm25-ft-lora-remine`
`--tag lfm25-ft-lora-remine --model LiquidAI/LFM2.5-Embedding-350M`. The other extra rows have their own scripts: `m2-transfer.sh` (cross-repo),
`m2-pooled.sh` (one model for all three repos), `m2-curve.sh` (question
count), `m2-seeds.sh` (seeds), `m2-4b.sh` (Qwen3-Embedding-4B, stock).

**5. Scoring.** `scripts/m2-score.sh <repo>` scores BM25, ColBERT and every
stock and tuned dense row on every test set (`eval_<set>.json`, with
per-question ranks), and benchmarks search speed. The doc2query rows come
from `scripts/d2q-write.sh <repo>` (writing) and `scripts/d2q-index.py`
(indexing and scoring).

**Tolerance.** Expect a rerun to move a tuned score. On Saleor's claude-m3
questions, our repeated runs (three seeds, two GPU models) moved MRR@10 by up
to .05 for LFM2.5, .02 for jina 0.5B and .03 for Qwen3 0.6B. A rerun with the
same seed on a different GPU model can keep a different epoch, because
validation scores are close across epochs.

**6. Intervals and tables.** `scripts/m2-intervals.sh final` (and `d2q`)
writes the paired bootstrap intervals and the summary tables under
`$TUNA_OUT/bootstrap` and `$TUNA_OUT/tables`.

**7. Agent experiments.** `scripts/cc-bench.py` runs Claude Code on the issue
sets in a clean, read-only copy of each repository, one arm at a time, and
grades the file and symbol it names; its docstring describes the isolation
and the arms. `scripts/cc-issues.sh` ran the first experiment (a search tool
over the index, `scripts/tuna-mcp`) and `scripts/cc-hint-full.sh` the second
(a card of search hits placed before the task by `scripts/tuna-hint-hook`).
`scripts/cc-analysis.py` writes the reports in `data/out/cc-bench/`. Running
them needs the Claude Code CLI and an account; the released `runs.jsonl`
files hold every run's result.

**The order we ran.** For each of Saleor, Zulip and PostHog, with the
question-writer server running:

```sh
.venv/bin/tuna questions --repo $r --name qwen-bug1 --prompt question-bug1 --n 500 --seed 20261113 \
  --workers 8 --exclude-files $TUNA_OUT/$r/questions/heldout-files.txt \
  --per-file <3|5|3> --symbols-per-question <3|1.5|4> --val-fraction 0.1 --timeout 240 --max-tool-calls 40
.venv/bin/tuna questions --repo $r --name qwen-m4-400 --prompt question-v2-gold \
  --gold-list $TUNA_OUT/$r/questions/claude-m3.jsonl --n 400 --writer pi --workers 8 --timeout 90 --max-tool-calls 40
```

(per-file and symbols-per-question for Saleor, Zulip, PostHog; as in
`scripts/m2-repo.sh`), then on the GPU `scripts/m2-train-models.sh $r` for each
repo, the re-mined row above, and the scoring loop
`for r in saleor zulip posthog; do scripts/m2-score.sh $r; done`, followed by
`scripts/m2-transfer.sh`, `scripts/m2-pooled.sh`,
`scripts/m2-testrepo.sh dispatch|homeassistant` and
`scripts/m2-intervals.sh final`. Every step appends to the run ledger
(`$TUNA_METRICS`); the released one is in `data/metrics/`.

## Scripts

- `scripts/embed-smoke.py`: loads the stock embedders on the GPU at pinned
  revisions and embeds a few symbols, as an environment check.
- `scripts/liquid-smoke.py embedding|colbert|lfm3`: the same check for the
  pinned Liquid models, one model per process; for LFM3, one chat turn with a
  tool definition and a check that the reply is a well-formed tool call.
- `scripts/pi-ask`: asks the pi coding agent a question about a checkout with
  read-only tools and prints the answer, used to draft questions.
  `scripts/pi-ask` needs `PI_CODING_AGENT_DIR`; set `PI_BIN` if `pi` is not on
  PATH.
- `scripts/nanobeir-fetch.py`, `scripts/nanobeir-check.py`: reproduce LFM2.5's
  NanoBEIR score through tuna's encoder.
- `scripts/merge-question-sets.py`: builds claude-m3 from claude-m2 and
  claude-m3-new.
- `scripts/seed-variance.py`, `scripts/writer-gap.py`: the seed-spread and
  writer-gap tables (they read the rank files, which `m2-score.sh` rebuilds).
- `scripts/paired-bootstrap.py`, `scripts/transfer-split.py`: the intervals.
- `scripts/bootstrap-txt.py`: renders a bootstrap JSON file as the text table
  that `paired-bootstrap.py` prints, without recomputing it.
- `scripts/public-scan.py`: the scan this release passed (home paths, personal
  emails, private addresses, keys and tokens).

## How this work was produced

The harness, the runs and the drafts of the text were produced with AI coding agents under the author's direction, and the notes in the run ledger are written by them. The author is responsible for all content.

## License

MIT, see [LICENSE](LICENSE). The benchmark repos keep their own licences
(see [Benchmark repos](#benchmark-repos)). The tuned weights, when released,
come under each base model's licence.
