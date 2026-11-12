"""Fine-tune a single-vector embedder on one repo's training questions.

  tuna train --repo saleor --questions qwen-train1 --mode full --name lfm25-ft
  tuna train --repo saleor --questions qwen-train1 --mode lora --name lfm25-lora

Plain PyTorch, no trainer framework:

  data       <set>.train.jsonl / <set>.val.jsonl (tuna questions --val-fraction).
             Every symbol is read at the model's own maximum length; longer
             symbols are split exactly as for evaluation (chunking.py).
  positives  the gold symbol's chunk the stock model finds most similar to
             the question (the whole symbol when it fits)
  negatives  in-batch (every other document in the batch) plus --hard-negatives
             per question, sampled each epoch from the stock model's top
             --mine-depth chunks. Never a negative: the gold symbol, a symbol
             enclosing it or nested in it (a class holds its methods' code),
             or anything in a held-out (test-set) file, so training never sees
             held-out code. The same masking applies in-batch.
  loss       InfoNCE, cosine similarities / --temperature
  modes      full: every weight trains. lora: rank --lora-rank adapters on all
             linear layers of the transformer (attention, short convolution
             projections, feed-forward); the rest stays frozen.
  epochs     after each epoch, MRR@10 of the validation questions against the
             whole repo (best chunk per symbol); the best epoch is kept.

Output in <out>/<repo>/models/<name>/: weights.safetensors (bfloat16; the
whole model for full, only the adapter for lora) and tuna-model.json (base
model and pinned revision, settings, data hashes, per-epoch validation,
train time, peak VRAM, size). tuna embed --weights loads it on top of the
pinned base model, so no new code is ever loaded.
"""
import json
import math
import os
import random
import time

from . import chunking, common, heldout, metrics, pinned

DEFAULT_BASE = "LiquidAI/LFM2.5-Embedding-350M"
# Question sets written for the index side (the doc2query control) carry this prefix; training refuses them.
INDEX_PREFIX = "index-"
# Every linear layer of the transformer blocks, by architecture: LFM2 (attention q/k/v/out,
# short-conv in/out, feed-forward w1/w2/w3) and Qwen2/Qwen3 (attention q/k/v/o, MLP
# gate/up/down). The names do not overlap, so one list serves both.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3",
                "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- LoRA

def lora_module():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """y = base(x) + scale * B(A(dropout(x))); B starts at zero, so training starts at the base model."""

        def __init__(self, base, rank, alpha, dropout):
            super().__init__()
            self.base = base
            self.rank, self.scale = rank, alpha / rank
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype,
                                                   device=base.weight.device))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                                                   device=base.weight.device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

        def merged(self):
            """The base Linear with the adapter folded into its weight."""
            with torch.no_grad():
                delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale
                self.base.weight += delta.to(self.base.weight.dtype)
            return self.base

    return LoRALinear


def add_lora(model, rank, alpha, dropout, targets=LORA_TARGETS):
    """Wrap target Linear layers of a transformer in LoRA adapters. Returns the adapter names."""
    from torch import nn
    LoRALinear = lora_module()
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    return wrapped


def merge_lora(model):
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == "LoRALinear":
                setattr(module, child_name, child.merged())


def lora_state(model):
    return {k: v for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


# ---------------------------------------------------------------- load a trained model

def load_weights(st_model, weights_dir, revision=None):
    """Apply a tuna-trained checkpoint to a freshly loaded base model (in place). The base
    must be the exact revision it was trained from: pins.toml's, or `revision` for others."""
    from safetensors.torch import load_file
    with open(os.path.join(weights_dir, "tuna-model.json")) as fh:
        meta = json.load(fh)
    if meta["base"] in pinned.pins():
        expected = pinned.pin_for(meta["base"]).sha
    else:
        expected = revision
    if expected != meta["base_revision"]:
        raise SystemExit(f"{weights_dir} was trained on {meta['base']}@{meta['base_revision']}, "
                         f"but the base loaded is at {expected}")
    model = st_model[0].auto_model
    tensors = load_file(os.path.join(weights_dir, "weights.safetensors"))
    dtype = next(model.parameters()).dtype
    if meta["mode"] == "full":
        missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in tensors.items()}, strict=True)
    else:
        lc = meta["lora"]
        add_lora(model, lc["rank"], lc["alpha"], 0.0, tuple(lc["targets"]))
        missing, unexpected = model.load_state_dict({k: v.to(dtype) for k, v in tensors.items()}, strict=False)
        if unexpected or any(".lora_" in k for k in missing):
            raise SystemExit(f"{weights_dir}: adapter does not fit the model")
        merge_lora(model)
    return meta


def load_base(base, revision, device, dtype="float32"):
    """(SentenceTransformer, revision) at an exact revision: pins.toml's for remote-code models,
    else --base-revision (no remote code)."""
    if base in pinned.pins():
        return pinned.load(base, device=device, dtype=dtype), pinned.pin_for(base).sha
    import torch
    from sentence_transformers import SentenceTransformer
    pinned.refuse_unpinned(base)
    if not revision or len(revision) != 40:
        raise SystemExit(f"{base} is not in pins.toml: give its full commit as --base-revision")
    st = SentenceTransformer(base, revision=revision, device=device, trust_remote_code=False,
                             model_kwargs={"dtype": getattr(torch, dtype)})
    return st, revision


# ---------------------------------------------------------------- data

def related(a, b):
    """True when one FQN encloses the other (or they are equal)."""
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def build_corpus(syms, count_tokens, limit):
    texts, owner, stats = chunking.split_symbols([common.symbol_doc(s) for s in syms], count_tokens, limit)
    return texts, owner, stats


def to_device(feats, device):
    """Model inputs from sentence-transformers' preprocess(), tensors moved to `device`."""
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in feats.items()}


def token_lengths(st, texts, prefix):
    tok = st.tokenizer
    return [min(len(tok(prefix + t, add_special_tokens=True)["input_ids"]), st.max_seq_length) for t in texts]


def encode(st, texts, prefix, batch_tokens, device, lengths=None):
    """Embed texts (inference, bf16 autocast on CUDA), longest first under a padded-token budget."""
    import numpy as np
    import torch
    if lengths is None:
        lengths = token_lengths(st, texts, prefix)
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    out = None
    st.eval()
    with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                         enabled=device.startswith("cuda")):
        i = 0
        while i < len(order):
            n = max(1, min(256, batch_tokens // max(lengths[order[i]], 1)))
            part = order[i:i + n]
            feats = to_device(st.preprocess([prefix + texts[j] for j in part]), device)
            emb = torch.nn.functional.normalize(st(feats)["sentence_embedding"].float(), dim=-1).cpu().numpy()
            if out is None:
                out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
            out[part] = emb
            i += n
    return out


def mine(Q, D, owner, syms, train_rows, by_fqn, held, depth, skip=None, stats=None):
    """Per question: (positive chunk, [candidate hard-negative chunks]) from stock similarities.

    Candidates are walked in the model's own stock-score order. Never a candidate: chunks of
    held-out files (pool = training files only), the gold symbol and symbols enclosing or
    nested in it. skip(qi, chunk, score, gold_score) -> True drops a likely false negative (see
    own_score_rule, judge_rule). `stats` (a dict) receives the walk's counts."""
    import numpy as np
    chunk_fqn = [syms[o]["fqn"] for o in owner]
    chunk_held = np.array([syms[o]["file_path"] in held for o in owner])
    chunks_of = {}
    for c, o in enumerate(owner):
        chunks_of.setdefault(o, []).append(c)
    S = Q @ D.T
    out = []
    for qi, row in enumerate(train_rows):
        g = row["gold_fqn"]
        pos = max(chunks_of[by_fqn[g]], key=lambda c: S[qi, c])
        s = S[qi].copy()
        s[chunk_held] = -np.inf
        gs = float(S[qi, pos])
        cand = []
        for c in np.argsort(-s):
            if len(cand) >= depth or not np.isfinite(s[c]):
                break
            if related(chunk_fqn[c], g):
                if stats is not None:
                    stats["related_skipped"] = stats.get("related_skipped", 0) + 1
                continue
            if skip is not None and skip(qi, int(c), float(s[c]), gs):
                if stats is not None:
                    stats["fn_dropped"] = stats.get("fn_dropped", 0) + 1
                continue
            cand.append(int(c))
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + len(cand)
        out.append((int(pos), cand))
    if stats is not None:
        walked = stats.get("kept", 0) + stats.get("fn_dropped", 0)
        stats["fn_dropped_share"] = round(stats.get("fn_dropped", 0) / walked, 4) if walked else 0.0
    return out


def own_score_rule(ratio):
    """Likely false negative: the model's own stock score above ratio x its score for the gold."""
    def skip(qi, c, score, gold_score):
        return score > (ratio * gold_score if gold_score > 0 else gold_score)
    return skip


def judge_rule(judge_sym_scores, owner, gold_syms, ratio):
    """Likely false negative, decided by one judge model for every trained model: the judge's
    score for the candidate's symbol above ratio x the judge's score for the gold symbol."""
    def skip(qi, c, score, gold_score):
        js = judge_sym_scores[qi]
        g = js[gold_syms[qi]]
        return js[owner[c]] > (ratio * g if g > 0 else g)
    return skip


def judge_scores(work, judge_tag, questions_set, train_rows, n_symbols, device, cfg_name):
    """Per question and symbol, the judge's stock score (best chunk). The judge's corpus vectors
    come from tuna embed; its question vectors are encoded once per repo and question set and
    cached (models/judge-<tag>_<set>.npy), so every trained model uses the same judgement."""
    import numpy as np
    from . import encoders
    vdir = os.path.join(work, "vectors")
    with open(os.path.join(vdir, f"{judge_tag}.json")) as fh:
        meta = json.load(fh)
    if meta["symbols_sha256"] != common.file_sha256(os.path.join(work, "symbols.jsonl")):
        raise SystemExit(f"vectors/{judge_tag} was built from another symbols.jsonl")
    cache = os.path.join(work, "models", f"judge-{judge_tag}_{questions_set}.npy")
    ids_path = cache[:-4] + ".ids.json"
    ids = [r["id"] for r in train_rows]
    if os.path.exists(cache) and os.path.exists(ids_path) and json.load(open(ids_path)) == ids:
        Qj = np.load(cache)
    else:
        with metrics.Stage("judge-encode", cfg_name, meta["model"], {
                "judge_vectors": judge_tag, "questions": questions_set, "query_prefix": meta.get("query_prefix"),
                "purpose": "false-negative filter for hard-negative mining"}) as stage:
            enc = encoders.from_meta(meta, device=device)
            Qj = enc.encode_queries([r["question"] for r in train_rows])
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, Qj)
            with open(ids_path, "w") as fh:
                json.dump(ids, fh)
            stage.counts(items=len(train_rows), questions=len(train_rows))
            stage.outputs(cache)
            del enc
    Dj = np.load(os.path.join(vdir, f"{judge_tag}.npy"))
    oj = np.load(os.path.join(vdir, f"{judge_tag}_owner.npy"))
    return chunking.max_per_symbol(Qj @ Dj.T, oj, n_symbols)


def stock_vectors(work, tag, base, owner):
    """The saved stock corpus matrix (tuna embed) if it is exactly this corpus, else (None, reason)."""
    import numpy as np
    if not tag:
        return None, "not requested"
    vdir = os.path.join(work, "vectors")
    try:
        with open(os.path.join(vdir, f"{tag}.json")) as fh:
            meta = json.load(fh)
    except OSError:
        return None, f"no vectors/{tag}"
    saved_owner = np.load(os.path.join(vdir, f"{tag}_owner.npy"))
    if (meta.get("model") != base or meta.get("weights") or meta["symbols_sha256"] != common.file_sha256(
            os.path.join(work, "symbols.jsonl")) or len(saved_owner) != len(owner)
            or not np.array_equal(saved_owner, np.asarray(owner))):
        return None, f"vectors/{tag} is not this corpus"
    return np.load(os.path.join(vdir, f"{tag}.npy")), f"vectors/{tag}"


# ---------------------------------------------------------------- train

def retrieval_ranks(Q, D, owner, n_symbols, gold_idx):
    """1-based rank of each question's gold symbol (best chunk per symbol)."""
    import numpy as np
    S = chunking.max_per_symbol(Q @ D.T, owner, n_symbols)
    gold_scores = S[np.arange(len(gold_idx)), gold_idx]
    return (S > gold_scores[:, None]).sum(1) + 1


def metrics_from_ranks(ranks, ks=(1, 5, 10)):
    import numpy as np
    ranks = np.asarray(ranks)
    out = {f"R@{k}": round(float((ranks <= k).mean()), 4) for k in ks}
    out["MRR@10"] = round(float(np.where(ranks <= 10, 1.0 / ranks, 0.0).mean()), 4)
    return out


def prepare_repo(repo, args, st, dp, qp, limit, device):
    """One repo's training data: questions (train/val, held-out checked), corpus split at the model's
    limit, stock vectors, mined hard negatives (own stock scores, the judge's false-negative rule, pool =
    training files only) and the stock validation ranks. Writes the repo's mining ledger line."""
    import numpy as np
    cfg = common.load_config(repo)
    work = common.workdir(args.out, cfg)
    syms = common.load_symbols(work)
    by_fqn = {s["fqn"]: i for i, s in enumerate(syms)}
    qdir = os.path.join(work, "questions")
    train_path, val_path = (os.path.join(qdir, f"{args.questions}.{s}.jsonl") for s in ("train", "val"))
    train_rows, val_rows = common.read_jsonl(train_path), common.read_jsonl(val_path)
    held_path = os.path.join(qdir, "heldout-files.txt")
    held = heldout.load_list(held_path)
    bad = [r["id"] for r in train_rows + val_rows if r["file"] in held or syms[by_fqn[r["gold_fqn"]]]["file_path"] in held]
    if bad:
        raise SystemExit(f"{repo}: training data touches held-out files: {bad[:5]}")
    all_symbols = len(syms)
    if args.limit_train:  # smoke tests: a few questions and a small corpus around their golds
        train_rows = train_rows[:args.limit_train]
        val_rows = val_rows[:max(4, args.limit_train // 4)]
        if args.limit_corpus:
            keep = {by_fqn[r["gold_fqn"]] for r in train_rows + val_rows} | set(range(args.limit_corpus))
            syms = [syms[i] for i in sorted(keep)]
            by_fqn = {s["fqn"]: i for i, s in enumerate(syms)}
    texts, owner, stats = build_corpus(syms, chunking.token_counter(st.tokenizer, dp), limit)
    print(chunking.describe(stats, cfg["name"], args.base), flush=True)
    doc_lengths = token_lengths(st, texts, dp)
    chunk_fqn = [syms[o]["fqn"] for o in owner]
    t0 = time.time()
    with metrics.Stage("mine", cfg["name"], args.base, {
            "mine_depth": args.mine_depth, "questions": args.questions, "stock_vectors": args.stock_vectors,
            "fn_rule": args.fn_rule, "fn_ratio": args.fn_ratio, "fn_judge": args.fn_judge,
            "pool": "training files only (held-out masked)", "pooled_run": bool(args.pool),
            "skip": "gold, enclosing and nested symbols; likely false negatives by fn_rule"}) as stage:
        D0, source = stock_vectors(work, args.stock_vectors, args.base, owner) if not args.limit_train else (None, None)
        if D0 is None:
            D0, source = encode(st, texts, dp, args.batch_tokens, device, doc_lengths), "encoded"
        Qtr = encode(st, [r["question"] for r in train_rows], qp, args.batch_tokens, device)
        Qval = encode(st, [r["question"] for r in val_rows], qp, args.batch_tokens, device)
        # both rules are measured; only args.fn_rule shapes the negatives
        own_walk, walk = {}, {}
        mine(Qtr, D0, owner, syms, train_rows, by_fqn, held, args.mine_depth, own_score_rule(args.fn_ratio), own_walk)
        if args.fn_rule == "judge":
            gold_syms = [by_fqn[r["gold_fqn"]] for r in train_rows]
            # the judge scores the whole repo; keep the columns of the symbols in use
            n_all = all_symbols if args.limit_corpus and args.limit_train else len(syms)
            J = judge_scores(work, args.fn_judge, args.questions, train_rows, n_all, device, cfg["name"])
            if n_all != len(syms):
                full = {s["fqn"]: i for i, s in enumerate(common.load_symbols(work))}
                J = J[:, [full[s["fqn"]] for s in syms]]
            rule = judge_rule(J, owner, gold_syms, args.fn_ratio)
        else:
            rule = own_score_rule(args.fn_ratio) if args.fn_rule == "own" else None
        mined = mine(Qtr, D0, owner, syms, train_rows, by_fqn, held, args.mine_depth, rule, walk)
        remine_rule = rule  # the same skips, the same (stock) judge, for re-mining rounds
        stage.extra(mining_walk=walk, own_rule_walk=own_walk,
                    fn_dropped_share={"own_score_rule": own_walk["fn_dropped_share"],
                                      "judge_rule": walk["fn_dropped_share"] if args.fn_rule == "judge" else None,
                                      "used": args.fn_rule})
        val_gold = np.array([by_fqn[r["gold_fqn"]] for r in val_rows])
        ranks0 = retrieval_ranks(Qval, D0, np.asarray(owner), len(syms), val_gold)
        val0 = metrics_from_ranks(ranks0)
        stage.config(corpus_vectors=source)
        stage.counts(items=len(train_rows), questions=len(train_rows), val_questions=len(val_rows),
                     symbols=len(syms), chunks=len(texts), split_symbols=stats.split,
                     negatives_per_question=round(float(np.mean([len(c) for _, c in mined])), 2))
        stage.extra(stock_val=val0)
        mined_path = os.path.join(work, "models", f"{args.name}.mined.jsonl")
        os.makedirs(os.path.dirname(mined_path), exist_ok=True)
        common.write_jsonl(mined_path, [{"id": r["id"], "positive": chunk_fqn[pos], "positive_chunk": pos,
                                         "negatives": [chunk_fqn[c] for c in cand], "negative_chunks": cand}
                                        for r, (pos, cand) in zip(train_rows, mined)])
        stage.outputs(mined_path)
    print(f"[train] {cfg['name']}: corpus vectors {source}; mined {args.mine_depth} candidates for {len(train_rows)} "
          f"questions in {time.time() - t0:.0f}s; false-negative rule {args.fn_rule} (> {args.fn_ratio} x gold) dropped "
          f"{walk.get('fn_dropped', 0)} ({walk['fn_dropped_share']:.1%}); the own-score rule would drop "
          f"{own_walk['fn_dropped_share']:.1%}; stock val {val0}", flush=True)
    return {"cfg": cfg, "work": work, "syms": syms, "train_rows": train_rows, "val_rows": val_rows,
            "train_path": train_path, "val_path": val_path, "held_path": held_path, "texts": texts,
            "owner_np": np.asarray(owner), "stats": stats, "doc_lengths": doc_lengths, "chunk_fqn": chunk_fqn,
            "mined": mined, "val_gold": val_gold, "ranks0": ranks0, "val0": val0, "owner": owner,
            "by_fqn": by_fqn, "held": held, "rule": remine_rule}


def main(args):
    import numpy as np
    import torch
    from safetensors.torch import save_file

    if args.lr is None:
        args.lr = 2e-5 if args.mode == "full" else 2e-4
    if args.questions.startswith(INDEX_PREFIX):  # index-side questions (doc2query control) are never training data
        raise SystemExit(f"{args.questions}: index-side question sets ({INDEX_PREFIX}*) are never training data")
    repos = [args.repo] + list(args.pool or [])
    for r in repos:  # transfer-only test repos (Dispatch, Home Assistant) never provide training data
        if common.load_config(r).get("test_only"):
            raise SystemExit(f"{r}: a test-only repo (test_only = true in its config); nothing trains on it")
    pooled = len(repos) > 1
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    gpu = device.startswith("cuda") and torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()

    st, base_revision = load_base(args.base, args.base_revision, device)
    model = st[0].auto_model
    # the prompts the model is scored with (encoders.LocalEncoder's defaults), unless given
    prompts = getattr(st, "prompts", None) or {}
    dp = args.prefix if args.prefix is not None else prompts.get("document", common.PASSAGE_PREFIX)
    qp = args.query_prefix if args.query_prefix is not None else prompts.get("query", common.QUERY_PREFIX)
    limit = int(st.max_seq_length)

    # every repo prepared on its own (held-out list, corpus, stock vectors, mining); then one pool of
    # training items with global chunk ids, so batches may mix repos
    R = [prepare_repo(r, args, st, dp, qp, limit, device) for r in repos]
    texts, chunk_fqn, items = [], [], []
    offsets = []
    for rd in R:
        off = len(texts)
        offsets.append(off)
        texts += rd["texts"]
        chunk_fqn += rd["chunk_fqn"]
        for row, (pos, cand) in zip(rd["train_rows"], rd["mined"]):
            items.append((row, pos + off, [c + off for c in cand]))
    item_start = [sum(len(r["train_rows"]) for r in R[:i]) for i in range(len(R))]
    n_val = sum(len(rd["val_rows"]) for rd in R)
    val0 = metrics_from_ranks(np.concatenate([rd["ranks0"] for rd in R]))

    if args.mode == "lora":
        for p in model.parameters():
            p.requires_grad_(False)
        wrapped = add_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
        model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        if args.mode == "lora":  # frozen embeddings: let gradients reach the checkpointed blocks
            model.enable_input_require_grads()
    steps_per_epoch = math.ceil(len(items) / args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    if pooled:
        out_dir = os.path.join(os.path.expanduser(args.out), "pooled", "models", args.name)
        stage_repo = "pooled:" + "+".join(rd["cfg"]["name"] for rd in R)
    else:
        out_dir = os.path.join(R[0]["work"], "models", args.name)
        stage_repo = R[0]["cfg"]["name"]
    os.makedirs(out_dir, exist_ok=True)
    history, best, best_epoch, bad_epochs, step = [{"epoch": 0, **val0}], -1.0, 0, 0, 0
    t_train = time.time()
    autocast = dict(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=gpu)

    def embed_batch(strings):
        feats = to_device(st.preprocess(strings), device)
        return torch.nn.functional.normalize(st(feats)["sentence_embedding"].float(), dim=-1)

    def minibatches(strings):
        """Index groups under --cache-tokens padded tokens, longest first (GradCache)."""
        lengths = [min(len(st.tokenizer(t, add_special_tokens=True)["input_ids"]), st.max_seq_length) for t in strings]
        order = sorted(range(len(strings)), key=lambda i: -lengths[i])
        groups, i = [], 0
        while i < len(order):
            n = max(1, args.cache_tokens // max(lengths[order[i]], 1))
            groups.append(order[i:i + n])
            i += n
        return groups

    def cached_step(q_strings, d_strings, labels, mask):
        """One contrastive step with GradCache: embeddings without gradients in small groups,
        the loss and its gradient on the embeddings, then each group again with gradients,
        its RNG state restored. Same gradients as the full-batch step, bounded memory."""
        strings = q_strings + d_strings
        groups = minibatches(strings)
        reps, states = [None] * len(strings), []
        with torch.no_grad():
            for g in groups:
                states.append((torch.get_rng_state(), torch.cuda.get_rng_state() if gpu else None))
                with torch.autocast(**autocast):
                    e = embed_batch([strings[i] for i in g])
                for k, i in enumerate(g):
                    reps[i] = e[k]
        E = torch.stack(reps).detach().requires_grad_(True)
        qe, de = E[:len(q_strings)], E[len(q_strings):]
        logits = ((qe @ de.T) / args.temperature).masked_fill(mask, float("-inf"))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels, device=device))
        loss.backward()
        grads = E.grad
        for g, (cpu_state, cuda_state) in zip(groups, states):
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)
            with torch.autocast(**autocast):
                e = embed_batch([strings[i] for i in g])
            torch.autograd.backward(e, grad_tensors=grads[g])
        return loss.detach()

    def validate(keep_corpus=False):
        """Each repo's val questions ranked against its own corpus; pooled over all val questions.
        keep_corpus: also return each repo's corpus vectors (for re-mining with this checkpoint)."""
        ranks, per_repo, corpora = [], {}, []
        for rd in R:
            D = encode(st, rd["texts"], dp, args.batch_tokens, device, rd["doc_lengths"])
            Qv = encode(st, [r["question"] for r in rd["val_rows"]], qp, args.batch_tokens, device)
            rk = retrieval_ranks(Qv, D, rd["owner_np"], len(rd["syms"]), rd["val_gold"])
            per_repo[rd["cfg"]["name"]] = metrics_from_ranks(rk)
            ranks.append(rk)
            corpora.append(D if keep_corpus else None)
            del D
        return metrics_from_ranks(np.concatenate(ranks)), per_repo, corpora

    def remine(epoch, corpora):
        """ANCE-style round: hard negatives re-mined from the current checkpoint's index (the
        corpus vectors validation just built), with the same depth, skips and stock judge."""
        rounds = []
        for i, (rd, D) in enumerate(zip(R, corpora)):
            with metrics.Stage("remine", rd["cfg"]["name"], args.base, {
                    "name": args.name, "after_epoch": epoch, "mine_depth": args.mine_depth, "fn_rule": args.fn_rule,
                    "fn_ratio": args.fn_ratio, "fn_judge": args.fn_judge,
                    "pool": "training files only (held-out masked)", "index": "current checkpoint"}) as stage:
                Qt = encode(st, [r["question"] for r in rd["train_rows"]], qp, args.batch_tokens, device)
                walk = {}
                new = mine(Qt, D, rd["owner"], rd["syms"], rd["train_rows"], rd["by_fqn"], rd["held"],
                           args.mine_depth, rd["rule"], walk)
                off, changed, total = offsets[i], 0, 0
                for k, (pos, cand) in enumerate(new):
                    j = item_start[i] + k
                    row, _, old = items[j]
                    gc = [c + off for c in cand]
                    changed += len(set(gc) - set(old))
                    total += len(gc)
                    items[j] = (row, pos + off, gc)
                share = round(changed / total, 4) if total else 0.0
                stage.counts(items=len(rd["train_rows"]), questions=len(rd["train_rows"]), chunks=len(rd["texts"]))
                stage.extra(new_negative_share=share, mining_walk=walk)
            rounds.append({"repo": rd["cfg"]["name"], "after_epoch": epoch, "new_negative_share": share,
                           "fn_dropped_share": walk.get("fn_dropped_share")})
            print(f"[train] re-mined {rd['cfg']['name']} after epoch {epoch}: {share:.1%} of negatives are new",
                  flush=True)
        return rounds

    train_stage = metrics.Stage("train", stage_repo, args.base, {
        "mode": args.mode, "repos": repos, **{k: getattr(args, k) for k in (
            "questions", "name", "epochs", "batch", "hard_negatives", "mine_depth", "fn_rule", "fn_ratio", "fn_judge",
            "lr", "weight_decay", "warmup", "temperature", "seed", "patience", "grad_checkpointing", "lora_rank",
            "lora_alpha", "lora_dropout", "limit_train", "limit_corpus", "remine_every")}})
    with train_stage:
        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()
            st.train()
            order = list(range(len(items)))
            rng.shuffle(order)
            losses = []
            for b in range(0, len(order), args.batch):
                batch = [items[i] for i in order[b:b + args.batch]]
                docs, labels = [], []
                for row, pos, cand in batch:
                    labels.append(len(docs))
                    docs.append(pos)
                    docs.extend(rng.sample(cand, min(args.hard_negatives, len(cand))))
                doc_fqn = [chunk_fqn[c] for c in docs]
                q_fqn = [row["gold_fqn"] for row, _, _ in batch]
                # mask documents that are, enclose or sit inside another question's gold: false negatives
                mask = torch.tensor([[related(df, qf) and j != labels[k] for j, df in enumerate(doc_fqn)]
                                     for k, qf in enumerate(q_fqn)], device=device)
                opt.zero_grad(set_to_none=True)
                if args.grad_cache:
                    loss = cached_step([qp + row["question"] for row, _, _ in batch],
                                       [dp + texts[c] for c in docs], labels, mask)
                else:
                    with torch.autocast(**autocast):
                        qe = embed_batch([qp + row["question"] for row, _, _ in batch])
                        de = embed_batch([dp + texts[c] for c in docs])
                    logits = (qe @ de.T) / args.temperature
                    logits = logits.masked_fill(mask, float("-inf"))
                    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels, device=device))
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                step += 1
                losses.append(loss.item())
            train_s = time.time() - t_epoch
            opt.zero_grad(set_to_none=True)  # free gradients before the validation pass
            if gpu:
                torch.cuda.empty_cache()
            do_remine = args.remine_every and epoch % args.remine_every == 0 and epoch < args.epochs
            val, val_per_repo, corpora = validate(keep_corpus=bool(do_remine))
            epoch_s = time.time() - t_epoch
            history.append({"epoch": epoch, "loss": round(float(np.mean(losses)), 4), **val,
                            "train_seconds": round(train_s, 1), "val_seconds": round(epoch_s - train_s, 1),
                            "epoch_seconds": round(epoch_s, 1),
                            "samples_per_s": round(len(items) / train_s, 2),
                            "documents_per_s": round(len(items) * (1 + args.hard_negatives) / train_s, 2),
                            "elapsed_s": round(time.time() - t_train, 1),
                            **({"val_per_repo": val_per_repo} if pooled else {})})
            print(f"[train] epoch {epoch}: loss {np.mean(losses):.4f} val {val}"
                  + (f" per repo {val_per_repo}" if pooled else ""), flush=True)
            if do_remine:
                history[-1]["remine"] = remine(epoch, corpora)
            del corpora
            if val["MRR@10"] > best:
                best, best_epoch, bad_epochs = val["MRR@10"], epoch, 0
                state = lora_state(model) if args.mode == "lora" else model.state_dict()
                save_file({k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in state.items()},
                          os.path.join(out_dir, "weights.safetensors"))
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[train] no improvement for {args.patience} epochs; stopping", flush=True)
                    break
        train_seconds = round(time.time() - t_train, 1)
        n_chunks = sum(len(rd["texts"]) for rd in R)
        # items = question-epochs (training questions x epochs run); the recipe's per-question counts are
        # spelled out so runs that stop at different epochs can be compared
        train_stage.counts(items=len(items) * len(history[1:]), question_epochs=len(items) * len(history[1:]),
                           questions=len(items), val_questions=n_val, epochs_run=len(history) - 1,
                           best_epoch=best_epoch, steps=step, symbols=sum(len(rd["syms"]) for rd in R),
                           chunks=n_chunks, trainable_params=n_trainable,
                           hard_negatives_per_question=args.hard_negatives,
                           documents_per_question=1 + args.hard_negatives,
                           in_batch_documents_per_step=args.batch * (1 + args.hard_negatives),
                           candidate_pool_per_question=int(np.median([len(c) for _, _, c in items])))
        train_stage.extra(history=history, stock_val=val0, best_val=history[best_epoch],
                          gpu_hours=round(train_seconds / 3600, 4) if gpu else 0.0)
        repo_meta = [{"repo": rd["cfg"]["name"], "train_questions": len(rd["train_rows"]),
                      "val_questions": len(rd["val_rows"]), "train_sha256": common.file_sha256(rd["train_path"]),
                      "val_sha256": common.file_sha256(rd["val_path"]),
                      "heldout_sha256": common.file_sha256(rd["held_path"]),
                      "symbols_sha256": common.file_sha256(os.path.join(rd["work"], "symbols.jsonl")),
                      "chunking": rd["stats"]._asdict(), "documents": len(rd["texts"]), "stock_val": rd["val0"]}
                     for rd in R]
        meta = {
            "base": args.base, "base_revision": base_revision, "mode": args.mode,
            "repo": stage_repo if pooled else R[0]["cfg"]["name"],
            "query_prefix": qp, "prefix": dp, "max_seq_length": limit,
            "questions": args.questions, "train_questions": len(items), "val_questions": n_val,
            **({"repos": repo_meta} if pooled else {k: v for k, v in repo_meta[0].items()
                                                     if k not in ("repo", "stock_val", "train_questions",
                                                                  "val_questions")}),
            "settings": {k: getattr(args, k) for k in ("epochs", "batch", "hard_negatives", "mine_depth", "lr",
                                                        "weight_decay", "warmup", "temperature", "seed", "patience",
                                                        "grad_checkpointing", "grad_cache", "cache_tokens",
                                                        "batch_tokens", "fn_rule", "fn_ratio", "fn_judge",
                                                        "stock_vectors", "limit_train", "limit_corpus", "pool",
                                                        "remine_every")},
            "lora": ({"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                      "targets": list(LORA_TARGETS), "layers": len(wrapped)} if args.mode == "lora" else None),
            "trainable_params": n_trainable, "best_epoch": best_epoch, "best_val": history[best_epoch],
            "stock_val": val0, "history": history, "train_seconds": train_seconds,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else None,
            "gpu": torch.cuda.get_device_name(0) if gpu else None,
            "checkpoint_bytes": os.path.getsize(os.path.join(out_dir, "weights.safetensors")),
        }
        with open(os.path.join(out_dir, "tuna-model.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        train_stage.outputs(os.path.join(out_dir, "weights.safetensors"), os.path.join(out_dir, "tuna-model.json"))
    print(f"[train] best epoch {best_epoch} val {history[best_epoch]}; {train_seconds}s, "
          f"peak {meta['peak_vram_gib']} GiB, checkpoint {meta['checkpoint_bytes'] / 2**20:.1f} MiB -> {out_dir}",
          flush=True)


def add_parser(sub):
    p = sub.add_parser("train", help="fine-tune an embedder on a training question set (full or LoRA)")
    p.add_argument("--questions", required=True, help="set under questions/ with .train/.val splits")
    p.add_argument("--name", required=True, help="output name under models/")
    p.add_argument("--base", default=DEFAULT_BASE, help="base model: from pins.toml, or any with --base-revision")
    p.add_argument("--pool", nargs="+", metavar="REPO",
                   help="pool these repos' training questions with --repo's (each repo keeps its own held-out list, "
                        "corpus, mining and validation); output under <out>/pooled/models/<name>")
    p.add_argument("--base-revision", help="full commit of --base when it is not in pins.toml")
    p.add_argument("--prefix", help="document prefix (default: the model's document prompt, as when scoring)")
    p.add_argument("--query-prefix", help="query prefix (default: the model's query prompt, as when scoring)")
    p.add_argument("--grad-cache", action="store_true",
                   help="GradCache contrastive step: same batch and negatives, bounded memory for long documents")
    p.add_argument("--cache-tokens", type=int, default=16384, help="--grad-cache: padded tokens per group")
    p.add_argument("--mode", choices=["full", "lora"], default="full")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3, help="stop after this many epochs without a better val MRR@10")
    p.add_argument("--batch", type=int, default=32, help="questions per step")
    p.add_argument("--hard-negatives", type=int, default=3, help="per question per step")
    p.add_argument("--mine-depth", type=int, default=30, help="stock top-k chunks to draw hard negatives from")
    p.add_argument("--remine-every", type=int, default=0,
                   help="ANCE-style: every N epochs re-mine hard negatives from the current checkpoint's index "
                        "(same depth, skips and stock judge); 0 = mine once from the stock model")
    p.add_argument("--fn-rule", choices=["judge", "own", "none"], default="judge",
                   help="false-negative filter: judge (one judge model for every trained model), own (the model's "
                        "own stock scores) or none")
    p.add_argument("--fn-ratio", type=float, default=0.95,
                   help="drop a candidate whose score is above this x the gold's score")
    p.add_argument("--fn-judge", default="qwen3-0.6b",
                   help="tuna embed tag of the judge's stock corpus vectors (its question encoder comes with them)")
    p.add_argument("--lr", type=float, help="default 2e-5 (full) or 2e-4 (lora)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="share of steps")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--batch-tokens", type=int, default=65536, help="padded tokens per inference batch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stock-vectors", default="lfm25-emb",
                   help="tuna embed tag of the stock model's corpus vectors, reused for mining (if it matches)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=0, help="smoke tests: use only this many questions")
    p.add_argument("--limit-corpus", type=int, default=0, help="smoke tests: small corpus around the golds")
    p.set_defaults(func=main)
    return p
