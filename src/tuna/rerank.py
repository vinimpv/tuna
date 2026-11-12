"""Rerank the top-N of an existing rank file with a Qwen3-Reranker, then RRF-fuse.

Two backends score the same prompt (the Qwen3-Reranker yes/no format):
  served  vLLM /rerank on a model launched as
            vllm serve Qwen/Qwen3-Reranker-0.6B --runner pooling \\
              --hf-overrides '{"architectures":["Qwen3ForSequenceClassification"],
                               "classifier_from_token":["no","yes"],
                               "is_original_qwen3_reranker":true}'
  hf      transformers on the local GPU, p(yes) from the last-token logits
Writes ranks/<name>.jsonl (reranker order) and ranks/<name>_fused.jsonl
(RRF of embedder position and reranker position, K=10). Below the top-N the
input order is kept, so hit@k past N equals the input ranking.
"""
import os
import time

from . import common

RR_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
             'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
             '<|im_end|>\n<|im_start|>user\n')
RR_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
DEFAULT_INSTRUCTION = ("Given a question about behavior in {desc}, judge whether this code symbol "
                       "is the one that implements the behavior the question asks about.")
RRF_K = 10


def instruction_for(cfg, override):
    if override:
        return override
    return cfg.get("rerank_instruction") or DEFAULT_INSTRUCTION.format(desc=cfg.get("description", "this codebase"))


def served_scorer(url, model, instruction, api_key):
    query_head = f"{RR_PREFIX}<Instruct>: {instruction}\n<Query>: "

    def score(question, docs):
        resp = common.post_json(url, {"model": model, "query": query_head + question + "\n",
                                      "documents": [f"<Document>: {d}{RR_SUFFIX}" for d in docs]},
                                api_key=api_key)
        out = [-1e9] * len(docs)
        for item in resp["results"]:
            out[item["index"]] = item["relevance_score"]
        return out
    return score


def hf_scorer(model_path, revision, instruction, batch, max_length):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import pinned
    pinned.refuse_unpinned(model_path)  # remote-code models load only through tuna.pinned
    tok = AutoTokenizer.from_pretrained(model_path, revision=revision, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision,
                                                 dtype=torch.bfloat16).cuda().eval()
    yes_id, no_id = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok(RR_PREFIX, add_special_tokens=False)["input_ids"]
    suf = tok(RR_SUFFIX, add_special_tokens=False)["input_ids"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def score(question, docs):
        out = []
        with torch.no_grad():
            for s in range(0, len(docs), batch):
                bodies = [f"<Instruct>: {instruction}\n<Query>: {question}\n<Document>: {d}"
                          for d in docs[s:s + batch]]
                enc = tok(bodies, add_special_tokens=False, truncation=True,
                          max_length=max_length - len(pre) - len(suf))["input_ids"]
                rows = [pre + e + suf for e in enc]
                n = max(len(r) for r in rows)
                ids = torch.full((len(rows), n), pad, dtype=torch.long)
                attn = torch.zeros((len(rows), n), dtype=torch.long)
                for i, r in enumerate(rows):
                    ids[i, n - len(r):] = torch.tensor(r)
                    attn[i, n - len(r):] = 1
                logits = model(input_ids=ids.cuda(), attention_mask=attn.cuda()).logits[:, -1]
                pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
                out.extend(torch.softmax(pair, dim=1)[:, 1].cpu().tolist())
        return out
    return score


def main(args):
    cfg = common.load_config(args.repo)
    work = common.workdir(args.out, cfg)
    syms = {s["fqn"]: s for s in common.load_symbols(work)}
    qs = {q["id"]: q for q in common.load_questions(args.questions)}
    src = common.read_jsonl(os.path.join(work, "ranks", f"{args.input}.jsonl"))
    instruction = instruction_for(cfg, args.instruction)
    print(f"[rerank] instruction: {instruction}", flush=True)
    if args.backend == "served":
        score = served_scorer(args.url, args.model, instruction,
                              os.environ.get(args.api_key_env) if args.api_key_env else None)
    else:
        score = hf_scorer(args.model, args.revision, instruction, args.batch, args.max_length)
    rr_rows, fused_rows, t0 = [], [], time.time()
    for n, row in enumerate(src):
        q = qs[row["id"]]
        cand, tail = row["ranked"][:args.topn], row["ranked"][args.topn:]
        sc = score(q["question"], [common.symbol_doc(syms[f], args.body_cap) for f in cand])
        order = sorted(range(len(cand)), key=lambda j: -sc[j])
        pos = {j: r for r, j in enumerate(order)}
        fused = sorted(range(len(cand)), key=lambda j: -(1 / (RRF_K + j + 1) + 1 / (RRF_K + pos[j] + 1)))
        rr_rows.append(common.rank_row(q, [cand[j] for j in order] + tail,
                                       [sc[j] for j in order] + [-1e9] * len(tail)))
        fused_rows.append(common.rank_row(q, [cand[j] for j in fused] + tail,
                                          [1 / (RRF_K + j + 1) + 1 / (RRF_K + pos[j] + 1) for j in fused]
                                          + [0.0] * len(tail)))
        if (n + 1) % 25 == 0:
            print(f"[rerank] {n + 1}/{len(src)} ({time.time() - t0:.0f}s)", flush=True)
    name = args.name or f"{args.input}__rr"
    common.write_jsonl(os.path.join(work, "ranks", f"{name}.jsonl"), rr_rows)
    common.write_jsonl(os.path.join(work, "ranks", f"{name}_fused.jsonl"), fused_rows)
    print(f"[rerank] wrote ranks/{name}.jsonl and ranks/{name}_fused.jsonl", flush=True)


def add_parser(sub):
    p = sub.add_parser("rerank", help="rerank top-N of a rank file (served vLLM or local transformers)")
    p.add_argument("--questions", required=True)
    p.add_argument("--input", required=True, help="rank file name under ranks/, without .jsonl")
    p.add_argument("--backend", choices=["served", "hf"], default="served")
    p.add_argument("--model", required=True, help="served model name, or HF id / local path for --backend hf")
    p.add_argument("--revision", default=None, help="HF revision for --backend hf")
    p.add_argument("--url", default=os.environ.get("TUNA_RR_URL", "http://127.0.0.1:8001/rerank"))
    p.add_argument("--topn", type=int, default=50)
    p.add_argument("--body-cap", type=int, default=4000, help="character cap per document (0 = none)")
    p.add_argument("--instruction", help="override the config/default instruction")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--api-key-env")
    p.add_argument("--name")
    p.set_defaults(func=main)
    return p
