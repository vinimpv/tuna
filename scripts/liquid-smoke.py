"""GPU smoke test for the pinned Liquid models, one model per process:

  .venv/bin/python scripts/liquid-smoke.py embedding   # LFM2.5-Embedding-350M
  .venv/bin/python scripts/liquid-smoke.py colbert     # LFM2.5-ColBERT-350M
  .venv/bin/python scripts/liquid-smoke.py lfm3        # LFM3-1.7B, one chat turn with a tool

The encoders embed the same 10 Saleor symbols and query as scripts/embed-smoke.py
and report the gold symbol's rank. LFM3 generates one assistant turn with one
tool defined and checks that the reply holds a well-formed tool call.
"""
import json
import re
import sys
import time

GOLD = "saleor.checkout.tasks.delete_expired_checkouts"
QUERY = "How long can an anonymous shopping cart sit untouched before a background job removes it?"
IDS = {"embedding": "LiquidAI/LFM2.5-Embedding-350M", "colbert": "LiquidAI/LFM2.5-ColBERT-350M",
       "lfm3": "LiquidAI/LFM3-1.7B"}


def saleor_docs(n=10, cap=4000):
    from tuna import common
    cfg = common.load_config("saleor")
    syms = common.load_symbols(common.workdir(common.default_out(), cfg))
    picked = syms[:n - 1] + [s for s in syms if s["fqn"] == GOLD][:1]
    return picked, [common.symbol_doc(s, cap) for s in picked]


def gold_rank(scores, picked):
    order = sorted(range(len(picked)), key=lambda i: -float(scores[i]))
    return 1 + [picked[i]["fqn"] for i in order].index(GOLD), picked[order[0]]["fqn"]


def encoders(which, torch, pinned):
    picked, docs = saleor_docs()
    t0 = time.time()
    model = pinned.load(IDS[which], device="cuda")
    load_s = time.time() - t0
    tok = model.tokenizer
    lengths = [len(tok(d, add_special_tokens=True)["input_ids"]) for d in docs]
    if which == "embedding":
        limit = model.max_seq_length
        D = model.encode(docs, prompt_name="document", normalize_embeddings=True, convert_to_tensor=True)
        Q = model.encode([QUERY], prompt_name="query", normalize_embeddings=True, convert_to_tensor=True)
        scores = (Q @ D.T)[0].float().cpu()
        shape = f"dim {D.shape[1]}"
        # padding effect: batched (padded) vs one document at a time (no padding)
        D1 = torch.cat([model.encode([d], prompt_name="document", normalize_embeddings=True, convert_to_tensor=True)
                        for d in docs])
        drift = (D * D1).sum(-1).float().cpu()
        extra = {"batched_vs_single_cosine": [round(float(x), 5) for x in drift],
                 "gold_rank_single": gold_rank((Q @ D1.T)[0].float().cpu(), picked)[0]}
    else:
        limit = model.max_seq_length
        Q = model.encode_query([QUERY])
        D = model.encode_document(docs)
        scores = model.similarity(Q, D)[0].float().cpu()
        shape = f"query {tuple(Q[0].shape)}, doc token-vectors {[int(d.shape[0]) for d in D]}"
        extra = {}
    rank, top1 = gold_rank(scores, picked)
    print(json.dumps({
        "model": IDS[which], "load_s": round(load_s, 1), "shape": shape, "max_seq_length": limit,
        "doc_tokens": lengths, "docs_over_limit": sum(n > limit for n in lengths),
        "gold_rank_of_10": rank, "top1": top1, "scores": [round(float(s), 4) for s in scores],
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)} | extra, indent=1))


TOOLS = [{"type": "function", "function": {
    "name": "search_code",
    "description": "Search the Saleor repository for code matching a natural-language query.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "what the code does"},
        "path_glob": {"type": "string", "description": "optional glob to restrict the search, e.g. saleor/checkout/*"}},
        "required": ["query"]}}}]
CALL = re.compile(r"<tool_call>\s*<function=(?P<name>[^\n>]+)>(?P<body>.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?</parameter>", re.S)


def lfm3(torch, pinned):
    t0 = time.time()
    tok, model = pinned.load(IDS["lfm3"], device="cuda")
    load_s = time.time() - t0
    messages = [{"role": "user", "content": "In the Saleor codebase, find the background job that deletes "
                                            "expired checkouts. Use the search tool."}]
    out = {"model": IDS["lfm3"], "load_s": round(load_s, 1)}
    for thinking in (False, True):
        enc = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, enable_thinking=thinking,
                                      return_tensors="pt", return_dict=True).to("cuda")
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=8, do_sample=False)  # warm-up: Triton JIT, CUDA graphs
            torch.cuda.synchronize()
            t = time.time()
            gen = model.generate(**enc, max_new_tokens=768, do_sample=False)
            torch.cuda.synchronize()
            dt = time.time() - t
        new = gen[0, n_in:]
        text = tok.decode(new, skip_special_tokens=False)
        calls = []
        for m in CALL.finditer(text):
            calls.append({"name": m["name"].strip(), "arguments": {p["key"]: p["value"] for p in PARAM.finditer(m["body"])}})
        ok = (len(calls) >= 1 and all(c["name"] == "search_code" and "query" in c["arguments"]
                                      and set(c["arguments"]) <= {"query", "path_glob"} for c in calls))
        out[f"thinking={thinking}"] = {
            "prompt_tokens": n_in, "new_tokens": int(new.shape[0]), "seconds": round(dt, 2),
            "tokens_per_s": round(new.shape[0] / dt, 1), "tool_calls": calls, "well_formed": ok,
            "ended_with_eos": bool(new[-1].item() == tok.eos_token_id), "raw": text}
    out["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(out, indent=1))


def main():
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        sys.exit(0)
    import torch

    from tuna import pinned
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in IDS:
        sys.exit(__doc__)
    print(f"GPU: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    if which == "lfm3":
        lfm3(torch, pinned)
    else:
        encoders(which, torch, pinned)


if __name__ == "__main__":
    main()
