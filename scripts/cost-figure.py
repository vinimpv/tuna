"""Accuracy against cost: one stock and one tuned point per model and repo, from logged numbers only.

  .venv/bin/python scripts/cost-figure.py [--set claude-m3] [--outdir $TUNA_OUT/figures]

Per repo and model (LFM2.5-Embedding-350M, jina-code-embeddings-0.5b, Qwen3-Embedding-0.6B):
  accuracy          MRR@10 on --set (eval_<set>.json), stock row and tuned row (--tuned picks the
                    variant per model; default: the pre-registered headline recipe)
  indexing          the embed stage of that row: wall seconds and GPU energy (Wh)
  query latency     search-bench p50, CUDA and CPU (out/<repo>/search-bench_<tag>_<device>.json)
  cold repo -> tuned index   question writing (qwen-train1), judge pass, mining, tuning and
                    re-indexing: minutes and Wh, each from its ledger line(s) joined with the GPU
                    log (scripts/gpu-energy.py); question writing uses the GPU log too, since the
                    writer ran on the same card
Writes cost-<set>.json, cost-<set>.csv and cost-<set>.svg (MRR@10 against cold-to-index minutes,
log x axis; stock at the indexing cost alone, an arrow to the tuned point).
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tuna import common, metrics  # noqa: E402

spec = importlib.util.spec_from_file_location("gpu_energy", os.path.join(HERE, "gpu-energy.py"))
gpu_energy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_energy)

MODELS = {"LFM2.5-350M": ("LiquidAI/LFM2.5-Embedding-350M", "lfm25-emb"),
          "jina-0.5B": ("jinaai/jina-code-embeddings-0.5b", "jina-0.5b"),
          "Qwen3-0.6B": ("Qwen/Qwen3-Embedding-0.6B", "qwen3-0.6b")}
# Locked variants (pre-registered variant rule, resolved on Saleor validation): LoRA for all three.
HEADLINE = {"LFM2.5-350M": "lfm25-ft-lora", "jina-0.5B": "jina-0.5b-ft-lora", "Qwen3-0.6B": "qwen3-0.6b-ft-lora"}
COLORS = {"LFM2.5-350M": "#c2410c", "jina-0.5B": "#1d4ed8", "Qwen3-0.6B": "#15803d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--repos", nargs="+", default=["saleor", "zulip", "posthog"])
    ap.add_argument("--set", default="claude-m3")
    ap.add_argument("--tuned", nargs="*", default=[], help="model=tag overrides, e.g. LFM2.5-350M=lfm25-ft-full")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--utc-offset-hours", type=float, default=0.0,
                    help="the GPU log's local time minus UTC, in hours (scripts/gpu-energy.py); no log: energy None")
    args = ap.parse_args()
    tuned = dict(HEADLINE, **dict(kv.split("=", 1) for kv in args.tuned))
    outdir = args.outdir or os.path.join(args.out, "figures")
    os.makedirs(outdir, exist_ok=True)
    md = metrics.metrics_dir()
    samples = gpu_energy.load_samples(os.path.join(md, "gpu-1s.csv"), args.utc_offset_hours)
    times = [s[0] for s in samples]
    ledger = [json.loads(l) for l in open(os.path.join(md, "ledger.jsonl"))]
    # local RTX 5090 lines only: runs on rented machines never enter the cost or speed numbers
    ok = [l for l in ledger if l.get("status") == "ok" and l.get("start_utc") and l.get("end_utc")
          and metrics.is_local(l)]

    def energy(line):
        w = gpu_energy.window(samples, times, gpu_energy.parse_iso(line["start_utc"]), gpu_energy.parse_iso(line["end_utc"]))
        return w.get("energy_wh"), w.get("coverage", 0.0)

    def cost(lines):
        """(minutes, Wh, lowest GPU-log coverage). Wh is None when any part is less than half covered by
        the GPU log (stages before the log started): a partial sum would understate it."""
        mins = sum(l["wall_seconds"] for l in lines) / 60
        wh, cov = 0.0, []
        for l in lines:
            e, c = energy(l)
            wh += e or 0.0
            if l["wall_seconds"] >= 10:  # a few-second stage has too few 1-s samples to judge coverage
                cov.append(c)
        low = min(cov) if cov else 1.0
        return round(mins, 2), (round(wh, 2) if low >= 0.5 else None), round(low, 2)

    def last(pred):
        hits = [l for l in ok if pred(l)]
        return hits[-1] if hits else None

    rows = []
    for repo in args.repos:
        ev_path = os.path.join(args.out, repo, f"eval_{args.set}.json")
        if not os.path.exists(ev_path):
            continue
        res = json.load(open(ev_path))["results"]
        qline = last(lambda l: l["stage"] == "questions" and l.get("repo") == repo
                     and l["config"].get("name") == "qwen-train1")
        jline = last(lambda l: l["stage"] == "judge-encode" and l.get("repo") == repo
                     and l["config"].get("questions") == "qwen-train1")
        for label, (model_id, stock_tag) in MODELS.items():
            for kind, tag in (("stock", stock_tag), ("tuned", tuned[label])):
                r = res.get(f"{tag}_{args.set}")
                if r is None:
                    continue
                emb = last(lambda l: l["stage"] == "embed" and l.get("repo") == repo
                           and (l["config"].get("tag") or l["extra"].get("tag")) == tag)
                idx_min, idx_wh, idx_cov = cost([emb]) if emb else (None, None, None)
                lat = {}
                for dev in ("cuda", "cpu"):
                    p = os.path.join(args.out, repo, f"search-bench_{tag}_{dev}.json")
                    if os.path.exists(p):
                        lat[dev] = json.load(open(p))["query_ms"]["p50"]
                row = {"repo": repo, "model": label, "kind": kind, "tag": tag, "mrr@10": r["mrr@10"],
                       "hit@10": r["hit@10"], "index_minutes": idx_min, "index_wh": idx_wh,
                       "query_p50_ms_cuda": lat.get("cuda"), "query_p50_ms_cpu": lat.get("cpu")}
                if kind == "stock":
                    row.update(cold_minutes=idx_min, cold_wh=idx_wh, cold_parts={"index": [idx_min, idx_wh]})
                else:
                    tr = last(lambda l: l["stage"] == "train" and l.get("repo") == repo and l["config"].get("name") == tag)
                    mi = None
                    if tr:
                        before = [l for l in ok if l["stage"] == "mine" and l.get("repo") == repo
                                  and l.get("model") == model_id and l["end_utc"] <= tr["start_utc"]]
                        mi = before[-1] if before else None
                    parts = {"questions": qline, "judge": jline, "mine": mi, "train": tr, "index": emb}
                    row["cold_parts"] = {k: (list(cost([v])[:2]) if v else None) for k, v in parts.items()}
                    have = [v for v in parts.values() if v]
                    row["cold_minutes"], row["cold_wh"], row["cold_min_coverage"] = cost(have)
                    row["cold_missing"] = [k for k, v in parts.items() if not v]
                rows.append(row)
    base = os.path.join(outdir, f"cost-{args.set}")
    with open(base + ".json", "w") as fh:
        json.dump({"set": args.set, "tuned": tuned, "rows": rows}, fh, indent=1)
    cols = ["repo", "model", "kind", "tag", "mrr@10", "hit@10", "index_minutes", "index_wh", "query_p50_ms_cuda",
            "query_p50_ms_cpu", "cold_minutes", "cold_wh"]
    with open(base + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_svg(base + ".svg", rows, args.set)
    print(f"{'repo':8s} {'model':12s} {'kind':6s} {'MRR@10':>7s} {'index min':>9s} {'index Wh':>8s} {'p50 cuda':>8s} "
          f"{'p50 cpu':>8s} {'cold min':>9s} {'cold Wh':>8s}")
    for r in rows:
        f = lambda v, d=1: "" if v is None else f"{v:.{d}f}"  # noqa: E731
        print(f"{r['repo']:8s} {r['model']:12s} {r['kind']:6s} {r['mrr@10']:7.3f} {f(r['index_minutes'],2):>9s} "
              f"{f(r['index_wh'],2):>8s} {f(r['query_p50_ms_cuda']):>8s} {f(r['query_p50_ms_cpu']):>8s} "
              f"{f(r['cold_minutes']):>9s} {f(r['cold_wh']):>8s}")
    print(f"wrote {base}.json/.csv/.svg")


def write_svg(path, rows, set_name):
    pts = [r for r in rows if r.get("cold_minutes")]
    if not pts:
        return
    W, H, L, B, T, Rm = 760, 460, 70, 50, 30, 170
    xs = [math.log10(r["cold_minutes"]) for r in pts]
    xmin, xmax = math.floor(min(xs)), math.ceil(max(xs))
    xmax = max(xmax, xmin + 1)
    ymin, ymax = 0.0, 1.0

    def X(m):
        return L + (math.log10(m) - xmin) / (xmax - xmin) * (W - L - Rm)

    def Y(v):
        return H - B - (v - ymin) / (ymax - ymin) * (H - B - T)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for e in range(xmin, xmax + 1):
        x = X(10 ** e)
        out.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x:.1f}" y="{H - B + 18}" text-anchor="middle">{10 ** e:g}</text>')
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = Y(v)
        out.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - Rm}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    out.append(f'<text x="{(L + W - Rm) / 2:.0f}" y="{H - 12}" text-anchor="middle">minutes from a cold repo to a '
               f'searchable index (log scale)</text>')
    out.append(f'<text x="18" y="{(T + H - B) / 2:.0f}" transform="rotate(-90 18 {(T + H - B) / 2:.0f})" '
               f'text-anchor="middle">MRR@10 on {set_name}</text>')
    marks = {"saleor": "circle", "zulip": "rect", "posthog": "diamond"}
    for r in pts:
        if r["kind"] != "tuned":
            continue
        s = next((p for p in pts if p["kind"] == "stock" and p["repo"] == r["repo"] and p["model"] == r["model"]), None)
        if s:
            out.append(f'<line x1="{X(s["cold_minutes"]):.1f}" y1="{Y(s["mrr@10"]):.1f}" x2="{X(r["cold_minutes"]):.1f}" '
                       f'y2="{Y(r["mrr@10"]):.1f}" stroke="{COLORS[r["model"]]}" stroke-opacity="0.5"/>')
    for r in pts:
        x, y, c = X(r["cold_minutes"]), Y(r["mrr@10"]), COLORS[r["model"]]
        fill = c if r["kind"] == "tuned" else "white"
        m = marks.get(r["repo"], "circle")
        if m == "circle":
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{c}" stroke-width="2"/>')
        elif m == "rect":
            out.append(f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" fill="{fill}" stroke="{c}" stroke-width="2"/>')
        else:
            out.append(f'<polygon points="{x:.1f},{y - 6:.1f} {x + 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} {x - 6:.1f},{y:.1f}" '
                       f'fill="{fill}" stroke="{c}" stroke-width="2"/>')
    ly = T + 10
    for label, c in COLORS.items():
        out.append(f'<rect x="{W - Rm + 20}" y="{ly}" width="12" height="12" fill="{c}"/><text x="{W - Rm + 38}" '
                   f'y="{ly + 10}">{label}</text>')
        ly += 20
    for repo, m in marks.items():
        out.append(f'<text x="{W - Rm + 20}" y="{ly + 10}">{m}: {repo}</text>')
        ly += 18
    out.append(f'<text x="{W - Rm + 20}" y="{ly + 10}">hollow: stock, filled: tuned</text>')
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out))


if __name__ == "__main__":
    main()
