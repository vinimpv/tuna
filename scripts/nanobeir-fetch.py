"""Fetch English NanoBEIR (LiquidAI/nanobeir-multilingual-extended) as JSONL, no parquet reader needed.

  python3 -I scripts/nanobeir-fetch.py [--out $TUNA_DATA/nanobeir] [--lang en]

Rows come from the Hugging Face datasets-server JSON API (the dataset's parquet
files, served as rows), 100 per request. Writes <out>/<lang>/<Subset>/{corpus,queries,qrels}.jsonl
and a manifest with row counts and the dataset revision. Skips files already
complete, so it can be rerun.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

DATASET = "LiquidAI/nanobeir-multilingual-extended"
API = "https://datasets-server.huggingface.co"
SUBSETS = ["NanoArguAna", "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", "NanoFiQA2018", "NanoHotpotQA",
           "NanoMSMARCO", "NanoNFCorpus", "NanoNQ", "NanoQuoraRetrieval", "NanoSCIDOCS", "NanoSciFact",
           "NanoTouche2020"]


def get(path, **params):
    url = f"{API}/{path}?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(30):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "tuna-nanobeir-fetch"}),
                                        timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:  # 429 rate limit: honour Retry-After, else back off
            last = e
            wait = e.headers.get("Retry-After") if e.headers else None
            time.sleep(float(wait) if wait and wait.replace(".", "", 1).isdigit() else min(60, 2 ** min(attempt, 6)))
        except Exception as e:  # noqa: BLE001 - timeouts: back off and retry
            last = e
            time.sleep(min(60, 2 ** min(attempt, 6)))
    raise RuntimeError(f"gave up on {url}: {last}")


def fetch(config, split, path):
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        d = get("rows", dataset=DATASET, config=config, split=split, offset=offset, length=100)
        total = d["num_rows_total"]
        rows += [r["row"] for r in d["rows"]]
        offset += 100
        time.sleep(1.0)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    os.replace(path + ".tmp", path)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(common.data_root(), "nanobeir"))
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    info = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}", timeout=60))
    manifest = {"dataset": DATASET, "revision": info.get("sha"), "last_modified": info.get("lastModified"),
                "lang": args.lang, "subsets": {}}
    for sub in SUBSETS:
        d = os.path.join(args.out, args.lang, sub)
        os.makedirs(d, exist_ok=True)
        counts = {}
        for config, split, name in ((f"{sub}_{args.lang}", "corpus", "corpus"), (f"{sub}_{args.lang}", "queries", "queries"),
                                    (f"{sub}_{args.lang}_qrels", "qrels", "qrels")):
            path = os.path.join(d, f"{name}.jsonl")
            if os.path.exists(path):
                counts[name] = sum(1 for _ in open(path, encoding="utf-8"))
            else:
                counts[name] = fetch(config, split, path)
        manifest["subsets"][sub] = counts
        print(sub, counts, flush=True)
    with open(os.path.join(args.out, args.lang, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)


if __name__ == "__main__":
    main()
