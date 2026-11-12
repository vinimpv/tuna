"""Home Assistant subset rule (transfer-only test repo), fixed before any question is written.

  .venv/bin/python scripts/ha-subset.py --components-symbols SYMBOLS_JSONL --core-symbols N [--write]

Rule: homeassistant/ minus homeassistant/components/ and homeassistant/generated/ (non-test files, the
same scope rules as the other repos), plus integrations (homeassistant/components/<name>/) drawn in a
seeded random order (sorted names, random.Random(SEED).shuffle) until the total symbol count first
reaches TARGET = 11,678, the median of our three training repos (Saleor). SYMBOLS_JSONL is an
extraction of every integration with the same extractor and scope rules. --write puts the chosen
integrations into repos/homeassistant.toml's include list.
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = 20261112
TARGET = 11678


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--components-symbols", required=True)
    ap.add_argument("--core-symbols", type=int, required=True)
    ap.add_argument("--clone", default=os.path.join(common.data_root(), "repos", "homeassistant"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    per = collections.Counter(json.loads(l)["file_path"].split("/")[2] for l in open(args.components_symbols))
    names = sorted(per)
    random.Random(SEED).shuffle(names)
    total, chosen = args.core_symbols, []
    for n in names:
        if total >= TARGET:
            break
        chosen.append(n)
        total += per[n]
    out = {"seed": SEED, "target": TARGET, "core_symbols": args.core_symbols, "integrations_available": len(per),
           "integrations_chosen": len(chosen), "integration_symbols": total - args.core_symbols,
           "expected_total": total, "chosen": sorted(chosen)}
    print(json.dumps(out))
    if args.write:
        p = os.path.join(ROOT, "repos", "homeassistant.toml")
        s = open(p).read()
        # core: every top-level entry of homeassistant/ except components/ and generated/ (files by exact
        # name, directories with /*), then the chosen integrations
        core = []
        for e in sorted(os.listdir(os.path.join(args.clone, "homeassistant"))):
            if e in ("components", "generated", "__pycache__"):
                continue
            full = os.path.join(args.clone, "homeassistant", e)
            core.append(f"homeassistant/{e}/*" if os.path.isdir(full) else f"homeassistant/{e}")
        inc = core + [f"homeassistant/components/{n}/*" for n in sorted(chosen)]
        body = "include = [\n" + "".join(f'  "{x}",\n' for x in inc) + "]"
        s = re.sub(r"include = \[[^\]]*\]", body, s, count=1)
        s = s.replace('  "homeassistant/components/*",\n', "")  # integrations come in through the include list
        s = s.replace('  "homeassistant/generated/*",                                   # generated tables\n', "")
        open(p, "w").write(s)


if __name__ == "__main__":
    main()
