"""Copy tuned checkpoints for release, each with its base model's licence beside it.

  python3 -I scripts/export-weights.py --licenses DIR [--out $TUNA_OUT] DEST REPO/ROW [REPO/ROW ...]

For each REPO/ROW, reads $TUNA_OUT/REPO/models/ROW/tuna-model.json for the base model and its
revision ("base", "base_revision"), and writes DEST/REPO/ROW/ with:
  weights.safetensors, tuna-model.json   the checkpoint as trained (a LoRA adapter or full weights)
  every file of DIR/<org>--<name>@<revision>/   the base model's licence (LICENSE, and NOTICE if any),
                                         as published in the base model's repo at that revision
  NOTICE.md                              which base model and revision this derives from, how it was
                                         modified, and that the base model's licence applies
The licence files are not shipped with tuna: download them from the base model's Hugging Face repo
at the pinned revision and put them under DIR first. A checkpoint whose base has no licence
directory stops the export (exit 1) with the list of what is missing; nothing is half-written.
Which checkpoints to release is the caller's choice; this script only packages them.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tuna import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=common.default_out())
    ap.add_argument("--licenses", required=True, help="directory of <org>--<name>@<revision>/ licence folders")
    ap.add_argument("dest")
    ap.add_argument("specs", nargs="+", metavar="REPO/ROW")
    args = ap.parse_args()
    plan, missing = [], {}
    for spec in args.specs:
        repo, _, row = spec.partition("/")
        src = os.path.join(args.out, repo, "models", row)
        meta_path = os.path.join(src, "tuna-model.json")
        if not row or not os.path.exists(meta_path):
            print(f"export-weights: no checkpoint {spec} ({meta_path})", file=sys.stderr)
            sys.exit(1)
        meta = json.load(open(meta_path))
        base, rev = meta["base"], meta.get("base_revision")
        if not rev:
            print(f"export-weights: {spec} records no base_revision; cannot pick its licence", file=sys.stderr)
            sys.exit(1)
        lic = os.path.join(args.licenses, f"{base.replace('/', '--')}@{rev}")
        if not os.path.isfile(os.path.join(lic, "LICENSE")):
            missing[lic] = f"https://huggingface.co/{base}/tree/{rev}"
        plan.append((spec, repo, row, src, meta, base, rev, lic))
    if missing:
        print("export-weights: base-model licence files missing; fetch each from the base model's repo at "
              "the pinned revision:", file=sys.stderr)
        for lic, url in sorted(missing.items()):
            print(f"  {lic}/LICENSE   <- {url}", file=sys.stderr)
        sys.exit(1)
    for spec, repo, row, src, meta, base, rev, lic in plan:
        d = os.path.join(args.dest, repo, row)
        os.makedirs(d, exist_ok=False)
        for f in ("weights.safetensors", "tuna-model.json"):
            shutil.copy2(os.path.join(src, f), d)
        for f in sorted(os.listdir(lic)):
            shutil.copy2(os.path.join(lic, f), d)
        kind = "a LoRA adapter for" if meta.get("mode") == "lora" else "a fully fine-tuned copy of"
        with open(os.path.join(d, "NOTICE.md"), "w") as fh:
            fh.write(f"# {repo}/{row}\n\n"
                     f"This checkpoint is {kind} [{base}](https://huggingface.co/{base}) at revision `{rev}`, "
                     f"modified by fine-tuning for code search on questions about the {repo} repository "
                     f"(training set `{meta.get('questions')}`; settings in `tuna-model.json`).\n\n"
                     f"It is a derivative of that model and is distributed under the base model's licence, "
                     f"included here as published in the base model's repository at that revision "
                     f"({', '.join(f'`{f}`' for f in sorted(os.listdir(lic)))}). "
                     f"The MIT licence of the tuna code does not apply to these weights.\n")
        print(f"export-weights: {spec} -> {d} (base {base}@{rev[:12]}, {meta.get('mode')})")


if __name__ == "__main__":
    main()
