"""Build src/tuna/data/english-words.txt.gz from SCOWL Hunspell dictionaries.

  python3 -I scripts/build-wordlist.py src/tuna/data/english-words.txt.gz \
      en_US.dic en_US.aff en_GB-ize.dic en_GB-ize.aff

Sources of the shipped list, from https://github.com/en-wl/wordlist/releases/tag/rel-2026.02.25
(SCOWL size 60):
  hunspell-en_US-2026.02.25.zip      sha256 ac8e73310e951d88c52c2cf2ba54ceaca34f8486a81630ac8a75dc5f931179f9
  hunspell-en_GB-ize-2026.02.25.zip  sha256 c5bdd92fc1e21da7503939fe4139f36f28f01c18ac35e2fc90b526a1b7bfb099
British spellings such as "catalogue" and "behaviour" are common in code and
questions, hence both. Every stem is expanded with its dictionary's prefix and
suffix rules, then kept if it is ASCII letters only and 3 or more long, lower-cased.
Licence: see src/tuna/data/NOTICE-english-words.
"""
import gzip
import hashlib
import re
import sys

# Two offensive words left out of the shipped list, identified by their
# sha256 so that the words themselves appear nowhere in this repository.
EXCLUDED_SHA256 = {
    "60f91a742458ae606572f40dd7df0a128e554acb023f6989fe5148872648fc56",
    "a0397adb8b2123d515dc07f72f1f9f3ebd9a3da4e6c4bf4336fa25bb23068828",
}


def parse_aff(path):
    rules = {}  # flag -> (kind, cross, [(strip, add, regex)])
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 4 and parts[0] in ("PFX", "SFX") and parts[2] in ("Y", "N"):
            kind, flag, cross, count = parts[0], parts[1], parts[2] == "Y", int(parts[3])
            entries = []
            for line in lines[i + 1:i + 1 + count]:
                _, _, strip, add, cond = line.split()[:5]
                strip = "" if strip == "0" else strip
                add = "" if add.split("/")[0] == "0" else add.split("/")[0]
                pat = ("^" + cond) if kind == "PFX" else (cond + "$")
                entries.append((strip, add, re.compile("" if cond == "." else pat)))
            rules[flag] = (kind, cross, entries)
            i += 1 + count
        else:
            i += 1
    return rules


def apply(word, kind, entries):
    out = []
    for strip, add, cond in entries:
        if cond.pattern and not cond.search(word):
            continue
        if kind == "SFX":
            if strip and not word.endswith(strip):
                continue
            out.append((word[:len(word) - len(strip)] if strip else word) + add)
        else:
            if strip and not word.startswith(strip):
                continue
            out.append(add + word[len(strip):])
    return out


def expand(stem, flags, rules):
    forms = {stem}
    prefixed = []
    for f in flags:
        if f in rules and rules[f][0] == "PFX":
            new = apply(stem, "PFX", rules[f][2])
            forms.update(new)
            if rules[f][1]:
                prefixed += new
    for f in flags:
        if f in rules and rules[f][0] == "SFX":
            forms.update(apply(stem, "SFX", rules[f][2]))
            if rules[f][1]:
                for p in prefixed:
                    forms.update(apply(p, "SFX", rules[f][2]))
    return forms


def main(out, *pairs):
    words = set()
    for dic, aff in zip(pairs[::2], pairs[1::2]):
        rules = parse_aff(aff)
        with open(dic, encoding="utf-8") as fh:
            next(fh)  # entry count
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                stem, _, flags = line.partition("/")
                for w in expand(stem, flags, rules):
                    w = w.lower()
                    if (len(w) >= 3 and re.fullmatch(r"[a-z]+", w)
                            and hashlib.sha256(w.encode()).hexdigest() not in EXCLUDED_SHA256):
                        words.add(w)
    with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
        # no name, no mtime in the header: byte-identical rebuilds
        fh.write(("\n".join(sorted(words)) + "\n").encode())
    print(f"{len(words)} words -> {out}")


if __name__ == "__main__":
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) < 4 or len(sys.argv) % 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
