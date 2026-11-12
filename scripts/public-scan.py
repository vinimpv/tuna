"""Scan a directory tree for anything that must not be published.

  python3 -I scripts/public-scan.py DIR [--allow-email ADDR_OR_@DOMAIN ...]

Checks every file (gzip files decompressed; other binaries by path only) for:
  home-path      absolute home directories (under /home, /Users, or root's home)
  email          addresses not on the allow list (bots, noreply, example domains)
  private-ip     RFC 1918, CGNAT/Tailscale (100.64/10) and link-local IPv4 addresses (version strings
                 such as nvidia-curand's are skipped)
  secret         private keys and token shapes (GitHub, Hugging Face, Anthropic, OpenAI, AWS, Slack),
                 and key/token/secret/password assignments with a literal value
  data-path      files under private data folders or with secret-looking names
                 (tuna-data/, out/, metrics/, cache/, vectors/, models/, ranks/, .env, *.pem, id_rsa, ...)
  private-name   names of this project's private machines, work trees and infrastructure (RELEASE_PATTERNS:
                 the zero-hit patterns of the pre-release audit), in every text file
Prints one line per finding (path:line: kind: excerpt) and exits 1 when there is any.
"""
import argparse
import gzip
import os
import re
import sys

ALLOWED_EMAILS = {"git@github.com"}
# third-party credits that a shipped licence notice must keep verbatim: (file, address)
ALLOWED_IN_FILE = {("src/tuna/data/NOTICE-english-words", "biljir" + "@pobox.com")}  # split so this line passes
# The author's contact address, published only in the README, pyproject.toml, LICENSE and the summary PDF;
# anywhere else it is still a finding. Split as above.
_CONTACT = "viniciusvmp" + "@poli.ufrj.br"
ALLOWED_IN_FILE |= {("README.md", _CONTACT), ("pyproject.toml", _CONTACT), ("LICENSE", _CONTACT),
                    ("summary.pdf", _CONTACT)}
# Example and test addresses quoted from the benchmark repos' own code and issues (issue text, agent answers):
# allowed only inside the released data folder (--release-data), each named exactly. The last one is the Python
# decorator "@app.task" right after a name ending in n, not an address.
UPSTREAM_EXAMPLES = {"delete-test" + "@zulip.com", "user7" + "@zulipdev.com", "user14" + "@zulipdev.com",
                     "test" + "@posthog.com", "john" + "@doe.com", "n" + "@app.task"}
# Documentation and example addresses quoted from the benchmark repos' issues and code: allowed only inside the
# released data (issue text is test input and stays as written)
UPSTREAM_EXAMPLE_IPS = {"10.0" + ".0.0", "192.168" + ".0.1", "192.168" + ".1.1", "169.254" + ".169.254"}  # split so this line passes
# Fake tokens quoted from the benchmark repos' tests (in agent answers): allowed only inside the released data
UPSTREAM_FAKE_SECRETS = {"xoxb-" + "invalid-token"}
# Home directories quoted from the benchmark repos' issues (tracebacks, deploy paths) and agent answers: allowed only
# inside the released data
UPSTREAM_HOMES = {"/home/" + "zulip", "/home/" + "vagrant", "/Users/" + "marcin"}
# A path that only exists as a fixture in a test: (file, home prefix)
HOME_ALLOWED_IN_FILE = {("tests/test_cc_bench.py", "/home/x")}
ALLOWED_DOMAINS = {"users.noreply.github.com", "example.com", "example.org", "example.net", "example.invalid",
                   "localhost"}

HOME = re.compile(r"(?<![\w.])(/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|/root)(?=/)")
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
IPV4 = re.compile(r"(?<![\w.-])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.-])")
VERSION_LINE = re.compile(r"(?i)\bversion\b\s*[:=]|==|>=|<=|~=")
SECRETS = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai-style key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}")),
    ("aws key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("assigned secret", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)\b"
                                   r"\s*[:=]\s*[\"'][^\"'\s$]{8,}[\"']")),
]
DATA_DIRS = {"tuna-data", "out", "metrics", "cache", "vectors", "models", "ranks", "history", "hf", "logs-issues"}
# Zero-hit patterns (pre-release audit, section F): work trees, private machines and services, loopback ports of
# private servers. Each is assembled from parts so that this file does not match itself.
_J = "".join
RELEASE_PATTERNS = re.compile("|".join([
    _J([r"\.t3", "-fork"]), _J(["cc", "-section"]), _J(["claude-sets", "-v2"]), _J(["eval", "-staging"]),
    _J(["tuna", "-public"]), _J([r"pi-1\.2", r"\.0"]), _J([r"\.pi", "-research"]), _J(["/home", "/dev"]),
    _J(["~/", "opt/"]), _J(["~/tuna", "/data"]), _J(["/~", "/"]), _J(["t3", "code"]),
    _J(["<private", "-host>"]), _J(["<cont", "ainer>"]), _J(["run", "pod"]), _J(["tuna-", "pod-"]),
    _J(["pod", "-split"]), _J(["pod", "gate"]), _J(["gpu", "-switch"]), _J(["gpu", "-request"]), _J(["gpu", "-hold"]),
    _J(["ninfer-batch", "-ready"]), _J(["pod", "_job"]), _J(["pod", "_commit"]), _J(["tuna", "_commit"]),
    _J([r"127\.0\.0\.1:", "(?:1[0-9]{4}|88[0-9]{2})"]), _J(['"machine": ?"g', 'pu"']), _J(["gpu", " box"]),
    _J(["UTC", "-3"]), _J(["hours", "=-3"]),
]))
SECRET_NAMES = re.compile(r"(^\.env(\..*)?$|\.pem$|\.key$|\.p12$|\.pfx$|^id_(rsa|ed25519|ecdsa)(\.pub)?$|"
                          r"^\.netrc$|^\.npmrc$|^\.pypirc$|^credentials(\.json)?$|^models\.json$|^ledger\.jsonl$)")


def private_ip(m):
    a, b, c, d = (int(x) for x in m.groups())
    if max(a, b, c, d) > 255:
        return False
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or \
        (a == 100 and 64 <= b <= 127) or (a == 169 and b == 254)


def email_allowed(addr, extra):
    addr = addr.lower()
    dom = addr.split("@", 1)[1]
    allowed = ALLOWED_EMAILS | {e.lower() for e in extra if not e.startswith("@")}
    domains = ALLOWED_DOMAINS | {e[1:].lower() for e in extra if e.startswith("@")}
    return addr in allowed or dom in domains or any(dom.endswith("." + d) for d in domains)


def read_text(path):
    raw = open(path, "rb").read()
    if path.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return None
    if b"\0" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def scan(root, extra_emails, release_data=None):
    findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            parts = rel.split(os.sep)
            released = bool(release_data) and rel.replace(os.sep, "/").startswith(release_data.rstrip("/") + "/")
            if DATA_DIRS & set(parts[:-1]) and not released:
                findings.append((rel, 0, "data-path", "file under a private data folder"))
            if SECRET_NAMES.search(fn) and not (released and fn == "ledger.jsonl"):
                findings.append((rel, 0, "data-path", "secret-looking file name"))
            if os.path.islink(path):
                continue
            text = read_text(path)
            if text is None:
                continue
            for i, line in enumerate(text.split("\n"), 1):
                for m in HOME.finditer(line):
                    if (rel.replace(os.sep, "/"), m.group(1)) in HOME_ALLOWED_IN_FILE:
                        continue
                    if released and m.group(1) in UPSTREAM_HOMES:
                        continue
                    findings.append((rel, i, "home-path", line.strip()[:120]))
                for m in EMAIL.finditer(line):
                    if (rel.replace(os.sep, "/"), m.group(0).lower()) in ALLOWED_IN_FILE:
                        continue
                    if released and m.group(0).lower() in UPSTREAM_EXAMPLES:
                        continue
                    if not email_allowed(m.group(0), extra_emails):
                        findings.append((rel, i, "email", m.group(0)))
                for m in IPV4.finditer(line):
                    if released and m.group(0) in UPSTREAM_EXAMPLE_IPS:
                        continue
                    if private_ip(m) and not VERSION_LINE.search(line):
                        findings.append((rel, i, "private-ip", m.group(0)))
                for m in RELEASE_PATTERNS.finditer(line):
                    findings.append((rel, i, "private-name", m.group(0)))
                for name, rx in SECRETS:
                    hit = rx.search(line)
                    if hit and released and hit.group(0) in UPSTREAM_FAKE_SECRETS and \
                            not any(m.group(0) not in UPSTREAM_FAKE_SECRETS for m in rx.finditer(line)):
                        continue
                    if hit:
                        findings.append((rel, i, "secret", f"{name}: {line.strip()[:80]}"))
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--allow-email", nargs="*", default=[], help="extra allowed addresses, or @domain")
    ap.add_argument("--release-data", metavar="REL", help="a folder of DIR that is released data on purpose: no data-path "
                    "finding for its files (their contents are still scanned); upstream example addresses allowed there")
    args = ap.parse_args()
    findings = scan(args.dir, args.allow_email, args.release_data)
    for rel, line, kind, what in findings:
        print(f"{rel}:{line}: {kind}: {what}")
    kinds = {}
    for f in findings:
        kinds[f[2]] = kinds.get(f[2], 0) + 1
    print(f"public-scan: {len(findings)} findings" + (f" ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})"
                                                    if kinds else ""), file=sys.stderr)
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
