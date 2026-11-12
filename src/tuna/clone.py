"""Shallow-clone a repo at the exact commit pinned in its config.

Fetches only that one commit (depth 1) into the config's clone path, then
checks it out detached. An existing clone is left alone if it already sits at
the pin; anything else is an error rather than a silent re-checkout.
"""
import os
import subprocess
import sys

from . import common


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def head_of(path):
    r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main(args):
    cfg = common.load_config(args.repo)
    path, commit = cfg["path"], cfg["commit"]
    if "url" not in cfg:
        sys.exit(f"{cfg['_file']}: missing 'url'")
    if os.path.isdir(os.path.join(path, ".git")):
        head = head_of(path)
        if head != commit:
            sys.exit(f"{path} exists at {head}, not the pinned {commit}; move it away first")
        common.check_commit(cfg)
        print(f"[clone] {cfg['name']}: already at {commit} in {path}", flush=True)
        return
    if os.path.exists(path) and os.listdir(path):
        sys.exit(f"{path} exists and is not a git clone")
    os.makedirs(path, exist_ok=True)
    git("init", "--quiet", path)
    git("remote", "add", "origin", cfg["url"], cwd=path)
    git("-c", "protocol.version=2", "fetch", "--quiet", "--depth", "1", "--no-tags", "origin", commit, cwd=path)
    git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=path)
    common.check_commit(cfg)
    print(f"[clone] {cfg['name']}: {cfg['url']} @ {commit} ({cfg.get('tag', 'no tag')}) -> {path}", flush=True)


def add_parser(sub):
    p = sub.add_parser("clone", help="shallow clone at the pinned commit")
    p.set_defaults(func=main)
    return p
