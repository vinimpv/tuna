"""Run ledger: one JSON line per pipeline stage, written as the stage ends.

  with metrics.Stage("train", repo="saleor", model="LiquidAI/LFM2.5-Embedding-350M", config={...}) as st:
      ...
      st.counts(questions=1350, chunks=13509)
      st.tokens(prompt=..., completion=...)
      st.outputs(path1, path2)
      st.extra(history=[...])

Each line records the stage, repo and model, start and end time (UTC, ISO
8601), wall time, status (ok, or the exception that ended it), git SHA and
dirty flag, the full config, library versions (torch, transformers,
sentence-transformers, peft when present, CUDA, GPU driver), counts, LLM
token totals, peak VRAM (PyTorch allocations in this process), throughput
(items/s and tokens/s when the counts allow) and output paths with sizes.
The ledger lives in $TUNA_METRICS (default ~/tuna-data/metrics)/ledger.jsonl,
next to env-<date>.json (GPU, driver, CPU, RAM, OS, versions), written once a
day. scripts/gpu-energy.py joins ledger lines with the 1-second GPU power log.
"""
import datetime
import functools
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

from . import common

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def metrics_dir():
    d = os.path.expanduser(os.environ.get("TUNA_METRICS") or os.path.join(common.data_root(), "metrics"))
    os.makedirs(d, exist_ok=True)
    return d


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t):
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git_info():
    def run(*a):
        r = subprocess.run(["git", "-C", PROJECT_DIR, *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


@functools.lru_cache(maxsize=1)
def nvidia_smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    name, driver, mem, power = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    return {"name": name, "driver": driver, "memory_mib": int(float(mem)), "power_limit_w": float(power)}


@functools.lru_cache(maxsize=1)
def versions():
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "sentence-transformers", "peft", "numpy", "tree-sitter", "safetensors"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    smi = nvidia_smi()
    if smi:
        out["gpu_driver"] = smi["driver"]
        out["gpu"] = smi["name"]
    return out


def environment():
    def read(path):
        try:
            return open(path, encoding="utf-8").read()
        except OSError:
            return ""
    cpu = next((l.split(":", 1)[1].strip() for l in read("/proc/cpuinfo").splitlines()
                if l.startswith("model name")), platform.processor())
    mem_kb = next((int(l.split()[1]) for l in read("/proc/meminfo").splitlines() if l.startswith("MemTotal")), None)
    osr = dict(l.split("=", 1) for l in read("/etc/os-release").splitlines() if "=" in l)
    return {"written_utc": iso(utc_now()), "gpu": nvidia_smi(),
            "cpu": {"model": cpu, "logical_cores": os.cpu_count()},
            "ram_gib": round(mem_kb / 2**20, 1) if mem_kb else None,
            "os": {"name": osr.get("PRETTY_NAME", "").strip('"') or platform.platform(), "kernel": platform.release(),
                   "machine": platform.machine()},
            "versions": versions()}


def ensure_env():
    path = os.path.join(metrics_dir(), f"env-{utc_now():%Y-%m-%d}.json")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            json.dump(environment(), fh, indent=1)
    return path


def path_bytes(path):
    if os.path.isdir(path):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    return os.path.getsize(path) if os.path.exists(path) else None


def config_of(args):
    """A command's arguments as a JSON-safe config."""
    return {k: v for k, v in vars(args).items() if k != "func" and not callable(v)}


def append(line):
    line.setdefault("machine", machine())
    for k in ("config", "outputs"):   # paths relative to $TUNA_OUT / $TUNA_DATA, so the ledger names no home directory
        if k in line:
            line[k] = portable(line[k])
    path = os.path.join(metrics_dir(), "ledger.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    return path


def _status(exc_type, exc):
    if exc_type is None:
        return "ok"
    if issubclass(exc_type, SystemExit):
        code = exc.code if isinstance(exc, SystemExit) else None
        return "ok" if code in (None, 0) else f"exit {code}"[:500]
    return f"failed: {exc_type.__name__}: {exc}"[:500]


# Machine label recorded on every ledger line and checkpoint (TUNA_MACHINE, else the hostname). The released ledger
# calls our local GPU machine "local"; timing and energy numbers in the summary use only those lines.
LOCAL_MACHINE = "local"


def machine():
    return os.environ.get("TUNA_MACHINE") or platform.node()


def is_local(line):
    """A line from the local machine: its label equals TUNA_LOCAL_MACHINE (default "local"); lines without the
    field predate it and are local."""
    local = os.environ.get("TUNA_LOCAL_MACHINE", LOCAL_MACHINE)
    return line.get("machine", local) == local


def portable(v):
    """Paths under the output and data roots written as $TUNA_OUT/... and $TUNA_DATA/..., paths inside this checkout
    relative to it, the home as ~."""
    if isinstance(v, dict):
        return {k: portable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [portable(x) for x in v]
    if isinstance(v, str):
        if v.startswith(PROJECT_DIR + os.sep):
            return os.path.relpath(v, PROJECT_DIR)
        for root, name in ((common.default_out(), "$TUNA_OUT"), (common.data_root(), "$TUNA_DATA"),
                           (os.path.expanduser("~"), "~")):
            root = root.rstrip(os.sep)
            if v == root or v.startswith(root + os.sep):
                return name + v[len(root):]
    return v


class Stage:
    def __init__(self, stage, repo=None, model=None, config=None):
        self.line = {"stage": stage, "repo": repo, "model": model, "config": dict(config or {}),
                     "counts": {}, "tokens": {}, "outputs": [], "extra": {}}

    def __enter__(self):
        self.t0, self.start = time.perf_counter(), utc_now()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        return self

    def counts(self, **kw):
        self.line["counts"].update(kw)

    def tokens(self, **kw):
        self.line["tokens"].update(kw)

    def config(self, **kw):
        self.line["config"].update(kw)

    def extra(self, **kw):
        self.line["extra"].update(kw)

    def outputs(self, *paths):
        self.line["outputs"] += [{"path": p, "bytes": path_bytes(p)} for p in paths]

    def __exit__(self, exc_type, exc, tb):
        end, wall = utc_now(), time.perf_counter() - self.t0
        torch = sys.modules.get("torch")
        peak = None
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        items = self.line["counts"].get("items")
        completion = self.line["tokens"].get("completion")
        self.line.update(
            start_utc=iso(self.start), end_utc=iso(end), wall_seconds=round(wall, 3),
            status=_status(exc_type, exc),
            git=git_info(), versions=versions(), peak_vram_gib=peak,
            throughput={k: v for k, v in {"items_per_s": round(items / wall, 3) if items and wall else None,
                                          "completion_tokens_per_s": round(completion / wall, 1)
                                          if completion and wall else None}.items() if v is not None})
        ensure_env()
        append(self.line)
        return False
