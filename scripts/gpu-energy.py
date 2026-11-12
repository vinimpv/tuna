"""GPU energy and power per run-ledger stage, from the 1-second GPU log.

  .venv/bin/python scripts/gpu-energy.py [--stage train] [--repo saleor] [--json OUT.json] [--utc-offset-hours H]

Inputs, both under $TUNA_METRICS (default $TUNA_DATA/metrics). gpu-1s.csv is not in this release: without
it every energy field is None.

  ledger.jsonl  one line per stage with start_utc / end_utc (tuna.metrics)
  gpu-1s.csv    no header; one sample a second from the GPU host:
                timestamp (the logging host's local time, "YYYY/MM/DD HH:MM:SS.mmm"; its offset from
                UTC is --utc-offset-hours, default 0),
                GPU utilization %, memory utilization %, power W, memory used MiB,
                SM clock MHz, temperature C

For each ledger line, the samples inside [start, end] give: energy (Wh, the
trapezoid over sample times), mean and peak power (W), mean GPU utilization,
peak memory, and how much of the stage the samples cover (a stage outside the
logged span, or in a gap, is reported with low coverage instead of a made-up
number). The GPU is shared by whatever ran in that window, so this is the
GPU's energy during the stage, not energy attributable to it alone.
"""
import argparse
import bisect
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tuna import metrics  # noqa: E402

def load_samples(path, utc_offset_hours=0.0):
    """The log's samples as (UTC seconds, util, mem util, W, MiB, MHz, C); [] when the log is missing."""
    offset = datetime.timedelta(hours=utc_offset_hours)
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                local = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
                t = (local - offset).replace(tzinfo=datetime.timezone.utc).timestamp()
                rows.append((t, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                             float(parts[5]), float(parts[6])))
            except ValueError:
                continue
    rows.sort()
    return rows


def parse_iso(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def window(samples, times, t0, t1):
    i, j = bisect.bisect_left(times, t0), bisect.bisect_right(times, t1)
    part = samples[i:j]
    if len(part) < 2:
        return {"samples": len(part), "coverage": 0.0}
    wh = sum((b[0] - a[0]) * (a[3] + b[3]) / 2 for a, b in zip(part, part[1:]) if b[0] - a[0] <= 5) / 3600
    covered = sum(min(b[0] - a[0], 5) for a, b in zip(part, part[1:]) if b[0] - a[0] <= 5)
    power = [s[3] for s in part]
    return {"samples": len(part), "coverage": round(covered / max(t1 - t0, 1e-9), 3), "energy_wh": round(wh, 3),
            "mean_power_w": round(sum(power) / len(power), 1), "peak_power_w": round(max(power), 1),
            "mean_gpu_util": round(sum(s[1] for s in part) / len(part), 1),
            "peak_memory_mib": max(s[4] for s in part)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage")
    ap.add_argument("--repo")
    ap.add_argument("--json", help="write the joined rows here")
    ap.add_argument("--utc-offset-hours", type=float, default=0.0, help="the GPU log's local time minus UTC, in hours")
    args = ap.parse_args()
    d = metrics.metrics_dir()
    samples = load_samples(os.path.join(d, "gpu-1s.csv"), args.utc_offset_hours)
    times = [s[0] for s in samples]
    rows = []
    for line in open(os.path.join(d, "ledger.jsonl")):
        led = json.loads(line)
        if not metrics.is_local(led):  # lines from rented machines have no local GPU power log
            continue
        if (args.stage and led["stage"] != args.stage) or (args.repo and led.get("repo") != args.repo):
            continue
        if not led.get("start_utc") or not led.get("end_utc"):
            continue
        w = window(samples, times, parse_iso(led["start_utc"]), parse_iso(led["end_utc"]))
        rows.append({"stage": led["stage"], "repo": led.get("repo"), "model": led.get("model"),
                     "start_utc": led["start_utc"], "end_utc": led["end_utc"],
                     "wall_seconds": led.get("wall_seconds"), **w})
    span = (metrics.iso(datetime.datetime.fromtimestamp(times[0], datetime.timezone.utc)),
            metrics.iso(datetime.datetime.fromtimestamp(times[-1], datetime.timezone.utc))) if times else None
    print(f"GPU log: {len(samples)} samples, {span[0]} .. {span[1]} UTC" if span else "GPU log: empty")
    print(f"{'stage':12s} {'repo':8s} {'model':34s} {'start (UTC)':24s} {'wall s':>8s} {'cov':>5s} {'Wh':>8s} "
          f"{'mean W':>7s} {'peak W':>7s} {'util%':>6s}")
    for r in rows:
        print(f"{r['stage']:12s} {str(r['repo']):8s} {str(r['model'])[:34]:34s} {r['start_utc']:24s} "
              f"{r['wall_seconds'] or 0:8.1f} {r['coverage']:5.2f} {r.get('energy_wh', float('nan')):8.2f} "
              f"{r.get('mean_power_w', float('nan')):7.1f} {r.get('peak_power_w', float('nan')):7.1f} "
              f"{r.get('mean_gpu_util', float('nan')):6.1f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"gpu_log_span_utc": span, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
