#!/usr/bin/env bash
# Run a long chain detached from the calling shell (survives the shell exiting).
#
#   scripts/detach.sh NAME COMMAND [ARGS...]
#
# Output goes to $TUNA_OUT/runs/NAME.log; when the command ends, $TUNA_OUT/runs/NAME.done holds its exit
# code and end time. A stale .done marker from an earlier run of NAME is removed first. Prints the PID.
set -euo pipefail
NAME=${1:?usage: detach.sh NAME COMMAND [ARGS...]}; shift
cd "$(dirname "$0")/.."
RUNS=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}/runs
mkdir -p "$RUNS"
rm -f "$RUNS/$NAME.done"
setsid nohup bash -c '
  "$@"
  rc=$?
  echo "exit=$rc end=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "'"$RUNS/$NAME.done"'"
' detach-"$NAME" "$@" > "$RUNS/$NAME.log" 2>&1 < /dev/null &
echo "$!"
