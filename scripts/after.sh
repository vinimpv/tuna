#!/usr/bin/env bash
# Run COMMAND once the detached run NAME has finished with exit 0 (its done marker).
#   scripts/after.sh NAME COMMAND [ARGS...]
set -euo pipefail
NAME=${1:?usage: after.sh NAME COMMAND [ARGS...]}; shift
RUNS=${TUNA_OUT:-${TUNA_DATA:-$HOME/tuna-data}/out}/runs
until [ -f "$RUNS/$NAME.done" ]; do sleep 30; done
if ! grep -q '^exit=0 ' "$RUNS/$NAME.done"; then
  echo "after.sh: $NAME ended with $(cat "$RUNS/$NAME.done"); not starting: $*" >&2
  exit 1
fi
exec "$@"
