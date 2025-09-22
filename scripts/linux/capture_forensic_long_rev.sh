#!/usr/bin/env bash
set -euo pipefail

# capture_forensic_long_rev.sh
# Capture a single track/side with a very high number of revolutions to maximize statistical fidelity.
# Forensic-rich defaults and strong logging.
#
# Example:
#   ./capture_forensic_long_rev.sh \
#     --track 0 --side 0 --revs 48 --cooldown 5 --spinup 2 \
#     --out-dir ./captures --label longrev0 --drive 0

usage() {
  cat <<'USAGE'
Usage: capture_forensic_long_rev.sh [options]

Required:
  --track <N>            Track number
  --side <0|1>          Side

Optional (common):
  --drive <N>           DTC drive index (default: 0)
  --dtc-path <path>     Path to dtc executable (default: dtc)
  --out-dir <dir>       Output directory (default: output_captures under repo)
  --label <name>        Label used in filenames (default: longrev)
  --no-sudo             Do not prefix dtc with sudo (default: sudo on)
  --dry-run             Print commands only; do not execute
  --rich                Enable extra dtc flags (-t1 -k1 and, unless disabled, -m2 -l63 -p)
  --no-p                When rich, do not include -p
  --no-ml               When rich, do not include -m2 -l63
  --sanity              Run stream_sanity.py over track*.raw at the end

Optional (capture profile):
  --revs <N>            Revolutions per capture (default: 3)
  --cooldown <sec>      Pause after capture (default: 5)
  --spinup <sec>        Spin-up delay before capture (default: 2)
USAGE
}

# Resolve script directory for repo-relative defaults
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
TRACK=""; SIDE=""
DRIVE=0
DTC_BIN="/usr/bin/dtc"
OUT_DIR="$SCRIPT_DIR/../../output_captures"
LABEL="longrev"
USE_SUDO=1
DRY_RUN=0
REVS=3
COOLDOWN=5
SPINUP=2
RICH=0
NO_P=0
NO_ML=0
SANITY=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) TRACK=${2:?}; shift 2;;
    --side)  SIDE=${2:?}; shift 2;;
    --drive) DRIVE=${2:?}; shift 2;;
    --dtc-path) DTC_BIN=${2:?}; shift 2;;
    --out-dir) OUT_DIR=${2:?}; shift 2;;
    --label) LABEL=${2:?}; shift 2;;
    --no-sudo) USE_SUDO=0; shift 1;;
    --dry-run) DRY_RUN=1; shift 1;;
    --rich) RICH=1; shift 1;;
    --no-p) NO_P=1; shift 1;;
    --no-ml) NO_ML=1; shift 1;;
    --sanity) SANITY=1; shift 1;;
    --revs) REVS=${2:?}; shift 2;;
    --cooldown) COOLDOWN=${2:?}; shift 2;;
    --spinup) SPINUP=${2:?}; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done

# Validate
if [[ -z "$TRACK" || -z "$SIDE" ]]; then
  echo "Error: --track and --side are required" >&2; usage; exit 2
fi
if ! command -v "$DTC_BIN" >/dev/null 2>&1 && [[ ! -x "$DTC_BIN" ]]; then
  echo "Error: dtc not found: $DTC_BIN" >&2; exit 2
fi

TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUT_DIR/${LABEL}_t$(printf '%02d' "$TRACK")_s${SIDE}_${TS}"
mkdir -p "$RUN_DIR"
SUDO_PREFIX=""; [[ $USE_SUDO -eq 1 ]] && SUDO_PREFIX="sudo " || true

pushd "$RUN_DIR" >/dev/null
LOG_FILE="run.log"
umask 0002

{
  echo "capture_forensic_long_rev.sh run at $(date -Iseconds)"
  echo "dtc path: $(command -v "$DTC_BIN" || echo "$DTC_BIN")"
  echo "dtc header:"; "$DTC_BIN" 2>&1 | head -n 4; echo
  echo "Track: ${TRACK}  Side: ${SIDE}  Revs: ${REVS}"
  echo "Cooldown: ${COOLDOWN}s  Spinup: ${SPINUP}s"
} > "$LOG_FILE"

echo "-- Spin-up ${SPINUP}s" | tee -a "$LOG_FILE"; sleep "$SPINUP"

build_read_cmd() {
  local start=$1 end=$2 side=$3
  local cmd="${SUDO_PREFIX}${DTC_BIN} -ftrack -i0 -d${DRIVE} -s${start} -e${end} -g${side} -r${REVS}"
  if (( RICH == 1 )); then
    if (( NO_ML == 0 )); then cmd+=" -m2 -l63"; fi
    cmd+=" -t1 -k1"
    if (( NO_P == 0 )); then cmd+=" -p"; fi
  fi
  echo "$cmd"
}

# Capture
TS2=$(date +"%Y%m%d_%H%M%S")
CMD=$(build_read_cmd "$TRACK" "$TRACK" "$SIDE")
echo "[READ ] $CMD" | tee -a "$LOG_FILE"
if [[ $DRY_RUN -eq 0 ]]; then
  eval "$CMD" | tee -a "$LOG_FILE"
fi

echo "Cooldown ${COOLDOWN}s..." | tee -a "$LOG_FILE"; sleep "$COOLDOWN"

shopt -s nullglob; files=(track*.raw)
if (( ${#files[@]} == 0 )); then
  echo "ERROR: No track*.raw files were written. Listing directory:" | tee -a "$LOG_FILE"
  pwd | tee -a "$LOG_FILE"
  ls -la | tee -a "$LOG_FILE"
  echo "Hints: reset KryoFlux/drive; try --rich/--no-p/--no-ml toggles; verify permissions." | tee -a "$LOG_FILE"
fi

echo "Done. Outputs in $RUN_DIR" | tee -a "$LOG_FILE"

# Optional: run sanity checker
if (( SANITY == 1 )); then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  TOOL="$SCRIPT_DIR/../../tools/stream_sanity.py"
  if [[ -f "$TOOL" ]]; then
    echo "Running stream sanity on track*.raw" | tee -a "$LOG_FILE"
    python3 "$TOOL" --glob 'track*.raw' | tee -a "$LOG_FILE" || true
  else
    echo "WARN: sanity tool not found at $TOOL" | tee -a "$LOG_FILE"
  fi
fi
popd >/dev/null
