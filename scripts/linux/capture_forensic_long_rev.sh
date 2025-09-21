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
  --out-dir <dir>       Output directory (default: ./captures)
  --label <name>        Label used in filenames (default: longrev)
  --no-sudo             Do not prefix dtc with sudo (default: sudo on)
  --dry-run             Print commands only; do not execute

Optional (capture profile):
  --revs <N>            Revolutions per capture (default: 3)
  --cooldown <sec>      Pause after capture (default: 5)
  --spinup <sec>        Spin-up delay before capture (default: 2)
USAGE
}

# Defaults
TRACK=""; SIDE=""
DRIVE=0
DTC_BIN="/usr/bin/dtc"
OUT_DIR="./captures"
LABEL="longrev"
USE_SUDO=1
DRY_RUN=0
REVS=3
COOLDOWN=5
SPINUP=2

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

{
  echo "capture_forensic_long_rev.sh run at $(date -Iseconds)"
  echo "dtc path: $(command -v "$DTC_BIN" || echo "$DTC_BIN")"
  echo "dtc version:"; "$DTC_BIN" -V || true; echo
  echo "Track: ${TRACK}  Side: ${SIDE}  Revs: ${REVS}"
  echo "Cooldown: ${COOLDOWN}s  Spinup: ${SPINUP}s"
} > "$LOG_FILE"

echo "-- Spin-up ${SPINUP}s" | tee -a "$LOG_FILE"; sleep "$SPINUP"

# Capture
TS2=$(date +"%Y%m%d_%H%M%S")
# Use a standard prefix so dtc writes track%02d.%d.raw under RUN_DIR
PREFIX="$RUN_DIR/track"
CMD="${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -i0 -p -s${TRACK} -e${TRACK} -g${SIDE} -r${REVS} -ftrack"
echo "[READ ] $CMD" | tee -a "$LOG_FILE"
if [[ $DRY_RUN -eq 0 ]]; then
  eval "$CMD" | tee -a "$LOG_FILE"
fi

echo "Cooldown ${COOLDOWN}s..." | tee -a "$LOG_FILE"; sleep "$COOLDOWN"

echo "Done. Outputs in $RUN_DIR" | tee -a "$LOG_FILE"
popd >/dev/null
