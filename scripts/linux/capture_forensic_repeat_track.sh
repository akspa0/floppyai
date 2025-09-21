#!/usr/bin/env bash
set -euo pipefail

# capture_forensic_repeat_track.sh
# Repeatedly capture the same track/side with high revolutions to study drift and media/head tolerances.
# Forensic-rich defaults: many revolutions, minimal retries (none configured), cooldowns, strong logging.
#
# Example:
#   ./capture_forensic_repeat_track.sh \
#     --track 40 --side 0 --repeats 12 --revs 16 \
#     --cooldown 5 --spinup 2 \
#     --out-dir ./captures --label repeat40 --drive 0

usage() {
  cat <<'USAGE'
Usage: capture_forensic_repeat_track.sh [options]

Required:
  --track <N>            Track number
  --side <0|1>          Side

Optional (common):
  --drive <N>           DTC drive index (default: 0)
  --dtc-path <path>     Path to dtc executable (default: dtc)
  --out-dir <dir>       Output directory (default: ./captures)
  --label <name>        Label used in filenames (default: repeat)
  --no-sudo             Do not prefix dtc with sudo (default: sudo on)
  --dry-run             Print commands only; do not execute

Optional (capture profile):
  --repeats <N>         Number of passes (default: 10)
  --revs <N>            Revolutions per capture (default: 16)
  --cooldown <sec>      Pause between passes (default: 5)
  --spinup <sec>        Spin-up delay before first capture (default: 2)
USAGE
}

# Defaults
TRACK=""; SIDE=""
DRIVE=0
DTC_BIN="/usr/bin/dtc"
OUT_DIR="./captures"
LABEL="repeat"
USE_SUDO=1
DRY_RUN=0
REPEATS=10
REVS=16
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
    --repeats) REPEATS=${2:?}; shift 2;;
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
LOG_PATH="$RUN_DIR/run.log"
SUDO_PREFIX=""; [[ $USE_SUDO -eq 1 ]] && SUDO_PREFIX="sudo " || true

pushd "$RUN_DIR" >/dev/null

{
  echo "capture_forensic_repeat_track.sh run at $(date -Iseconds)"
  echo "dtc path: $(command -v "$DTC_BIN" || echo "$DTC_BIN")"
  echo "dtc version:"; "$DTC_BIN" -V || true; echo
  echo "Track: ${TRACK}  Side: ${SIDE}  Repeats: ${REPEATS}  Revs: ${REVS}"
  echo "Cooldown: ${COOLDOWN}s  Spinup: ${SPINUP}s"
} > "$LOG_PATH"

echo "-- Spin-up ${SPINUP}s" | tee -a "$LOG_PATH"; sleep "$SPINUP"

for ((i=1; i<=REPEATS; i++)); do
  ts=$(date +"%Y%m%d_%H%M%S")
  # Run within RUN_DIR; use simple prefix 'track' so files are track%02d.%d.raw
  cmd="${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -i0 -p -s${TRACK} -e${TRACK} -g${SIDE} -r${REVS} -ftrack"
  echo "[READ ] $cmd" | tee -a "$LOG_PATH"
  if [[ $DRY_RUN -eq 0 ]]; then
    bash -lc "$cmd" | tee -a "$LOG_PATH"
  fi
  if (( i < REPEATS )); then
    echo "Cooldown ${COOLDOWN}s..." | tee -a "$LOG_PATH"
    sleep "$COOLDOWN"
  fi
done

echo "Done. Outputs in $RUN_DIR" | tee -a "$LOG_PATH"
popd >/dev/null
