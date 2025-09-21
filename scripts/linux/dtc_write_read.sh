#!/usr/bin/env bash
set -euo pipefail

# dtc_write_read.sh
# Write a .raw to a track/side and read back N revolutions using KryoFlux DTC on Linux.
# Intended to run on the Linux host connected to the KryoFlux board.
# Requires sudo for DTC operations on most systems.
#
# Example:
#   ./dtc_write_read.sh \
#     --write /path/to/generated.raw --track 80 --side 0 --revs 3 \
#     --out-dir ./captures --drive 0
#
# Output:
#   - Capture saved under --out-dir with a timestamped filename
#   - A .log file saved alongside containing dtc version and the exact commands run

usage() {
  cat <<'USAGE'
Usage: dtc_write_read.sh [options]

Required:
  --write <file>         Input .raw to write to disk
  --track <N>            Track number (e.g., 80)
  --side <0|1>           Side (0 or 1)

Optional:
  --revs <N>             Revolutions to read back (default: 3)
  --drive <N>            Drive index for dtc -d (default: 0)
  --dtc-path <path>      Path to dtc executable (default: dtc in PATH)
  --out-dir <dir>        Output directory for captured .raw and logs (default: ./captures)
  --label <name>         Optional label in output filename (default: none)
  --no-sudo              Do not prefix dtc commands with sudo (default: sudo on)
  --dry-run              Print commands without executing
  -h|--help              Show this help

Notes:
- This script assumes:
    * Write:   dtc -d <drive> -i 21 -f <input.raw> -t <track> -s <side> write
    * Read:    dtc -d <drive> -i 0  -t <track> -s <side> -r <revs> -f <output.raw> read
  Adjust flags if your dtc version differs.
- Use sacrificial media; allow cooldowns; prefer outer tracks first.
USAGE
}

WRITE_INPUT=""
TRACK=""
SIDE=""
REVS=3
DRIVE=0
DTC_BIN="/usr/bin/dtc"
OUT_DIR="./captures"
LABEL=""
USE_SUDO=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --write) WRITE_INPUT=${2:?}; shift 2;;
    --track) TRACK=${2:?}; shift 2;;
    --side)  SIDE=${2:?}; shift 2;;
    --revs)  REVS=${2:?}; shift 2;;
    --drive) DRIVE=${2:?}; shift 2;;
    --dtc-path) DTC_BIN=${2:?}; shift 2;;
    --out-dir) OUT_DIR=${2:?}; shift 2;;
    --label) LABEL=${2:?}; shift 2;;
    --no-sudo) USE_SUDO=0; shift 1;;
    --dry-run) DRY_RUN=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done

# Validate inputs
if [[ -z "$WRITE_INPUT" || -z "$TRACK" || -z "$SIDE" ]]; then
  echo "Error: --write, --track, and --side are required" >&2
  usage
  exit 2
fi
if [[ ! -f "$WRITE_INPUT" ]]; then
  echo "Error: input file not found: $WRITE_INPUT" >&2
  exit 2
fi

# Resolve dtc
if ! command -v "$DTC_BIN" >/dev/null 2>&1; then
  if [[ -x "$DTC_BIN" ]]; then
    : # OK explicit path
  else
    echo "Error: dtc not found: $DTC_BIN" >&2
    exit 2
  fi
fi

mkdir -p "$OUT_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
BASE_NAME="capture_t${TRACK}_s${SIDE}_r${REVS}_${TS}"
if [[ -n "$LABEL" ]]; then
  SAFE_LABEL=$(echo "$LABEL" | tr -cd '[:alnum:]_.-')
  BASE_NAME="${SAFE_LABEL}_${BASE_NAME}"
fi
CAPTURE_PATH="$OUT_DIR/${BASE_NAME}.raw"
LOG_PATH="$OUT_DIR/${BASE_NAME}.log"

SUDO_PREFIX=""
if [[ $USE_SUDO -eq 1 ]]; then
  SUDO_PREFIX="sudo "
fi

{
  echo "dtc_write_read.sh run at $(date -Iseconds)"
  echo "dtc path: $(command -v "$DTC_BIN" || echo "$DTC_BIN")"
  echo "dtc version:"
  $DTC_BIN -V || true
  echo
} >"$LOG_PATH"

WRITE_DIR=$(dirname "${WRITE_INPUT}")
WRITE_BASE=$(basename "${WRITE_INPUT}")
pushd "$WRITE_DIR" >/dev/null
WRITE_CMD=(bash -lc "${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -wi4 -f${WRITE_BASE} -s${TRACK} -e${TRACK} -g${SIDE} -w")

# For read, run within OUT_DIR and use a prefix so DTC creates BASE_NAME%02d.%d.raw
READ_CMD=(bash -lc "${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -i0 -s${TRACK} -e${TRACK} -g${SIDE} -r${REVS} -f${BASE_NAME}")

echo "[WRITE] ${WRITE_CMD[*]}"
echo "[READ ] ${READ_CMD[*]}"
{
  echo "[WRITE] ${WRITE_CMD[*]}"
  echo "[READ ] ${READ_CMD[*]}"
} >>"$LOG_PATH"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry-run: not executing dtc commands. Log at $LOG_PATH"
  exit 0
fi

# Execute
# shellcheck disable=SC2068
eval ${WRITE_CMD[@]} | tee -a "$LOG_PATH"
popd >/dev/null
pushd "$OUT_DIR" >/dev/null
# shellcheck disable=SC2068
eval ${READ_CMD[@]} | tee -a "$LOG_PATH"
popd >/dev/null

echo "Capture saved to: $OUT_DIR/${BASE_NAME}%02d.%d.raw"
echo "Log saved to:     $LOG_PATH"
