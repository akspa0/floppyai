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
# Resolve script directory for repo-relative default output
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../../output_captures"
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
  echo "dtc header:"
  "$DTC_BIN" 2>&1 | head -n 4 || true
  echo
} >"$LOG_PATH"

WRITE_DIR=$(dirname "${WRITE_INPUT}")
WRITE_BASE=$(basename "${WRITE_INPUT}")
pushd "$WRITE_DIR" >/dev/null

# Decide write mode for image type 4: prefix (preferred) or per-file fallback
WRITE_PREFIX=""
PERFILE_BASE=""

# Detect naming from the provided file name
if [[ "$WRITE_BASE" =~ ^track([0-9]{2})\.([01])\.raw$ ]]; then
  WRITE_PREFIX="track"
  EXPECTED=$(printf "track%02d.%d.raw" "$TRACK" "$SIDE")
elif [[ "$WRITE_BASE" =~ ^([0-9]{2})\.([01])\.raw$ ]]; then
  WRITE_PREFIX="./"
  EXPECTED=$(printf "%02d.%d.raw" "$TRACK" "$SIDE")
else
  # Unknown naming; use per-file base (strip .raw)
  PERFILE_BASE="${WRITE_BASE%.raw}"
  EXPECTED="${PERFILE_BASE}.raw"
fi

# Validate expected file exists for the requested track/side
if [[ -n "$WRITE_PREFIX" ]]; then
  if [[ ! -f "$EXPECTED" ]]; then
    echo "[ERROR] Expected file not found for track=$TRACK side=$SIDE: $EXPECTED" | tee -a "$LOG_PATH"
    echo "[PWD] $(pwd)" | tee -a "$LOG_PATH"
    ls -l | tee -a "$LOG_PATH"
    popd >/dev/null
    exit 2
  fi
else
  if [[ ! -f "$EXPECTED" ]]; then
    echo "[WARN] Per-file mode: input file not found as '$EXPECTED' in current dir; trying original name '$WRITE_BASE'"
    if [[ -f "$WRITE_BASE" ]]; then
      PERFILE_BASE="${WRITE_BASE%.raw}"
    else
      echo "[ERROR] Neither '$EXPECTED' nor '$WRITE_BASE' exists in $(pwd)." | tee -a "$LOG_PATH"
      ls -l | tee -a "$LOG_PATH"
      popd >/dev/null
      exit 2
    fi
  fi
fi

# Build exact write/read command strings for logging
if [[ -n "$WRITE_PREFIX" ]]; then
  WRITE_STR="${SUDO_PREFIX}${DTC_BIN} -f${WRITE_PREFIX} -i4 -d${DRIVE} -s${TRACK} -e${TRACK} -g${SIDE} -w"
else
  WRITE_STR="${SUDO_PREFIX}${DTC_BIN} -f${PERFILE_BASE} -i4 -d${DRIVE} -s${TRACK} -e${TRACK} -g${SIDE} -w"
fi
READ_STR="${SUDO_PREFIX}${DTC_BIN} -f${BASE_NAME} -i0 -d${DRIVE} -s${TRACK} -e${TRACK} -g${SIDE} -r${REVS}"

echo "[WRITE] $WRITE_STR"
echo "[READ ] $READ_STR"
{
  echo "[WRITE] $WRITE_STR"
  echo "[READ ] $READ_STR"
} >>"$LOG_PATH"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry-run: not executing dtc commands. Log at $LOG_PATH"
  popd >/dev/null
  exit 0
fi

# Execute write with fallbacks. Try up to 4 base variants; stop on first success.
attempt_write() {
  local base="$1"
  local cmd="${SUDO_PREFIX}${DTC_BIN} -f${base} -i0 -d${DRIVE} -s${TRACK} -e${TRACK} -g${SIDE} -w"
  echo "[WRITE] $cmd" | tee -a "$LOG_PATH"
  if [[ $USE_SUDO -eq 1 ]]; then
    sudo "$DTC_BIN" -f"$base" -i0 -d"$DRIVE" -s"$TRACK" -e"$TRACK" -g"$SIDE" -w | tee -a "$LOG_PATH"
  else
    "$DTC_BIN" -f"$base" -i0 -d"$DRIVE" -s"$TRACK" -e"$TRACK" -g"$SIDE" -w | tee -a "$LOG_PATH"
  fi
  return $?
}

ABS_DIR="$(pwd)"
TRIED_Bases=()
RC=1
if [[ -n "$WRITE_PREFIX" ]]; then
  # 1) detected prefix (track or ./)
  TRIED_Bases+=("$WRITE_PREFIX")
  attempt_write "$WRITE_PREFIX"; RC=$?
  if [[ $RC -ne 0 ]]; then
    # 2) suffix with .raw
    if [[ "$WRITE_PREFIX" == "track" ]]; then
      TRIED_Bases+=("track.raw")
      attempt_write "track.raw"; RC=$?
    else
      TRIED_Bases+=("./.raw")
      attempt_write "./.raw"; RC=$?
    fi
  fi
  if [[ $RC -ne 0 ]]; then
    # 3) absolute path prefix
    if [[ "$WRITE_PREFIX" == "track" ]]; then
      TRIED_Bases+=("$ABS_DIR/track")
      attempt_write "$ABS_DIR/track"; RC=$?
    else
      TRIED_Bases+=("$ABS_DIR/")
      attempt_write "$ABS_DIR/"; RC=$?
    fi
  fi
  if [[ $RC -ne 0 ]]; then
    # 4) absolute path with .raw suffix
    if [[ "$WRITE_PREFIX" == "track" ]]; then
      TRIED_Bases+=("$ABS_DIR/track.raw")
      attempt_write "$ABS_DIR/track.raw"; RC=$?
    else
      TRIED_Bases+=("$ABS_DIR/.raw")
      attempt_write "$ABS_DIR/.raw"; RC=$?
    fi
  fi
else
  # Per-file base fallback
  TRIED_Bases+=("$PERFILE_BASE")
  attempt_write "$PERFILE_BASE"; RC=$?
fi

if [[ $RC -ne 0 ]]; then
  echo "[ERROR] DTC write failed for all base variants: ${TRIED_Bases[*]}" | tee -a "$LOG_PATH"
  echo "[PWD] $(pwd)" | tee -a "$LOG_PATH"
  ls -l | tee -a "$LOG_PATH"
  popd >/dev/null
  exit $RC
fi
popd >/dev/null

# Execute read into OUT_DIR with prefix BASE_NAME%02d.%d.raw
pushd "$OUT_DIR" >/dev/null
if [[ $USE_SUDO -eq 1 ]]; then
  sudo "$DTC_BIN" -f"$BASE_NAME" -i0 -d"$DRIVE" -s"$TRACK" -e"$TRACK" -g"$SIDE" -r"$REVS" | tee -a "$LOG_PATH"
else
  "$DTC_BIN" -f"$BASE_NAME" -i0 -d"$DRIVE" -s"$TRACK" -e"$TRACK" -g"$SIDE" -r"$REVS" | tee -a "$LOG_PATH"
fi
popd >/dev/null

echo "Capture saved to: $OUT_DIR/${BASE_NAME}%02d.%d.raw"
echo "Log saved to:     $LOG_PATH"
