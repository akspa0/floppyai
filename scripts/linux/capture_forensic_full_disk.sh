#!/usr/bin/env bash
set -euo pipefail

# capture_forensic_full_disk.sh
# Forensic-rich full-disk STREAM capture using KryoFlux DTC on Linux.
# - Captures high revolutions per track to maximize statistical fidelity
# - Minimal retries (none configured) to avoid repeated seeks on fragile media
# - Strong logging of exact commands run
# - Cooldowns and optional spin-up per side to reduce thermal/drive stress
#
# Example:
#   ./capture_forensic_full_disk.sh \
#     --profile 35HD --sides both --revs 3 \
#     --start-track 0 --end-track 79 --step 1 \
#     --cooldown 3 --spinup 2 \
#     --out-dir ./captures --label win95_set --drive 0
#
# Output:
#   <out-dir>/<label_or_full-disk>_<timestamp>/
#     - *.raw STREAM files (e.g., label_t00_s0_r3_YYYYmmdd_HHMMSS.raw)
#     - run.log with DTC version and command audit

usage() {
  cat <<'USAGE'
Usage: capture_forensic_full_disk.sh [options]

Optional (common):
  --drive <N>            DTC drive index (default: 0)
  --dtc-path <path>      Path to dtc executable (default: dtc)
  --out-dir <dir>        Output directory (default: ./captures)
  --label <name>         Label used in filenames (default: full-disk)
  --no-sudo              Do not prefix dtc with sudo (default: sudo on)
  --dry-run              Print commands only; do not execute
  --rich                 Enable extra dtc flags (-t1 -k1 and, unless disabled, -m2 -l63 -p)
  --no-p                 When rich, do not include -p
  --no-ml                When rich, do not include -m2 -l63

Optional (capture profile):
  --profile <name>       One of: 35HD, 35DD, 525HD, 525DD (sets default track range)
  --sides <both|0|1>     Which sides to capture (default: both)
  --start-track <N>      Starting track (default from profile or 0)
  --end-track <N>        Ending track inclusive (default from profile or 79/39)
  --step <N>             Track step (default: 1)
  --revs <N>             Revolutions per track (default: 3)
  --cooldown <sec>       Pause between tracks (default: 2)
  --spinup <sec>         Spin-up delay before each side (default: 2)

Notes:
- Captures STREAM files using -i0. No decoding is performed.
- Retries are intentionally not configured; rely on revolutions (-r N).
- Use sacrificial media for experiments.
USAGE
}

# Defaults
DRIVE=0
DTC_BIN="/usr/bin/dtc"
OUT_DIR="/srv/kryoflux/captures"
LABEL="full-disk"
USE_SUDO=1
DRY_RUN=0
PROFILE=""
SIDES="both"
START_TRACK=""
END_TRACK=""
STEP=1
REVS=3
COOLDOWN=2
SPINUP=2
RICH=0
NO_P=0
NO_ML=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --drive) DRIVE=${2:?}; shift 2;;
    --dtc-path) DTC_BIN=${2:?}; shift 2;;
    --out-dir) OUT_DIR=${2:?}; shift 2;;
    --label) LABEL=${2:?}; shift 2;;
    --no-sudo) USE_SUDO=0; shift 1;;
    --dry-run) DRY_RUN=1; shift 1;;
    --profile) PROFILE=${2:?}; shift 2;;
    --sides) SIDES=${2:?}; shift 2;;
    --start-track) START_TRACK=${2:?}; shift 2;;
    --end-track) END_TRACK=${2:?}; shift 2;;
    --step) STEP=${2:?}; shift 2;;
    --revs) REVS=${2:?}; shift 2;;
    --cooldown) COOLDOWN=${2:?}; shift 2;;
    --spinup) SPINUP=${2:?}; shift 2;;
    --rich) RICH=1; shift 1;;
    --no-p) NO_P=1; shift 1;;
    --no-ml) NO_ML=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done

# Apply profile defaults for track ranges if not provided
if [[ -n "$PROFILE" ]]; then
  case "${PROFILE^^}" in
    35HD|35DD|525HD)
      : # 0..79
      : ${START_TRACK:="0"}
      : ${END_TRACK:="79"}
      ;;
    525DD)
      : ${START_TRACK:="0"}
      : ${END_TRACK:="39"}
      ;;
    *) echo "Warning: unknown profile '$PROFILE', using generic defaults";;
  esac
fi
: ${START_TRACK:="0"}
: ${END_TRACK:="79"}

# Validate
if ! [[ "$SIDES" == "both" || "$SIDES" == "0" || "$SIDES" == "1" ]]; then
  echo "Error: --sides must be one of both|0|1" >&2; exit 2
fi
if ! command -v "$DTC_BIN" >/dev/null 2>&1 && [[ ! -x "$DTC_BIN" ]]; then
  echo "Error: dtc not found: $DTC_BIN" >&2; exit 2
fi

TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$OUT_DIR/${LABEL}_${TS}"
mkdir -p "$RUN_DIR"

SUDO_PREFIX=""; [[ $USE_SUDO -eq 1 ]] && SUDO_PREFIX="sudo " || true

pushd "$RUN_DIR" >/dev/null
LOG_FILE="run.log"
umask 0002

{
  echo "capture_forensic_full_disk.sh run at $(date -Iseconds)"
  echo "dtc path: $(command -v "$DTC_BIN" || echo "$DTC_BIN")"
  echo "dtc header:"
  "$DTC_BIN" 2>&1 | head -n 4 || true
  echo
  echo "Profile: ${PROFILE}  Sides: ${SIDES}  Tracks: ${START_TRACK}..${END_TRACK} step ${STEP}  Revs: ${REVS}"
  echo "Cooldown: ${COOLDOWN}s  Spinup: ${SPINUP}s"
} > "$LOG_FILE"

build_read_cmd() {
  local start=$1 end=$2 side=$3
  local cmd="${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -i0 -s${start} -e${end} -g${side} -r${REVS} -ftrack"
  if (( RICH == 1 )); then
    # Add extras conservatively
    if (( NO_ML == 0 )); then cmd+=" -m2 -l63"; fi
    cmd+=" -t1 -k1"
    if (( NO_P == 0 )); then cmd+=" -p"; fi
  fi
  echo "$cmd"
}

run_read() {
  local track=$1 side=$2
  # Use builder; files are track%02d.%d.raw under current dir
  local cmd=$(build_read_cmd "$track" "$track" "$side")
  echo "[READ ] $cmd"
  echo "[READ ] $cmd" >> "$LOG_FILE"
  if [[ $DRY_RUN -eq 0 ]]; then
    eval "$cmd" | tee -a "$LOG_FILE"
  fi
}

capture_side() {
  local side=$1
  echo "-- Side ${side} spin-up: ${SPINUP}s" | tee -a "$LOG_FILE"
  sleep "$SPINUP"
  if (( STEP == 1 )); then
    # Single dtc invocation across the whole track range for this side with fallbacks
    local cmd1=$(build_read_cmd "$START_TRACK" "$END_TRACK" "$side")
    echo "[READ ] $cmd1" | tee -a "$LOG_FILE"
    if [[ $DRY_RUN -eq 0 ]]; then
      eval "$cmd1" | tee -a "$LOG_FILE"
    fi
    shopt -s nullglob; files=(track*.raw)
    if (( ${#files[@]} == 0 )); then
      echo "WARN: No track*.raw found after first attempt; retrying with -ftrack.raw" | tee -a "$LOG_FILE"
      # Modify -f argument from -ftrack to -ftrack.raw
      local cmd2=$(build_read_cmd "$START_TRACK" "$END_TRACK" "$side")
      cmd2=${cmd2/-ftrack/-ftrack.raw}
      echo "[READ ] $cmd2" | tee -a "$LOG_FILE"
      if [[ $DRY_RUN -eq 0 ]]; then
        eval "$cmd2" | tee -a "$LOG_FILE"
      fi
      files=(track*.raw)
    fi
    if (( ${#files[@]} == 0 )); then
      echo "WARN: Still no files; retrying with per-track loop" | tee -a "$LOG_FILE"
      local t=$START_TRACK
      while (( t <= END_TRACK )); do
        run_read "$t" "$side"
        t=$(( t + 1 ))
      done
      files=(track*.raw)
    fi
    if (( ${#files[@]} == 0 )); then
      echo "ERROR: No track*.raw files were written. Listing directory and environment:" | tee -a "$LOG_FILE"
      pwd | tee -a "$LOG_FILE"
      id -u | tee -a "$LOG_FILE"
      ls -la | tee -a "$LOG_FILE"
      echo "Hints: (1) Power-cycle/reset KryoFlux and drive. (2) Rerun with --dry-run to inspect commands. (3) Verify permissions in $PWD." | tee -a "$LOG_FILE"
    fi
  else
    # Fallback: per-track loop when a custom STEP is requested
    local t=$START_TRACK
    while (( t <= END_TRACK )); do
      run_read "$t" "$side"
      echo "Cooldown ${COOLDOWN}s..." | tee -a "$LOG_FILE"
      sleep "$COOLDOWN"
      t=$(( t + STEP ))
    done
  fi
}

# Execute
if [[ "$SIDES" == "both" || "$SIDES" == "0" ]]; then
  capture_side 0
fi
if [[ "$SIDES" == "both" || "$SIDES" == "1" ]]; then
  capture_side 1
fi

echo "Done. Outputs in $RUN_DIR" | tee -a "$LOG_FILE"
popd >/dev/null
