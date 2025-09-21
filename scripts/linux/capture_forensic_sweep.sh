#!/usr/bin/env bash
set -euo pipefail

# capture_forensic_sweep.sh
# Perform repeated sweeps across a track range with high revolutions to study drift over time.
# Forensic-rich defaults: many revolutions, minimal retries (none configured), cooldowns, strong logging.
#
# Example:
#   ./capture_forensic_sweep.sh \
#     --profile 35HD --sides both \
#     --start-track 0 --end-track 79 --step 1 \
#     --sweeps 3 --revs 16 --cooldown 3 --spinup 2 \
#     --out-dir ./captures --label sweepA --drive 0

usage() {
  cat <<'USAGE'
Usage: capture_forensic_sweep.sh [options]

Optional (common):
  --drive <N>            DTC drive index (default: 0)
  --dtc-path <path>      Path to dtc executable (default: dtc)
  --out-dir <dir>        Output directory (default: ./captures)
  --label <name>         Label used in filenames (default: sweep)
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
  --sweeps <N>           Number of sweeps (default: 3). Each sweep goes forward then backward.
  --revs <N>             Revolutions per capture (default: 3)
  --cooldown <sec>       Pause between captures (default: 3)
  --spinup <sec>         Spin-up delay before each side (default: 2)
USAGE
}

# Defaults
DRIVE=0
DTC_BIN="/usr/bin/dtc"
OUT_DIR="/srv/kryoflux/captures"
LABEL="sweep"
USE_SUDO=1
DRY_RUN=0
PROFILE=""
SIDES="both"
START_TRACK=""
END_TRACK=""
STEP=1
SWEEPS=3
REVS=3
COOLDOWN=3
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
    --sweeps) SWEEPS=${2:?}; shift 2;;
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
LOG_PATH="$RUN_DIR/run.log"

SUDO_PREFIX=""; [[ $USE_SUDO -eq 1 ]] && SUDO_PREFIX="sudo " || true

pushd "$RUN_DIR" >/dev/null
umask 0002

{
  echo "capture_forensic_sweep.sh run at $(date -Iseconds)"
  echo "dtc path: $(command -v "$DTC_BIN" || echo "$DTC_BIN")"
  echo "dtc header:"; "$DTC_BIN" 2>&1 | head -n 4; echo
  echo "Profile: ${PROFILE}  Sides: ${SIDES}  Tracks: ${START_TRACK}..${END_TRACK} step ${STEP}  Sweeps: ${SWEEPS}  Revs: ${REVS}"
  echo "Cooldown: ${COOLDOWN}s  Spinup: ${SPINUP}s"
} > "$LOG_PATH"

build_read_cmd() {
  local start=$1 end=$2 side=$3
  local cmd="${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -i0 -s${start} -e${end} -g${side} -r${REVS} -ftrack"
  if (( RICH == 1 )); then
    if (( NO_ML == 0 )); then cmd+=" -m2 -l63"; fi
    cmd+=" -t1 -k1"
    if (( NO_P == 0 )); then cmd+=" -p"; fi
  fi
  echo "$cmd"
}

run_read() {
  local track=$1 side=$2
  local cmd=$(build_read_cmd "$track" "$track" "$side")
  echo "[READ ] $cmd" | tee -a "$LOG_PATH"
  if [[ $DRY_RUN -eq 0 ]]; then
    eval "$cmd" | tee -a "$LOG_PATH"
  fi
}

capture_side() {
  local side=$1
  echo "-- Side ${side} spin-up: ${SPINUP}s" | tee -a "$LOG_PATH"
  sleep "$SPINUP"
  for ((sw=1; sw<=SWEEPS; sw++)); do
    echo "-- Sweep ${sw} forward" | tee -a "$LOG_PATH"
    local t=$START_TRACK
    while (( t <= END_TRACK )); do
      run_read "$t" "$side"
      echo "Cooldown ${COOLDOWN}s..." | tee -a "$LOG_PATH"; sleep "$COOLDOWN"
      t=$(( t + STEP ))
    done
    echo "-- Sweep ${sw} backward" | tee -a "$LOG_PATH"
    t=$END_TRACK
    while (( t >= START_TRACK )); do
      run_read "$t" "$side"
      echo "Cooldown ${COOLDOWN}s..." | tee -a "$LOG_PATH"; sleep "$COOLDOWN"
      t=$(( t - STEP ))
    done
  done
}

# Execute
if [[ "$SIDES" == "both" || "$SIDES" == "0" ]]; then
  capture_side 0
fi
if [[ "$SIDES" == "both" || "$SIDES" == "1" ]]; then
  capture_side 1
fi

echo "Done. Outputs in $RUN_DIR" | tee -a "$LOG_PATH"
shopt -s nullglob; files=(track*.raw)
if (( ${#files[@]} == 0 )); then
  echo "ERROR: No track*.raw files were written. Listing directory:" | tee -a "$LOG_PATH"
  pwd | tee -a "$LOG_PATH"
  ls -la | tee -a "$LOG_PATH"
  echo "Hints: reset KryoFlux/drive; try --rich/--no-p/--no-ml toggles; verify permissions." | tee -a "$LOG_PATH"
fi
popd >/dev/null
