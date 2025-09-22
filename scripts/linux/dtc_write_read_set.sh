#!/usr/bin/env bash
set -euo pipefail

# dtc_write_read_set.sh
# Write a full set of NN.S.raw streams (KryoFlux-style) and optionally read back captures.
# Intended to run on the Linux host connected to the KryoFlux board.
# Requires sudo for DTC operations on most systems.
#
# Example:
#   ./dtc_write_read_set.sh \
#     --image-dir /path/to/disk_image --drive 0 --revs 3 --read-back \
#     --tracks 0-80 --sides 0,1
#
# File naming expected in --image-dir:
#   NN.S.raw (zero-padded track, single-digit side), e.g., 00.0.raw, 80.1.raw

usage() {
  cat <<'USAGE'
Usage: dtc_write_read_set.sh [options]

Required:
  --image-dir <dir>      Directory containing NN.S.raw files

Optional:
  --drive <N>            Drive index for dtc -d (default: 0)
  --dtc-path <path>      Path to dtc executable (default: dtc in PATH)
  --tracks <spec>        Track filter: 'a-b' or comma list (e.g., '0-80' or '0,1,2')
  --sides <list>         Side filter: comma list (default: 0,1)
  --revs <N>             Revolutions for read-back (default: 3)
  --out-dir <dir>        Output dir for read-back captures and logs (default: ./captures)
  --read-back            After writing, read back captures for the filtered set
  --no-sudo              Do not prefix dtc commands with sudo (default: sudo on)
  --dry-run              Print commands without executing
  -h|--help              Show this help

Notes:
- Safe track limits (default expectations): 3.5" drives 0-80, 5.25" drives 0-81.
  Do not exceed your hardware limits.
- Ensure your set is complete for your chosen range.
USAGE
}

IMAGE_DIR=""
DRIVE=0
DTC_BIN="/usr/bin/dtc"
TRACKS_SPEC=""
SIDES_SPEC="0,1"
REVS=3
# Resolve script directory for repo-relative default output
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../../output_captures"
READ_BACK=0
USE_SUDO=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-dir) IMAGE_DIR=${2:?}; shift 2;;
    --drive) DRIVE=${2:?}; shift 2;;
    --dtc-path) DTC_BIN=${2:?}; shift 2;;
    --tracks) TRACKS_SPEC=${2:?}; shift 2;;
    --sides) SIDES_SPEC=${2:?}; shift 2;;
    --revs) REVS=${2:?}; shift 2;;
    --out-dir) OUT_DIR=${2:?}; shift 2;;
    --read-back) READ_BACK=1; shift 1;;
    --no-sudo) USE_SUDO=0; shift 1;;
    --dry-run) DRY_RUN=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$IMAGE_DIR" ]]; then
  echo "Error: --image-dir is required" >&2
  usage
  exit 2
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Error: image dir not found: $IMAGE_DIR" >&2
  exit 2
fi

# Resolve dtc
if ! command -v "$DTC_BIN" >/dev/null 2>&1; then
  if [[ -x "$DTC_BIN" ]]; then
    :
  else
    echo "Error: dtc not found: $DTC_BIN" >&2
    exit 2
  fi
fi

mkdir -p "$OUT_DIR"
LOG_PATH="$OUT_DIR/dtc_write_read_set_$(date +"%Y%m%d_%H%M%S").log"

SUDO_PREFIX=""
if [[ $USE_SUDO -eq 1 ]]; then
  SUDO_PREFIX="sudo "
fi

parse_tracks() {
  local spec="$1"
  if [[ -z "$spec" ]]; then
    echo ""
    return 0
  fi
  if [[ "$spec" == *-* && "$spec" != *","* ]]; then
    local a=${spec%-*}
    local b=${spec#*-}
    if (( a > b )); then local t=$a; a=$b; b=$t; fi
    seq $a $b
  else
    echo "$spec" | tr ',' '\n'
  fi
}

parse_sides() {
  local spec="$1"
  if [[ -z "$spec" ]]; then
    echo -e "0\n1"
    return 0
  fi
  echo "$spec" | tr ',' '\n'
}

# Build list of NN.S.raw files, optionally filter tracks/sides
mapfile -t ALL_FILES < <(find "$IMAGE_DIR" -maxdepth 1 -type f -name "*.raw" | sort)

# Extract candidates as "T:S:path"
CANDIDATES=()
for f in "${ALL_FILES[@]}"; do
  base=$(basename "$f")
  # Match NN.S.raw
  if [[ "$base" =~ ^([0-9]{2})\.([01])\.raw$ ]]; then
    t=${BASH_REMATCH[1]}
    s=${BASH_REMATCH[2]}
    CANDIDATES+=("$((10#$t)):$s:$f")
    continue
  fi
  # Match trackNN.S.raw
  if [[ "$base" =~ ^track([0-9]{2})\.([01])\.raw$ ]]; then
    t=${BASH_REMATCH[1]}
    s=${BASH_REMATCH[2]}
    CANDIDATES+=("$((10#$t)):$s:$f")
    continue
  fi
done

# Apply filters
TRACK_FILTER=($(parse_tracks "$TRACKS_SPEC"))
SIDE_FILTER=($(parse_sides "$SIDES_SPEC"))

should_keep() {
  local t=$1
  local s=$2
  local keep_t=1
  local keep_s=1
  if [[ ${#TRACK_FILTER[@]} -gt 0 ]]; then
    keep_t=0
    for x in "${TRACK_FILTER[@]}"; do [[ "$x" == "$t" ]] && keep_t=1; done
  fi
  if [[ ${#SIDE_FILTER[@]} -gt 0 ]]; then
    keep_s=0
    for y in "${SIDE_FILTER[@]}"; do [[ "$y" == "$s" ]] && keep_s=1; done
  fi
  [[ $keep_t -eq 1 && $keep_s -eq 1 ]]
}

FILTERED=()
for entry in "${CANDIDATES[@]}"; do
  IFS=':' read -r t s p <<<"$entry"
  if should_keep "$t" "$s"; then
    FILTERED+=("$t:$s:$p")
  fi
done

# Sort by track then side
IFS=$'\n' SORTED=($(printf '%s\n' "${FILTERED[@]}" | sort -t: -k1,1n -k2,2n))
unset IFS

{
  echo "dtc_write_read_set.sh run at $(date -Iseconds)"
  echo "dtc path: $(command -v "$DTC_BIN" || echo "$DTC_BIN")"
  echo "Image dir: $IMAGE_DIR"
  echo "Files to write: ${#SORTED[@]}"
} >"$LOG_PATH"

pushd "$IMAGE_DIR" >/dev/null

# Determine dtc prefix expected by image type 4
PREFIX="./"
if [[ ${#SORTED[@]} -gt 0 ]]; then
  IFS=':' read -r t0 s0 p0 <<<"${SORTED[0]}"
  b0=$(basename "$p0")
  if [[ "$b0" =~ ^track[0-9]{2}\.[01]\.raw$ ]]; then
    PREFIX="track"
  else
    PREFIX="./"
  fi
fi
echo "Using dtc prefix: $PREFIX" | tee -a "$LOG_PATH"

# Write all (use write-from-stream: -i4 with -w); dtc will append NN.S.raw to PREFIX
for entry in "${SORTED[@]}"; do
  IFS=':' read -r t s p <<<"$entry"
  LOG_CMD="${SUDO_PREFIX}${DTC_BIN} -f${PREFIX} -i4 -d${DRIVE} -s${t} -e${t} -g${s} -w"
  echo "[WRITE] $LOG_CMD"
  echo "[WRITE] $LOG_CMD" >>"$LOG_PATH"
  if [[ $DRY_RUN -eq 1 ]]; then continue; fi
  if [[ $USE_SUDO -eq 1 ]]; then
    sudo "$DTC_BIN" -f"$PREFIX" -i4 -d"$DRIVE" -s"$t" -e"$t" -g"$s" -w | tee -a "$LOG_PATH"
  else
    "$DTC_BIN" -f"$PREFIX" -i4 -d"$DRIVE" -s"$t" -e"$t" -g"$s" -w | tee -a "$LOG_PATH"
  fi
  sleep 0.2
done
popd >/dev/null

echo "Write phase complete." | tee -a "$LOG_PATH"

# Optional read-back
if [[ $READ_BACK -eq 1 ]]; then
  mkdir -p "$OUT_DIR"
  pushd "$OUT_DIR" >/dev/null
  for entry in "${SORTED[@]}"; do
    IFS=':' read -r t s p <<<"$entry"
    pref="capture_$(printf "%02d" "$t").$s"
    LOG_CMD="${SUDO_PREFIX}${DTC_BIN} -f${pref} -i0 -d${DRIVE} -s${t} -e${t} -g${s} -r${REVS}"
    echo "[READ ] $LOG_CMD"
    echo "[READ ] $LOG_CMD" >>"$LOG_PATH"
    if [[ $DRY_RUN -eq 1 ]]; then continue; fi
    if [[ $USE_SUDO -eq 1 ]]; then
      sudo "$DTC_BIN" -f"$pref" -i0 -d"$DRIVE" -s"$t" -e"$t" -g"$s" -r"$REVS" | tee -a "$LOG_PATH"
    else
      "$DTC_BIN" -f"$pref" -i0 -d"$DRIVE" -s"$t" -e"$t" -g"$s" -r"$REVS" | tee -a "$LOG_PATH"
    fi
    sleep 0.2
  done
  popd >/dev/null
  echo "Read-back phase complete. Captures in $OUT_DIR" | tee -a "$LOG_PATH"
fi

echo "Log saved to: $LOG_PATH"
