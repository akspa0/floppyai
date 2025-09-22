#!/usr/bin/env bash
set -euo pipefail

# experiment_write_read_analyze.sh
# End-to-end helper to:
#  1) Generate a NN.S.raw stream set from synthetic patterns (C2/OOB)
#  2) Write the set to disk with DTC (-wi4)
#  3) Read back captures with DTC (-i0)
#  4) Analyze captures with FluxAnalyzer (JSON + console summary)
#
# Usage example:
#   ./experiment_write_read_analyze.sh \
#     --tracks 0-82 --sides 0,1 --revs 3 --drive 0 \
#     --label pat_const4us \
#     --pattern constant --interval-ns 4000 --rev-time-ns 200000000 \
#     --sanity
#
# Patterns supported by generator:
#   constant | random | alt | zeros | ones | sweep | prbs7
#
# Examples:
#   # PRBS7 with explicit long/short and seed
#   ./experiment_write_read_analyze.sh \
#     --tracks 0-9 --sides 0 --revs 3 --drive 0 \
#     --label prbs7_L4200_S2200 \
#     --pattern prbs7 --long-ns 4200 --short-ns 2200 --seed 123
#
# Notes:
# - Requires Python 3 environment with repo's FloppyAI/src on sys.path.
# - Uses repo-local output folder 'output_captures/experiments'.
# - Runs dtc with compact flags and proper ordering (-f first).

# Defaults
TRACKS="0-82"
SIDES="0,1"
REVS=3
DRIVE=0
LABEL="experiment"
PATTERN="constant"      # constant|random|alt|zeros|ones|sweep|prbs7
INTERVAL_NS=4000         # for constant
MEAN_NS=4000.0           # for random
STD_NS=400.0             # for random
REV_TIME_NS=200000000    # ~300 RPM
SCK_HZ=24027428.5714285
USE_SUDO=1
SANITY=0
# Extended pattern params
LONG_NS=4000
SHORT_NS=2000
SWEEP_MIN_NS=2000
SWEEP_MAX_NS=6000
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tracks) TRACKS=${2:?}; shift 2;;
    --sides) SIDES=${2:?}; shift 2;;
    --revs) REVS=${2:?}; shift 2;;
    --drive) DRIVE=${2:?}; shift 2;;
    --label) LABEL=${2:?}; shift 2;;
    --pattern) PATTERN=${2:?}; shift 2;;
    --interval-ns) INTERVAL_NS=${2:?}; shift 2;;
    --mean-ns) MEAN_NS=${2:?}; shift 2;;
    --std-ns) STD_NS=${2:?}; shift 2;;
    --rev-time-ns) REV_TIME_NS=${2:?}; shift 2;;
    --sck-hz) SCK_HZ=${2:?}; shift 2;;
    --long-ns) LONG_NS=${2:?}; shift 2;;
    --short-ns) SHORT_NS=${2:?}; shift 2;;
    --sweep-min-ns) SWEEP_MIN_NS=${2:?}; shift 2;;
    --sweep-max-ns) SWEEP_MAX_NS=${2:?}; shift 2;;
    --seed) SEED=${2:?}; shift 2;;
    --no-sudo) USE_SUDO=0; shift 1;;
    --sanity) SANITY=1; shift 1;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_ROOT="$REPO_ROOT/FloppyAI/output_captures/experiments"
RUN_DIR="$OUT_ROOT/${LABEL}_$(date +"%Y%m%d_%H%M%S")"
PAT_DIR="$RUN_DIR/patterns"
CAP_DIR="$RUN_DIR/captures"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$PAT_DIR" "$CAP_DIR"
umask 0002

SUDO_PREFIX=""; [[ $USE_SUDO -eq 1 ]] && SUDO_PREFIX="sudo " || true

TOOLS_DIR="$REPO_ROOT/FloppyAI/tools"
PATTERN_GEN="$TOOLS_DIR/patterns_to_stream_set.py"
ANALYZE="$TOOLS_DIR/analyze_captures.py"
WRITE_READ_SET="$SCRIPT_DIR/dtc_write_read_set.sh"
SANITY_TOOL="$TOOLS_DIR/stream_sanity.py"

{
  echo "experiment_write_read_analyze.sh run at $(date -Iseconds)"
  echo "Tracks=$TRACKS Sides=$SIDES Revs=$REVS Drive=$DRIVE Label=$LABEL"
  echo "Pattern=$PATTERN interval_ns=$INTERVAL_NS mean_ns=$MEAN_NS std_ns=$STD_NS"
  echo "rev_time_ns=$REV_TIME_NS sck_hz=$SCK_HZ"
  echo "long_ns=$LONG_NS short_ns=$SHORT_NS sweep_min_ns=$SWEEP_MIN_NS sweep_max_ns=$SWEEP_MAX_NS seed=$SEED"
  echo "Run dir: $RUN_DIR"
} > "$LOG_FILE"

# 1) Generate pattern set
python3 "$PATTERN_GEN" \
  --tracks "$TRACKS" \
  --sides "$SIDES" \
  --revs "$REVS" \
  --rev-time-ns "$REV_TIME_NS" \
  --sck-hz "$SCK_HZ" \
  --pattern "$PATTERN" \
  --interval-ns "$INTERVAL_NS" \
  --mean-ns "$MEAN_NS" \
  --std-ns "$STD_NS" \
  --long-ns "$LONG_NS" \
  --short-ns "$SHORT_NS" \
  --sweep-min-ns "$SWEEP_MIN_NS" \
  --sweep-max-ns "$SWEEP_MAX_NS" \
  --seed "$SEED" \
  --output-dir "$PAT_DIR" | tee -a "$LOG_FILE"

# 2) Write set and 3) Read back captures
CMD_WR="$WRITE_READ_SET --image-dir \"$PAT_DIR\" --drive \"$DRIVE\" --revs \"$REVS\" --out-dir \"$CAP_DIR\" --read-back --tracks \"$TRACKS\" --sides \"$SIDES\""
if (( USE_SUDO == 0 )); then
  CMD_WR+=" --no-sudo"
fi
echo "Invoking: $CMD_WR" | tee -a "$LOG_FILE"
bash -lc "$CMD_WR" | tee -a "$LOG_FILE"

# 4) Analyze captures
python3 "$ANALYZE" --dir "$CAP_DIR" --glob '*.raw' | tee -a "$LOG_FILE"

# Optional: sanity pass
if (( SANITY == 1 )); then
  pushd "$CAP_DIR" >/dev/null
  python3 "$SANITY_TOOL" --glob '*.raw' | tee -a "$LOG_FILE" || true
  popd >/dev/null
fi

echo "Done. See $RUN_DIR" | tee -a "$LOG_FILE"
