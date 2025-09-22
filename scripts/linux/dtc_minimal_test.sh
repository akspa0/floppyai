#!/usr/bin/env bash
set -euo pipefail

# dtc_minimal_test.sh
# Minimal, deterministic dtc STREAM write test to diagnose "dtc reads but writes no files" issues.
# Runs inside a target directory and tries -ftrack first, then falls back to -ftrack.raw.
# Optionally wraps dtc in strace to reveal where it attempts to open/create files.
#
# Usage:
#   ./dtc_minimal_test.sh [--drive N] [--track N] [--side 0|1] [--revs N] [--dir DIR] [--no-sudo] [--strace]
#
# Examples:
#   ./dtc_minimal_test.sh --dir ../../output_captures/test_min
#   ./dtc_minimal_test.sh --drive 0 --track 40 --side 0 --revs 1 --dir ../../output_captures/test_mid --strace
#
# Notes:
# - This uses bare-minimum flags in recommended order: -f<prefix> -i0 -d<i> -s<track> -e<track> -g<side> -r<revs>
# - It will retry with -ftrack.raw if no files are produced by the first attempt.
# - If --strace is provided, dtc is wrapped with: strace -ff -o strace -e openat,creat,write,rename

DRIVE=0
TRACK=0
SIDE=0
REVS=1
# Resolve repo-local default target directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/../../output_captures/test_min"
DTC_BIN="/usr/bin/dtc"
USE_SUDO=1
USE_STRACE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drive) DRIVE=${2:?}; shift 2;;
    --track) TRACK=${2:?}; shift 2;;
    --side)  SIDE=${2:?}; shift 2;;
    --revs)  REVS=${2:?}; shift 2;;
    --dir)   TARGET_DIR=${2:?}; shift 2;;
    --no-sudo) USE_SUDO=0; shift 1;;
    --strace) USE_STRACE=1; shift 1;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

if ! command -v "$DTC_BIN" >/dev/null 2>&1 && [[ ! -x "$DTC_BIN" ]]; then
  echo "Error: dtc not found at $DTC_BIN" >&2; exit 2
fi

mkdir -p "$TARGET_DIR"
# Try to ensure we can write here (common issue)
if command -v id >/dev/null 2>&1; then echo "uid=$(id -u) gid=$(id -g)"; fi
if command -v whoami >/dev/null 2>&1; then echo "whoami=$(whoami)"; fi

# Attempt to claim ownership; ignore errors if we lack privileges
if [[ -n "${SUDO_USER:-}" ]]; then OWNER="$SUDO_USER"; else OWNER="$USER"; fi
sudo chown -R "$OWNER":"$OWNER" "$TARGET_DIR" 2>/dev/null || true

pushd "$TARGET_DIR" >/dev/null
umask 0002

SUDO_PREFIX=""; [[ $USE_SUDO -eq 1 ]] && SUDO_PREFIX="sudo " || true
TRACE_PREFIX=""
if [[ $USE_STRACE -eq 1 ]]; then
  TRACE_PREFIX="strace -ff -o strace -e trace=openat,creat,write,rename "
fi

echo "== dtc header =="
"$DTC_BIN" 2>&1 | head -n 4 || true

echo
CMD1="${SUDO_PREFIX}${TRACE_PREFIX}${DTC_BIN} -ftrack -i0 -d${DRIVE} -s${TRACK} -e${TRACK} -g${SIDE} -r${REVS}"
echo "[TEST] $CMD1"
set +e
bash -lc "$CMD1"
RC1=$?
set -e

shopt -s nullglob
files=(track*.raw)
if (( ${#files[@]} > 0 )); then
  echo "OK: wrote ${#files[@]} file(s):"
  ls -l track*.raw
  popd >/dev/null
  exit 0
fi

echo "No files from first attempt; retrying with -ftrack.raw"
CMD2="${SUDO_PREFIX}${TRACE_PREFIX}${DTC_BIN} -ftrack.raw -i0 -d${DRIVE} -s${TRACK} -e${TRACK} -g${SIDE} -r${REVS}"
echo "[TEST] $CMD2"
set +e
bash -lc "$CMD2"
RC2=$?
set -e

files=(track*.raw)
if (( ${#files[@]} > 0 )); then
  echo "OK: wrote ${#files[@]} file(s) after suffix fallback:"
  ls -l track*.raw
  popd >/dev/null
  exit 0
fi

pwd
ls -la

echo "ERROR: dtc produced no stream files in $TARGET_DIR"
echo "- Return codes: first=$RC1 second=$RC2"
echo "- If --strace was used, inspect ./strace* files for open/create attempts"
echo "- Try power-cycling the KryoFlux and drive, replug USB, and rerun"

popd >/dev/null
exit 1
