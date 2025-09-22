#!/usr/bin/env bash
set -euo pipefail

# dtc_probe.sh
# Utility to probe drive capacity and RPM using DTC.
#
# Capacity check:
#   dtc -d<drive> -c2
# RPM check (requires Ctrl+C normally):
#   timeout --signal INT <secs> dtc -d<drive> -c3
#
# Usage examples:
#   ./dtc_probe.sh --drive 0 --capacity
#   ./dtc_probe.sh --drive 0 --rpm --seconds 6
#   ./dtc_probe.sh --drive 0 --capacity --rpm --seconds 8
#
# Notes:
# - Output includes a raw section and a parsed summary when possible.
# - This script does not require a disk to be safe, but RPM needs a spinning disk.

DRIVE=0
DTC_BIN="/usr/bin/dtc"
USE_SUDO=1
DO_CAP=0
DO_RPM=0
RPM_SECONDS=6

while [[ $# -gt 0 ]]; do
  case "$1" in
    --drive) DRIVE=${2:?}; shift 2;;
    --dtc-path) DTC_BIN=${2:?}; shift 2;;
    --no-sudo) USE_SUDO=0; shift 1;;
    --capacity) DO_CAP=1; shift 1;;
    --rpm) DO_RPM=1; shift 1;;
    --seconds) RPM_SECONDS=${2:?}; shift 2;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done

if ! command -v "$DTC_BIN" >/dev/null 2>&1 && [[ ! -x "$DTC_BIN" ]]; then
  echo "Error: dtc not found: $DTC_BIN" >&2
  exit 2
fi

SUDO_PREFIX=""; [[ $USE_SUDO -eq 1 ]] && SUDO_PREFIX="sudo " || true

echo "dtc_probe.sh run at $(date -Iseconds)"
"$DTC_BIN" 2>&1 | head -n 4 || true

if (( DO_CAP == 1 )); then
  echo
  echo "== Capacity probe (dtc -d${DRIVE} -c2) =="
  RAW=$(bash -lc "${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -c2" 2>&1 || true)
  echo "$RAW"
  # Best-effort parse: look for tracks mentioned as numbers and min/max them
  TRACKS=$(echo "$RAW" | grep -Eo '\\b[0-9]{1,3}\\b' | tr '\n' ' ')
  if [[ -n "$TRACKS" ]]; then
    MIN=$(echo "$TRACKS" | tr ' ' '\n' | awk 'NR==1{min=$1} {if($1<min)min=$1} END{print min}')
    MAX=$(echo "$TRACKS" | tr ' ' '\n' | awk 'NR==1{max=$1} {if($1>max)max=$1} END{print max}')
    echo "Summary: observed track numbers range from $MIN to $MAX"
  fi
fi

if (( DO_RPM == 1 )); then
  echo
  echo "== RPM probe (timeout --signal INT ${RPM_SECONDS}s dtc -d${DRIVE} -c3) =="
  RAW=$(timeout --signal INT ${RPM_SECONDS} bash -lc "${SUDO_PREFIX}${DTC_BIN} -d${DRIVE} -c3" 2>&1 || true)
  echo "$RAW"
  # Parse rpm values like 'rpm: 301.2' or 'rpm: 301.284'
  RPMS=$(echo "$RAW" | grep -Eo 'rpm: *[0-9]+(\\.[0-9]+)?' | awk '{print $2}')
  if [[ -n "$RPMS" ]]; then
    COUNT=$(echo "$RPMS" | wc -w | tr -d ' ')
    SUM=$(echo "$RPMS" | awk '{s+=$1} END{print s}')
    AVG=$(awk -v s="$SUM" -v c="$COUNT" 'BEGIN{if(c>0) printf("%.3f", s/c); else print "0"}')
    echo "Summary: samples=$COUNT average_rpm=${AVG}"
  fi
fi
