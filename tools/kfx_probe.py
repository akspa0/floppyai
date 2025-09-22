#!/usr/bin/env python3
"""
kfx_probe.py

Probe a KryoFlux STREAM file to dump OOB blocks, ISB opcode histogram, and sanity
metrics. Useful for comparing generated streams against DTC-captured references.

Usage:
  python3 FloppyAI/tools/kfx_probe.py --input path/to/track00.0.raw [--max 0]

Notes:
- This is a read-only inspection tool. It does not modify input files.
- It parses the STREAM byte-wise and attempts to be resilient to unknown OOB types.
- It reports:
  * OOB blocks (type, size, file offsets; parsed fields for known types)
  * ISB opcode histogram: Flux1, Flux2, Flux3, OVL16, NOP1..3
  * Flux value count and sum of ticks; total ISB bytes encountered
  * EOF presence
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple


def parse_stream(data: bytes) -> Dict[str, object]:
    pos = 0
    n = len(data)

    # Stats
    oob_list = []  # (offset, type, size, parsed)
    flux_count = 0
    flux_sum_ticks = 0
    eof_found = False
    isb_bytes = 0

    hist = {
        "FLUX1": 0,
        "FLUX2": 0,
        "FLUX3": 0,
        "OVL16": 0,
        "NOP1": 0,
        "NOP2": 0,
        "NOP3": 0,
        "OOB": 0,
    }

    def read_u16_le(b0: int, b1: int) -> int:
        return (b1 << 8) | b0

    # Overflow accumulator applies to the next flux block only
    ovl16_count = 0

    while pos < n:
        b = data[pos]
        # OOB block
        if b == 0x0D and pos + 3 < n:
            oob_type = data[pos + 1]
            size_le = data[pos + 2] | (data[pos + 3] << 8)
            oob_start = pos
            payload_start = pos + 4
            payload_end = payload_start + size_le
            # Special handling for EOF sentinel: type=0x0D, size typically 0x0D0D, no payload
            if oob_type == 0x0D:
                eof_found = True
                parsed = {"EOF.size": size_le}
                hist["OOB"] += 1
                oob_list.append((oob_start, oob_type, size_le, parsed))
                # Stop parsing at EOF regardless of indicated size
                break
            if payload_end > n:
                # Truncated OOB (but not EOF); stop
                oob_list.append((oob_start, oob_type, size_le, {"error": "truncated"}))
                break
            parsed = {}
            # Known OOB types
            if oob_type == 0x04:
                # KFInfo ASCII string
                try:
                    s = data[payload_start:payload_end].split(b"\x00", 1)[0].decode("ascii", errors="replace")
                except Exception:
                    s = "<decode error>"
                parsed = {"KFInfo": s}
            elif oob_type == 0x02 and size_le == 12:
                sp = int.from_bytes(data[payload_start:payload_start+4], "little", signed=False)
                sc = int.from_bytes(data[payload_start+4:payload_start+8], "little", signed=False)
                ic = int.from_bytes(data[payload_start+8:payload_start+12], "little", signed=False)
                parsed = {"Index.StreamPosition": sp, "Index.SampleCounter": sc, "Index.IndexCounter": ic}
            elif oob_type == 0x03 and size_le == 8:
                sp = int.from_bytes(data[payload_start:payload_start+4], "little", signed=False)
                rc = int.from_bytes(data[payload_start+4:payload_start+8], "little", signed=False)
                parsed = {"StreamEnd.StreamPosition": sp, "StreamEnd.ResultCode": rc}
            elif oob_type == 0x01 and size_le == 8:
                sp = int.from_bytes(data[payload_start:payload_start+4], "little", signed=False)
                tt = int.from_bytes(data[payload_start+4:payload_start+8], "little", signed=False)
                parsed = {"StreamInfo.StreamPosition": sp, "StreamInfo.TransferTimeMs": tt}
            else:
                parsed = {"payload_len": size_le}
            hist["OOB"] += 1
            oob_list.append((oob_start, oob_type, size_le, parsed))
            pos = payload_end
            # OOB bytes are not part of ISB bytes. Continue.
            continue

        # ISB path
        if b == 0x0B:
            # OVL16
            hist["OVL16"] += 1
            ovl16_count += 1
            pos += 1
            continue
        if b == 0x0C:
            # Flux3: needs 2 more bytes
            if pos + 2 >= n:
                break
            b1 = data[pos + 1]
            b2 = data[pos + 2]
            val = ((b1 << 8) | b2) + (ovl16_count * 0x10000)
            ovl16_count = 0
            hist["FLUX3"] += 1
            flux_count += 1
            flux_sum_ticks += val
            isb_bytes += 3
            pos += 3
            continue
        if b in (0x08, 0x09, 0x0A):
            # NOPs
            if b == 0x08:
                hist["NOP1"] += 1
                isb_bytes += 1
                pos += 1
            elif b == 0x09:
                hist["NOP2"] += 1
                isb_bytes += 2
                pos += 2
            else:
                hist["NOP3"] += 1
                isb_bytes += 3
                if pos + 2 >= n:
                    break
                pos += 3
            continue
        if 0x00 <= b <= 0x07:
            # Flux2: needs 1 more byte
            if pos + 1 >= n:
                break
            v1 = data[pos + 1]
            val = ((b & 0x07) << 8) + v1 + (ovl16_count * 0x10000)
            ovl16_count = 0
            hist["FLUX2"] += 1
            flux_count += 1
            flux_sum_ticks += val
            isb_bytes += 2
            pos += 2
            continue
        if 0x0E <= b <= 0xFF:
            # Flux1
            val = b + (ovl16_count * 0x10000)
            ovl16_count = 0
            hist["FLUX1"] += 1
            flux_count += 1
            flux_sum_ticks += val
            isb_bytes += 1
            pos += 1
            continue

        # Unknown or stray 0x0D without enough bytes: stop to avoid infinite loop
        break

    return {
        "oob": oob_list,
        "hist": hist,
        "flux_count": flux_count,
        "flux_sum_ticks": flux_sum_ticks,
        "isb_bytes": isb_bytes,
        "eof_found": eof_found,
        "total_bytes": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe a KryoFlux STREAM file for OOB/ISB structure")
    ap.add_argument("--input", required=True, help="Path to .raw stream file")
    ap.add_argument("--max", type=int, default=0, help="Optional max bytes to read (0 = whole file)")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.is_file():
        print(f"Input not found: {p}")
        return 2

    data = p.read_bytes()
    if args.max and args.max > 0:
        data = data[: args.max]

    result = parse_stream(data)
    print(f"File: {p}")
    print(f"Total bytes: {result['total_bytes']}")
    print(f"EOF present: {result['eof_found']}")
    print("\nISB Histogram:")
    for k in ("FLUX1", "FLUX2", "FLUX3", "OVL16", "NOP1", "NOP2", "NOP3", "OOB"):
        print(f"  {k:6s}: {result['hist'][k]}")
    print(f"Flux values parsed: {result['flux_count']}")
    print(f"Sum of ticks: {result['flux_sum_ticks']}")
    print(f"ISB bytes (counted): {result['isb_bytes']}")

    print("\nOOB Blocks:")
    for (off, typ, size, parsed) in result["oob"]:
        print(f"  @0x{off:08X}: type=0x{typ:02X} size={size}")
        if parsed:
            for pk, pv in parsed.items():
                print(f"    - {pk}: {pv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
