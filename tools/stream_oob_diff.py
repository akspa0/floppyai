#!/usr/bin/env python3
"""
stream_oob_diff.py

Compare OOB sequences of two KryoFlux STREAM files and print a structured diff.

Usage:
  python tools/stream_oob_diff.py --a path/to/real.raw --b path/to/generated.raw [--max 0]

It prints, for each file:
- OOB blocks (offset, type, size, parsed fields for known types)
Then prints a side-by-side diff by block index, reporting mismatches in type/size
and key parsed fields (KFInfo text, StreamInfo SP/TT, Index SP/SC/IC, StreamEnd SP/RC, EOF size).

This is read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_oob(data: bytes) -> List[Tuple[int, int, int, dict]]:
    pos = 0
    n = len(data)
    out: List[Tuple[int, int, int, dict]] = []

    def u16le(lo: int, hi: int) -> int:
        return (hi << 8) | lo

    while pos + 4 <= n:
        b = data[pos]
        if b == 0x0D:
            typ = data[pos + 1]
            size = u16le(data[pos + 2], data[pos + 3])
            start = pos
            payload_start = pos + 4
            payload_end = payload_start + size
            if typ == 0x0D:
                out.append((start, typ, size, {"EOF.size": size}))
                break
            if payload_end > n:
                out.append((start, typ, size, {"error": "truncated"}))
                break
            parsed: dict = {}
            if typ == 0x04:
                # KFInfo
                try:
                    txt = data[payload_start:payload_end].split(b"\x00", 1)[0].decode("ascii", errors="replace")
                except Exception:
                    txt = "<decode error>"
                parsed = {"KFInfo": txt}
            elif typ == 0x01 and size == 8:
                sp = int.from_bytes(data[payload_start:payload_start+4], "little")
                tt = int.from_bytes(data[payload_start+4:payload_start+8], "little")
                parsed = {"StreamInfo.StreamPosition": sp, "StreamInfo.TransferTimeMs": tt}
            elif typ == 0x02 and size == 12:
                sp = int.from_bytes(data[payload_start:payload_start+4], "little")
                sc = int.from_bytes(data[payload_start+4:payload_start+8], "little")
                ic = int.from_bytes(data[payload_start+8:payload_start+12], "little")
                parsed = {"Index.StreamPosition": sp, "Index.SampleCounter": sc, "Index.IndexCounter": ic}
            elif typ == 0x03 and size == 8:
                sp = int.from_bytes(data[payload_start:payload_start+4], "little")
                rc = int.from_bytes(data[payload_start+4:payload_start+8], "little")
                parsed = {"StreamEnd.StreamPosition": sp, "StreamEnd.ResultCode": rc}
            else:
                parsed = {"payload_len": size}
            out.append((start, typ, size, parsed))
            pos = payload_end
            continue
        # Non-OOB byte; consume until next potential OOB or EOF
        pos += 1
    return out


def _print_oobs(label: str, oobs: List[Tuple[int, int, int, dict]]) -> None:
    print(f"\n== {label} OOBs ==")
    for (off, typ, size, parsed) in oobs:
        print(f"  @0x{off:08X} type=0x{typ:02X} size={size}")
        if parsed:
            for k, v in parsed.items():
                print(f"    - {k}: {v}")


def _diff(a: List[Tuple[int, int, int, dict]], b: List[Tuple[int, int, int, dict]]) -> None:
    print("\n== OOB Diff (by index) ==")
    m = max(len(a), len(b))
    for i in range(m):
        if i >= len(a):
            print(f"[{i:02d}] only in B: type=0x{b[i][1]:02X} size={b[i][2]}")
            continue
        if i >= len(b):
            print(f"[{i:02d}] only in A: type=0x{a[i][1]:02X} size={a[i][2]}")
            continue
        offA, typA, sizeA, pA = a[i]
        offB, typB, sizeB, pB = b[i]
        mismatch = []
        if typA != typB:
            mismatch.append(f"type A=0x{typA:02X} B=0x{typB:02X}")
        if sizeA != sizeB:
            mismatch.append(f"size A={sizeA} B={sizeB}")
        # Compare parsed keys we know about
        keys = set(pA.keys()).union(pB.keys())
        for k in sorted(keys):
            if pA.get(k) != pB.get(k):
                mismatch.append(f"{k} A={pA.get(k)} B={pB.get(k)}")
        if mismatch:
            print(f"[{i:02d}] @A:0x{offA:08X} @B:0x{offB:08X} -> " + "; ".join(mismatch))


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff OOB sequences between two KryoFlux STREAM files")
    ap.add_argument("--a", required=True, help="Path to first .raw (reference)")
    ap.add_argument("--b", required=True, help="Path to second .raw (candidate)")
    ap.add_argument("--max", type=int, default=0, help="Max bytes to read from each (0=all)")
    args = ap.parse_args()

    pa = Path(args.a)
    pb = Path(args.b)
    if not pa.is_file():
        print(f"Missing: {pa}")
        return 2
    if not pb.is_file():
        print(f"Missing: {pb}")
        return 2

    da = pa.read_bytes()
    db = pb.read_bytes()
    if args.max > 0:
        da = da[: args.max]
        db = db[: args.max]

    oA = _parse_oob(da)
    oB = _parse_oob(db)

    _print_oobs("A (reference)", oA)
    _print_oobs("B (candidate)", oB)
    _diff(oA, oB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
