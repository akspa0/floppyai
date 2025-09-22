#!/usr/bin/env python3
"""
stream_sanity.py

Lightweight sanity checks for KryoFlux STREAM files (.raw) captured via dtc -i0.

What it does right now (safe, zero-dependency):
- Globs files or accepts explicit paths
- For each file:
  - Prints size (bytes), mtime, and a quick non-empty check
  - Optionally dumps first N bytes in hex for quick visual inspection (--head-bytes)
- Aggregates basic stats across files

Notes
- Parsing OOB control structures to count indices/revolutions is planned, but not implemented here yet.
  The KryoFlux stream protocol is non-trivial; we’ll add an OOB scanner in a later iteration.

Usage examples:
  python3 FloppyAI/tools/stream_sanity.py --glob 'track*.raw'
  python3 FloppyAI/tools/stream_sanity.py --files track00.0.raw track00.1.raw
  python3 FloppyAI/tools/stream_sanity.py --glob 'track*.raw' --head-bytes 64
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import datetime as dt
from typing import List


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def hexdump_head(path: str, n: int) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(n)
        return " ".join(f"{b:02x}" for b in data)
    except Exception as e:
        return f"<error reading head: {e}>"


def describe_file(path: str, head_bytes: int | None) -> dict:
    st = os.stat(path)
    size = st.st_size
    mtime = dt.datetime.fromtimestamp(st.st_mtime).isoformat()
    ok = size > 0
    head = hexdump_head(path, head_bytes) if (head_bytes and head_bytes > 0) else None
    return {
        "path": path,
        "size": size,
        "size_h": human_bytes(size),
        "mtime": mtime,
        "non_empty": ok,
        "head": head,
    }


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Sanity-check KryoFlux STREAM files (.raw)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--glob", dest="glob_pat", help="Glob for files, e.g. 'track*.raw'")
    g.add_argument("--files", nargs="*", help="Explicit list of files")
    ap.add_argument("--head-bytes", type=int, default=0, help="Dump first N bytes in hex (default 0 = off)")
    args = ap.parse_args(argv)

    files: List[str] = []
    if args.glob_pat:
        files = sorted(glob.glob(args.glob_pat))
    else:
        files = args.files or []

    if not files:
        print("No files matched.")
        return 2

    total = 0
    nonempty = 0
    print("STREAM Sanity Report:\n")
    for p in files:
        try:
            info = describe_file(p, args.head_bytes)
        except FileNotFoundError:
            print(f"- {p}: NOT FOUND")
            continue
        total += 1
        if info["non_empty"]:
            nonempty += 1
        print(f"- {os.path.basename(p)}")
        print(f"  path   : {info['path']}")
        print(f"  size   : {info['size_h']} ({info['size']} bytes)")
        print(f"  mtime  : {info['mtime']}")
        print(f"  nonempty: {info['non_empty']}")
        if info["head"] is not None:
            print(f"  head[{args.head_bytes}] : {info['head']}")
        print()

    print("Summary:")
    print(f"- files    : {total}")
    print(f"- non-empty: {nonempty}")
    return 0 if nonempty > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
