import struct
from pathlib import Path
from typing import List
import numpy as np

# Constants for revolution times (ns)
REV_TIME_NS_300 = 200_000_000  # ~200ms (300 RPM)
REV_TIME_NS_360 = 166_666_667  # ~166.67ms (360 RPM)


def write_internal_raw(flux_intervals: List[int], track: int, side: int, output_path: str, num_revs: int = 1) -> None:
    """
    Write a very simple internal .raw format used by FloppyAI tools for analysis-only.
    Layout:
      - 4 bytes magic: b'FLUX'
      - 4 bytes uint32: number of intervals
      - 4 bytes uint32: number of revolutions
      - 4*count bytes: uint32 little-endian flux intervals (ns)
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    count = len(flux_intervals)
    with open(p, 'wb') as f:
        f.write(b'FLUX')
        f.write(struct.pack('<I', count))
        f.write(struct.pack('<I', int(num_revs)))
        arr = np.asarray(flux_intervals, dtype=np.uint32)
        f.write(arr.tobytes(order='C'))


def write_kryoflux_stream(
    flux_intervals: List[int],
    track: int,
    side: int,
    output_path: str,
    num_revs: int = 1,
    version: str = '3.50',
    rpm: float | None = None,
    sck_hz: float = 24000000.0,
    rev_lengths: List[int] | None = None,
    header_mode: str = 'oob',
    include_sck_oob: bool = True,
    include_initial_index: bool = True,
) -> None:
    """
    Write a KryoFlux C2/OOB stream (simplified but valid) so dtc can ingest it.
    Layout:
      - OOB info block with ASCII text including 'KryoFlux' and sck=
      - Encoded sample stream (C2) of tick counts converted from ns
      - OOB index (type=2) after each revolution
      - OOB stream end (type=3)
    Encoding rules (aligned to our parser in flux_analyzer.py):
      - small sample: 0x00, then 1 byte 0..0x0D for 0..13 ticks
      - single byte sample: 0x0E..0xFF for 14..255 ticks
      - overflow: repeat 0x0B to add 65536 per occurrence, then 0x0C + uint16 LE sample for remaining 0..65535
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def ns_to_ticks(ns_val: int | float) -> int:
        return max(0, int(round((float(ns_val) * sck_hz) / 1e9)))

    def encode_ticks(ticks: int) -> bytes:
        """
        KryoFlux C2 sample encoding (commonly accepted by tools):
          - 0..13 ticks:  0x00, <byte:0..13>
          - 14..255:     <single byte: 0x0E..0xFF>
          - >=256:       0x0B repeated (adds 65536 each), then 0x0C + uint16 LE
        """
        if ticks < 0:
            ticks = 0
        parts = bytearray()
        # Emit overflow markers in chunks of 65536
        while ticks > 0xFFFF:
            parts.append(0x0B)  # c2eOverflow16
            ticks -= 65536
        if ticks <= 13:
            parts.append(0x00)  # small sample prefix
            parts.append(ticks & 0xFF)
        elif ticks <= 255:
            # Single byte sample. 0x0D is reserved (OOB), but ticks in this branch are 14..255.
            parts.append(ticks & 0xFF)
        else:
            parts.append(0x0C)  # c2eValue16
            parts.append(ticks & 0xFF)
            parts.append((ticks >> 8) & 0xFF)
        return bytes(parts)

    def oob_block(typ: int, payload: bytes = b"") -> bytes:
        # 0x0D, type, size (LE 16), payload
        sz = len(payload)
        return bytes((0x0D, typ, sz & 0xFF, (sz >> 8) & 0xFF)) + payload

    # Prepare intervals and per-rev splitting
    if num_revs <= 0:
        num_revs = 1
    intervals = np.asarray(flux_intervals, dtype=np.uint64)
    n = int(intervals.size)
    if rev_lengths is not None and len(rev_lengths) == num_revs:
        # Use exact boundaries provided by the caller
        splits = list(rev_lengths)
    else:
        if num_revs == 1:
            splits = [n]
        else:
            base = n // num_revs
            rem = n % num_revs
            splits = [base] * num_revs
            if rem:
                splits[-1] += rem

    # Build stream that begins with optional sample clock OOB (type=8) and an OOB info block (type=4)
    stream = bytearray()
    if include_sck_oob:
        try:
            sck_u32 = int(round(float(sck_hz))) & 0xFFFFFFFF
            stream.extend(oob_block(8, struct.pack('<I', sck_u32)))
        except Exception:
            pass
    info_txt = f"KryoFlux stream - version {version}\x00"
    stream.extend(oob_block(4, info_txt.encode('ascii')))
    # Initial index marker at start-of-stream
    if include_initial_index:
        stream.extend(oob_block(2, struct.pack('<H', 0)))

    pos = 0
    for i, cnt in enumerate(splits):
        if cnt <= 0:
            # Still write an index for an empty revolution (timer=0)
            stream.extend(oob_block(2, struct.pack('<H', 0)))
            continue
        rev = intervals[pos:pos+cnt]
        pos += cnt
        # Encode each ns interval into C2 ticks
        for ns_val in rev:
            t = ns_to_ticks(int(ns_val))
            stream.extend(encode_ticks(t))
        # OOB index marker with timer payload (0 for now)
        stream.extend(oob_block(2, struct.pack('<H', 0)))

    # OOB stream end
    stream.extend(oob_block(3))

    with open(p, 'wb') as f:
        # Header mode: ascii (preamble) or oob (start with OOB info block)
        if str(header_mode).lower() == 'ascii':
            pre_txt = f"KryoFlux stream - version {version}"
            preamble = pre_txt.encode('ascii', errors='ignore') + b"\x00"
            f.write(preamble)
        # Write C2/OOB stream
        f.write(stream)
    try:
        print(f"Wrote KryoFlux C2 stream: track={track} side={side} sck={sck_hz} Hz -> {p}")
    except Exception:
        pass
