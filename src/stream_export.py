import struct
from pathlib import Path
from typing import List
from datetime import datetime
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
    sck_hz: float = 24027428.5714285,
    rev_lengths: List[int] | None = None,
    header_mode: str = 'oob',
    include_sck_oob: bool = True,
    include_initial_index: bool = False,
    ick_hz: float = 3003428.5714285625,
    include_kf_version_info: bool = True,
    include_clock_info: bool = False,
    include_hw_info: bool = False,
    include_streaminfo: bool = True,
    streaminfo_chunk_bytes: int = 32756,
    streaminfo_transfer_ms: int = 170,
) -> None:
    """
    Write a KryoFlux C2/OOB stream with spec-compliant ISB and OOB blocks.
    Default layout (HxC-compatible):
      - OOB KFInfo (type=0x04) at byte 0: ASCII "KryoFlux stream - version {version}" (no trailing NUL)
      - OOB StreamInfo (type=0x01): initial (StreamPosition=0, TransferTimeMs=0) and periodic thereafter.
        Periodic payload StreamPosition equals the actual ISB byte count at the moment of insertion.
      - Encoded ISB flux blocks (C2 timing values in sample ticks)
      - OOB Index (type=0x02, 12-byte LE payload) after each revolution boundary
      - OOB StreamEnd (type=0x03, 8-byte LE payload)
      - OOB EOF (type=0x0D, size=0x0D0D)
    ISB encoding rules:
      - Flux1: 0x0E..0xFF (1 byte) => value = header (14..255)
      - Flux2: 0x00..0x07 + 1 byte => value = (header<<8) + value1 (0..0x7FF)
      - Flux3: 0x0C + 2 bytes => value = (value1<<8) + value2 (MSB then LSB)
      - Ovl16: 0x0B adds 0x10000 to the next flux block; emit repeats as needed for very large values
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def ns_to_ticks(ns_val: int | float) -> int:
        return max(0, int(round((float(ns_val) * sck_hz) / 1e9)))

    def encode_ticks(ticks: int) -> bytes:
        """
        Encode a single flux interval (in sample ticks) into KryoFlux C2 ISB blocks.
        Spec mapping:
          - Flux1: 0x0E..0xFF for values 14..255
          - Flux2: 0x00..0x07, then one byte => (hdr<<8)+val (0..0x7FF)
          - Flux3: 0x0C, then two bytes => (b1<<8)+b2 (MSB then LSB) for 0x0800..0xFFFF
          - Ovl16: 0x0B increments the next flux value by 0x10000 per occurrence
        """
        if ticks < 0:
            ticks = 0
        parts = bytearray()
        # Emit overflow markers for values > 0xFFFF
        while ticks > 0xFFFF:
            parts.append(0x0B)  # Ovl16
            ticks -= 0x10000
        if ticks >= 14 and ticks <= 0xFF:
            # Flux1
            parts.append(ticks & 0xFF)
        elif ticks <= 0x7FF:
            # Flux2
            hdr = (ticks >> 8) & 0x07
            val = ticks & 0xFF
            parts.append(hdr)
            parts.append(val)
        else:
            # Flux3
            parts.append(0x0C)
            b1 = (ticks >> 8) & 0xFF  # MSB
            b2 = ticks & 0xFF         # LSB
            parts.append(b1)
            parts.append(b2)
        return bytes(parts)

    def oob_block(typ: int, payload: bytes = b"") -> bytes:
        # Generic OOB: 0x0D, Type, Size(LE16), Payload
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

    # Build stream starting with KFInfo (Type 0x04), matching real DTC ordering/content.
    stream = bytearray()
    # 0) Optional ASCII preamble is written later when header_mode == 'ascii'.
    # 1) Short KFInfo with host date/time and hc=0 (optional; off by default)
    now = datetime.now()
    host_date = now.strftime('%Y.%m.%d')
    host_time = now.strftime('%H:%M:%S')
    if include_hw_info:
        info1 = f"host_date={host_date}, host_time={host_time}, hc=0"
        stream.extend(oob_block(0x04, info1.encode('ascii')))
    # 2) Long KFInfo with device/version/hwid/hwrv/hs and clocks
    if include_hw_info:
        dev_date = now.strftime('%b %d %Y')
        dev_time = now.strftime('%H:%M:%S')
        info2 = (
            f"name=KryoFlux DiskSystem, version={version}, date={dev_date}, time={dev_time}, "
            f"hwid=1, hwrv=1, hs=1, sck={float(sck_hz):.13f}, ick={float(ick_hz):.13f}"
        )
        stream.extend(oob_block(0x04, info2.encode('ascii')))
    # 3) Optional standalone version string as KFInfo (rare in real streams)
    if include_kf_version_info:
        vtxt = f"KryoFlux stream - version {version}"
        stream.extend(oob_block(0x04, vtxt.encode('ascii')))
    # Optional initial index marker at start-of-stream (counters at 0)
    # Using full 12-byte payload per spec (LE32: StreamPosition, SampleCounter, IndexCounter)
    isb_bytes = 0
    total_sck_ticks = 0  # cumulative SCK ticks across stream
    index_counter = 0    # in ICK cycles (derived from total SCK time)
    # StreamInfo scheduling
    next_streaminfo = 0
    if include_streaminfo:
        # Emit initial StreamInfo with position=0 and transfer time=0
        stream.extend(oob_block(0x01, struct.pack('<II', 0, 0)))
        next_streaminfo = int(streaminfo_chunk_bytes)
    if include_initial_index:
        payload = struct.pack('<III', int(isb_bytes), 0, int(index_counter))
        stream.extend(oob_block(0x02, payload))

    pos = 0
    for i, cnt in enumerate(splits):
        if cnt <= 0:
            # Still write an index block for an empty revolution
            payload = struct.pack('<III', int(isb_bytes), 0, int(index_counter))
            stream.extend(oob_block(0x02, payload))
            continue
        rev = intervals[pos:pos+cnt]
        pos += cnt
        # Encode each ns interval into C2 ticks
        rev_ticks = 0
        for ns_val in rev:
            t = ns_to_ticks(int(ns_val))
            enc = encode_ticks(t)
            stream.extend(enc)
            isb_bytes += len(enc)
            total_sck_ticks += int(t)
            rev_ticks += int(t)
            # Periodic StreamInfo based on ISB byte count; payload SP = actual ISB bytes at insertion
            if include_streaminfo and next_streaminfo > 0:
                while isb_bytes >= next_streaminfo:
                    sp = int(isb_bytes)
                    tt = int(streaminfo_transfer_ms)
                    stream.extend(oob_block(0x01, struct.pack('<II', sp, tt)))
                    next_streaminfo += int(streaminfo_chunk_bytes)
        # At the index boundary, SampleCounter is ticks since last flux to index.
        # Our generator aligns the boundary exactly, so this is typically 0.
        sample_since_last_flux = 0
        # Compute IndexCounter via elapsed seconds * ick_hz using total_sck_ticks
        elapsed_seconds = float(total_sck_ticks) / float(sck_hz) if sck_hz > 0 else 0.0
        index_counter = int(round(elapsed_seconds * float(ick_hz)))
        # OOB Index marker (Type 0x02) with 12-byte payload
        payload = struct.pack('<III', int(isb_bytes), int(sample_since_last_flux), int(index_counter))
        stream.extend(oob_block(0x02, payload))

    # OOB StreamEnd (Type 0x03): 8-byte payload (LE32 StreamPosition, LE32 ResultCode)
    stream.extend(oob_block(0x03, struct.pack('<II', int(isb_bytes), 0)))
    # OOB EOF (Type 0x0D) with size 0x0D0D
    eof_size = bytes((0x0D, 0x0D))
    stream.extend(bytes((0x0D, 0x0D)) + eof_size)

    with open(p, 'wb') as f:
        # Header mode: ascii (legacy preamble) or oob (KFInfo first)
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
