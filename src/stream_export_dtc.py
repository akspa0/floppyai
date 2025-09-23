import struct
from pathlib import Path
from typing import List
from datetime import datetime
import numpy as np

# DTC-style STREAM writer
# Matches real DTC OOB ordering/content more closely to maximize acceptance by KryoFlux software.
# Invariants:
# - OOB-first (no ASCII preamble by default)
# - KFInfo #1: "host_date=YYYY.MM.DD, host_time=HH:MM:SS, hc=0"
# - KFInfo #2: "name=KryoFlux DiskSystem, version={version}, date=Mon DD YYYY, time=HH:MM:SS, hwid=1, hwrv=1, hs=1, sck={sck_hz}, ick={ick_hz}"
# - StreamInfo initial (SP=0, ms=0), then periodic with payload SP equal to actual ISB bytes at insertion
# - Index at each rev boundary only; no initial index by default
# - StreamEnd (SP=current ISB bytes, RC=0) then EOF sentinel (0x0D0D)

def write_kryoflux_stream_dtc(
    flux_intervals: List[int],
    track: int,
    side: int,
    output_path: str,
    num_revs: int = 1,
    version: str = '3.00s',
    rpm: float | None = None,
    sck_hz: float = 24027428.5714285,
    rev_lengths: List[int] | None = None,
    header_mode: str = 'oob',
    include_sck_oob: bool = True,  # ignored (we never emit type 0x08)
    include_initial_index: bool = False,
    ick_hz: float = 3003428.5714285625,
    include_kf_version_info: bool = True,  # we always emit DTC-style KFInfo lines
    include_clock_info: bool = True,       # included in long KFInfo line
    include_hw_info: bool = True,          # included in both KFInfo lines
    include_streaminfo: bool = True,
    streaminfo_chunk_bytes: int = 32756,
    streaminfo_transfer_ms: int = 170,
) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def ns_to_ticks(ns_val: int | float) -> int:
        return max(0, int(round((float(ns_val) * sck_hz) / 1e9)))

    def encode_ticks(ticks: int) -> bytes:
        # KryoFlux C2 encoding
        if ticks < 0:
            ticks = 0
        parts = bytearray()
        # Emit overflow markers for values > 0xFFFF
        while ticks > 0xFFFF:
            parts.append(0x0B)  # OVL16
            ticks -= 0x10000
        if 14 <= ticks <= 0xFF:
            parts.append(ticks & 0xFF)  # Flux1
        elif ticks <= 0x7FF:
            hdr = (ticks >> 8) & 0x07
            val = ticks & 0xFF
            parts.append(hdr)          # Flux2 hdr 0..7
            parts.append(val)
        else:
            parts.append(0x0C)         # Flux3
            b1 = (ticks >> 8) & 0xFF   # MSB
            b2 = ticks & 0xFF          # LSB
            parts.append(b1)
            parts.append(b2)
        return bytes(parts)

    def oob_block(typ: int, payload: bytes = b"") -> bytes:
        sz = len(payload)
        return bytes((0x0D, typ, sz & 0xFF, (sz >> 8) & 0xFF)) + payload

    # Prepare revolution splits
    if num_revs <= 0:
        num_revs = 1
    intervals = np.asarray(flux_intervals, dtype=np.uint64)
    n = int(intervals.size)
    if rev_lengths is not None and len(rev_lengths) == num_revs:
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

    # Build OOB header matching DTC ordering/content
    stream = bytearray()

    # KFInfo #1: host date/time + hc=0 (no NUL terminator)
    now = datetime.now()
    host_date = now.strftime('%Y.%m.%d')
    host_time = now.strftime('%H:%M:%S')
    info1 = f"host_date={host_date}, host_time={host_time}, hc=0"
    b_info1 = info1.encode('ascii')
    # CAPS/DTC reference commonly shows size 47 for this OOB; pad with spaces if shorter
    if len(b_info1) < 47:
        b_info1 = b_info1 + b" " * (47 - len(b_info1))
    stream.extend(oob_block(0x04, b_info1))

    # KFInfo #2: device info + sck/ick (no NUL terminator)
    dev_date = now.strftime('%b %d %Y')   # e.g., "Mar 27 2018"
    dev_time = now.strftime('%H:%M:%S')
    # Print clocks matching DTC reference formatting when near canonical values.
    # Reference strings (from real DTC captures):
    #   sck=24027428.5714285
    #   ick=3003428.5714285625
    def fmt_float(v: float) -> str:
        try:
            # Snap to canonical text when within a tiny epsilon
            if abs(v - 24027428.5714285) < 1e-3:
                return '24027428.5714285'
            if abs(v - 3003428.5714285625) < 1e-4:
                return '3003428.5714285625'
            # Fallback: trim to <= 10 decimals sans trailing zeros
            s = f"{float(v):.10f}".rstrip('0').rstrip('.')
            return s
        except Exception:
            return str(v)
    info2 = (
        f"name=KryoFlux DiskSystem, version={version}, date={dev_date}, time={dev_time}, "
        f"hwid=1, hwrv=1, hs=1, sck={fmt_float(sck_hz)}, ick={fmt_float(ick_hz)}"
    )
    b_info2 = info2.encode('ascii')
    # CAPS/DTC reference commonly shows size 141 for this OOB; pad with spaces if shorter
    if len(b_info2) < 141:
        b_info2 = b_info2 + b" " * (141 - len(b_info2))
    stream.extend(oob_block(0x04, b_info2))

    # Optional initial StreamInfo (SP=0, ms=0)
    isb_bytes = 0
    total_sck_ticks = 0
    index_counter = 0

    if include_streaminfo:
        stream.extend(oob_block(0x01, struct.pack('<II', 0, 0)))
        next_streaminfo = int(streaminfo_chunk_bytes)
    else:
        next_streaminfo = 0

    # Optional initial index (rare; default OFF)
    if include_initial_index:
        payload = struct.pack('<III', int(isb_bytes), 0, int(index_counter))
        stream.extend(oob_block(0x02, payload))

    # Encode revolutions
    pos = 0
    last_ticks = 0  # ticks of last emitted flux value (for Index.SampleCounter)
    for cnt in splits:
        if cnt <= 0:
            # Still emit index for empty rev
            payload = struct.pack('<III', int(isb_bytes), 0, int(index_counter))
            stream.extend(oob_block(0x02, payload))
            continue
        rev = intervals[pos:pos+cnt]
        pos += cnt

        for ns_val in rev:
            t = ns_to_ticks(int(ns_val))
            enc = encode_ticks(t)
            stream.extend(enc)
            isb_bytes += len(enc)
            total_sck_ticks += int(t)
            last_ticks = int(t)

            # Periodic StreamInfo: payload SP equals current ISB bytes at insertion
            if include_streaminfo and next_streaminfo > 0:
                while isb_bytes >= next_streaminfo:
                    sp = int(isb_bytes)
                    tt = int(streaminfo_transfer_ms)
                    stream.extend(oob_block(0x01, struct.pack('<II', sp, tt)))
                    next_streaminfo += int(streaminfo_chunk_bytes)

        # At rev boundary emit Index (Type 0x02)
        # Payload fields (per CAPS/C2 definitions): (StreamPosition, timer, systime)
        # For DTC file compatibility in practice:
        # - timer: ticks of the last emitted flux (since previous flux) -> last_ticks
        # - systime: index clock ticks since stream start -> round(elapsed_seconds * ick_hz)
        # The absolute baseline in real captures can be large; we start at ~0 which is acceptable for writing.
        sample_since_last_flux = int(last_ticks)
        elapsed_seconds = float(total_sck_ticks) / float(sck_hz) if sck_hz > 0 else 0.0
        index_counter = int(round(elapsed_seconds * float(ick_hz)))
        payload = struct.pack('<III', int(isb_bytes), int(sample_since_last_flux), int(index_counter))
        stream.extend(oob_block(0x02, payload))

    # Ensure at least one flux cell after the final Index so all parsers register it
    dummy_ticks = int(round(float(sck_hz) * 12e-6))  # ~12us
    if dummy_ticks <= 0:
        dummy_ticks = 1
    enc = encode_ticks(dummy_ticks)
    stream.extend(enc)
    isb_bytes += len(enc)
    total_sck_ticks += int(dummy_ticks)

    # StreamEnd (SP=isb_bytes, RC=0) then EOF (0x0D0D)
    stream.extend(oob_block(0x03, struct.pack('<II', int(isb_bytes), 0)))
    stream.extend(bytes((0x0D, 0x0D, 0x0D, 0x0D)))

    # Write file; DTC-style prefers OOB-first; only write ASCII preamble if explicitly requested
    with open(p, 'wb') as f:
        if str(header_mode).lower() == 'ascii':
            pre_txt = f"KryoFlux stream - version {version}"
            preamble = pre_txt.encode('ascii', errors='ignore') + b"\x00"
            f.write(preamble)
        f.write(stream)

    try:
        print(
            f"Wrote DTC-style KryoFlux C2 stream: track={track} side={side} sck={sck_hz} Hz -> {p}"
        )
    except Exception:
        pass
