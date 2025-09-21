import struct
from pathlib import Path
from typing import List

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
        for iv in flux_intervals:
            f.write(struct.pack('<I', int(iv)))


def write_kryoflux_stream(flux_intervals: List[int], track: int, side: int, output_path: str, num_revs: int = 1, version: str = '3.00s', rpm: float = 360.0) -> None:
    """
    Write a pseudo-KryoFlux stream that many tools can ingest. This is not guaranteed
    to satisfy every dtc version but serves as a starting point for hardware trials.
    Strategy:
      - ASCII metadata header (null-terminated) with track/side and nominal clocks
      - Sequence of uint32 intervals with an index marker (0xFFFFFFFF) after each rev
    Notes:
      - We will iterate based on dtc logs from the Linux host if needed.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata header
    metadata = (
        f"host_date=2025.09.20, host_time=00:00:00, hc=0, name=KryoFlux DiskSystem, "
        f"version={version}, date=Sep 20 2025, time=00:00:00, hwvid=1, hwrv=1, hs=0, "
        f"sck=24027428.5714285, ick=3003428.57142857, track={track}, side={side}\x00"
    ).encode('ascii')

    # Write header + flux with index pulses
    with open(p, 'wb') as f:
        f.write(metadata)
        if num_revs <= 0:
            num_revs = 1
        per_rev = max(1, len(flux_intervals) // num_revs)
        pos = 0
        for _ in range(num_revs):
            rev_intervals = flux_intervals[pos:pos + per_rev]
            for iv in rev_intervals:
                f.write(struct.pack('<I', int(iv)))
            # Index marker
            f.write(struct.pack('<I', 0xFFFFFFFF))
            pos += per_rev
