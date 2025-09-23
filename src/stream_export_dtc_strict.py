import struct
from pathlib import Path
from typing import List

# Strict DTC-style STREAM writer (OOB-first only)
# This wraps the baseline DTC writer and enforces DTC-friendly defaults:
# - OOB-first (no ASCII preamble)
# - Two KFInfo lines first (host_date/time; device+clocks)
# - StreamInfo initial (SP=0, ms=0) and periodic
# - Index at revolution boundaries only; SampleCounter = last flux ticks
# - Dummy flux after final Index; then StreamEnd and EOF
#
# NOTE: We intentionally force key flags to reduce variability. Further
# byte-for-byte alignment (e.g., float precision, cadence) will be tuned
# against real captures using tools/stream_oob_diff.py.

from stream_export_dtc import write_kryoflux_stream_dtc as _baseline_write


def write_kryoflux_stream_dtc_strict(
    flux_intervals: List[int],
    track: int,
    side: int,
    output_path: str,
    num_revs: int = 1,
    version: str = "3.00s",
    rpm: float | None = None,
    sck_hz: float = 24027428.5714285,
    rev_lengths: List[int] | None = None,
    # Intentionally ignore header_mode arg from the caller; we force OOB-first
    header_mode: str = "oob",
    include_sck_oob: bool = True,   # ignored (we never emit type 0x08)
    include_initial_index: bool = False,
    ick_hz: float = 3003428.5714285625,
    # We force inclusion of these KFInfo lines and clocks by default
    include_kf_version_info: bool = True,
    include_clock_info: bool = True,
    include_hw_info: bool = True,
    include_streaminfo: bool = True,
    streaminfo_chunk_bytes: int = 32756,
    streaminfo_transfer_ms: int = 170,
) -> None:
    """Write a DTC-compatible STREAM with enforced OOB-first header and KFInfo semantics."""
    # Force OOB-first header regardless of caller input
    forced_header_mode = "oob"
    # Force DTC reference-friendly version string regardless of caller
    forced_version = "3.00s"
    _baseline_write(
        flux_intervals=flux_intervals,
        track=track,
        side=side,
        output_path=output_path,
        num_revs=num_revs,
        version=forced_version,
        rpm=rpm,
        sck_hz=sck_hz,
        rev_lengths=rev_lengths,
        header_mode=forced_header_mode,
        include_sck_oob=include_sck_oob,
        include_initial_index=include_initial_index,
        ick_hz=ick_hz,
        include_kf_version_info=True,
        include_clock_info=True,
        include_hw_info=True,
        include_streaminfo=include_streaminfo,
        streaminfo_chunk_bytes=streaminfo_chunk_bytes,
        streaminfo_transfer_ms=streaminfo_transfer_ms,
    )
