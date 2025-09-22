# Active Context — FloppyAI

Last updated: 2025-09-21 20:31 (local)

## Current Focus
- Stabilize Linux DTC read pipeline (now working reliably after flag ordering fix).
- Next: Test our own flux writing from the silkscreen module (image→flux), then write→read→analyze on Linux host.
- Keep forensic‑rich defaults while reducing friction (profiles, sensible revs, cooldowns, logging).

## Recent Changes
- Linux DTC scripts hardened and standardized:
  - Correct dtc flag order everywhere: `-f<prefix>` FIRST, then `-i0` (read) / `-wi4` (write-from-stream).
  - Default outputs are repo-local: `FloppyAI/output_captures/…` (no `/srv/kryoflux`).
  - Default track range now 0..82 (83 tracks) for full-disk/sweep.
  - New options: `--passes` (full-disk multi-pass with per-pass subdirs), `--sanity` (run `tools/stream_sanity.py`), `--corpus` (run `tools/compare_corpus.py`).
  - Repeat/sweep scripts write into distinct subfolders to avoid overwrite (`read_01/`, `sweep_01_fwd/`, …).
- New helper scripts/tools:
  - `scripts/linux/dtc_probe.sh` (capacity `-c2`, RPM `-c3` via timed SIGINT).
  - `scripts/linux/dtc_minimal_test.sh` (bare read sanity, ordered flags).
  - `scripts/linux/experiment_write_read_analyze.sh` (generate→write→read→analyze turnkey).
  - `tools/patterns_to_stream_set.py` (emit valid C2/OOB NN.S.raw for `-wi4`).
  - `tools/analyze_captures.py` (batch FluxAnalyzer over a folder).
  - `tools/compare_corpus.py` (cross-pass diffs: rpm/mean/fluxes).

## Next Steps
- Integrate silkscreen (image→flux) writing tests on Linux: generate streams → `dtc_write_read_set.sh` write/verify → analyze stability.
- Expand stream sanity to parse OOB counts and RPM per revolution; add jitter metrics in comparisons.
- Update docs for experiments using the new turnkey script and corpus comparison.
- Continue Phase 2 cleanup (ensure all JSON writes use `utils.json_io.dump_json`).

## Open Decisions
- Whether to expose a user knob for instability contrast percentile; currently hard‑coded to a sensible default.
- Optional addition of per‑track overlay JSON for zoned GCR in future iterations.

## Links
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
