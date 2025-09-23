# FloppyAI Refactor Plan

Goal: reduce per-file size to ~750–800 LOC, improve modularity, and keep the CLI stable (`python -m src.main`).

## Principles
- Thin CLI in `main.py` that only wires argparse subcommands to implementation modules.
- Clear module boundaries: commands, analysis, rendering, detection, utils.
- No behavior changes during refactor (pure moves + imports).
- Each step compiles and runs independently.

## Target Module Layout
```
src/
  main.py                              # CLI entry, argparse only
  overlay_detection.py                 # MFM/GCR overlay detection (done)
  rendering.py                         # Polar/overlay/instability rendering (done)
  analysis/
    analyze_disk.py                    # Full analyze_disk pipeline (phase 3)
  cmd_corpus.py                        # analyze_corpus (phase 1)
  cmd_diff.py                          # compare_reads (phase 1)
  cmd_stream_ops.py                    # analyze_stream/read/write/generate/encode/decode (phase 1)
  utils/
    __init__.py
    json_io.py                         # _json_default + dump_json (phase 2)
    io_paths.py                        # get_output_dir, label/path helpers (phase 2)
```

## Phases

### Phase 1: Split high-churn commands out of main.py
- Create:
  - `cmd_corpus.py`: move `analyze_corpus`.
  - `cmd_diff.py`: move `compare_reads`.
  - `cmd_stream_ops.py`: move `analyze_stream`, `read_track`, `write`, `generate`, `encode`, `decode`.
- Update `main.py` to import these and set `set_defaults(func=...)` accordingly.
- Keep behavior unchanged; continue using existing helpers.

### Phase 2: Extract shared utilities
- Create `utils/json_io.py` with `_json_default(obj)` and `dump_json(path, obj, **kwargs)`.
- Create `utils/io_paths.py` with `get_output_dir(output_dir)` and label/path helpers.
- Update modules to import from `utils` instead of duplicating helpers.

### Phase 3: Isolate analyze_disk pipeline
- Create `analysis/analyze_disk.py` and move the full pipeline:
  - Input discovery, flux aggregation, single‑sided detection.
  - Classification with media override (source + confidence).
  - Overlay detection (calls `overlay_detection.py`).
  - Rendering (calls `rendering.py`).
  - JSON/CSV/PNG outputs (via `utils/json_io.py`).
- `main.py` delegates to `analysis.analyze_disk.run(args)`.

### Phase 4: Optional enhancements
- Per‑track overlay rendering option to visualize GCR zoning.
- Add unit tests for `utils/json_io.py`, `overlay_detection.py` (basic cases).
- Developer docs: `docs/dev/architecture.md` describing module boundaries.

## Current Status (2025‑09‑19)
- Detection and rendering already split (overlay_detection.py, rendering.py).
- JSON dump hardened; remaining step: move serializer to `utils/json_io.py`.
- Classification honors `--media-type`; single‑sided thresholds tightened.
- Surface map written as a stub at start; corpus tracks missing maps.

## Acceptance Checklist
- `main.py` under 800 LOC; commands live in `cmd_*.py` and `analysis/`.
- CLI behavior and outputs unchanged.
- `surface_map.json` and `overlay_debug.json` always produced.
- Corpus writes `corpus_inputs.txt` and `corpus_missing_inputs.txt`.
- Classification includes `source` and `confidence` in JSON and text.

## Rollout Plan
- Land Phase 1 + 2 in small patches, verify CLI.
- Land Phase 3, verify `analyze_disk` on sample inputs.
- Update `docs/roadmap.md` to reference this plan.
