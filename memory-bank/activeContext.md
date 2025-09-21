# Active Context — FloppyAI

Last updated: 2025-09-21 01:01 (local)

## Current Focus
- Visualization and overlay fidelity on full-disk runs.
- Reduce CLI friction with profile‑driven defaults (MFM vs GCR) while keeping forensic‑rich outputs concise.

## Recent Changes
- Visualization fixes and enhancements:
  - `render_instability_map()` now accepts the surface map and renders angular‑resolved instability when available; colormap switched to `magma_r` with percentile contrast for bright hotspots.
  - Side report and dashboard instability panels use the same angular‑resolved profile and contrast.
  - Wedge spokes, sector labels, and intra‑wedge peak markers integrated across surface, side reports, and dashboard.
- Overlay simplification:
  - Added GCR profiles `35DDGCR`, `35HDGCR`, `525DDGCR`.
  - `--overlay-mode` defaults to `auto`; in `analysis/analyze_disk.py` it is profile‑driven (GCR profiles → GCR; others → MFM). GCR candidates auto‑selected per profile.
  - Gentle log notes when overlay mode conflicts with the chosen profile.
- Docs updated (`README.md`, `docs/usage.md`) to reflect simplified commands and brighter instability visuals.

## Next Steps
- Validate sector overlay results on representative MFM and GCR datasets; fine‑tune candidate lists per zone if needed.
- Consider optional `--overlay-fill` (subtle wedge shading) to further emphasize sectors.
- Extend experiments docs with guidance on forensic‑rich captures and multi‑read comparisons; keep per‑side composites to avoid PNG spam.
- Continue Phase 2 cleanup (ensure all JSON writes use `utils.json_io.dump_json`).

## Open Decisions
- Whether to expose a user knob for instability contrast percentile; currently hard‑coded to a sensible default.
- Optional addition of per‑track overlay JSON for zoned GCR in future iterations.

## Links
- README: `../README.md`
- Docs hub: `../docs/index.md`
- Experiments hub: `../docs/experiments/index.md`
