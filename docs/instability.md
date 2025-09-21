# Instability — Definition and Interpretation

This page explains exactly what the “instability” visualizations represent in FloppyAI, how they’re computed from flux data, and how to read them without mistaking them for recorded waveforms or invented content.

## What instability is

- Instability measures repeatability of the transition pattern as the disk spins. We look at multiple revolutions and ask: at a given angular position, do we see a similar number of flux transitions every time, or does it vary?
- The result is expressed as:
  - An angular‑resolved profile per track, `instability_theta` (0..1), indicating which angles around the disk vary more/less across revolutions.
  - A scalar per track, `instability_score` (0..1), combining several factors into a single track‑level score.

## What instability is not

- It is not an audio‑like waveform recorded along the track.
- It is not a direct measurement of sectors or data content. It can correlate with formats because format structures influence transition statistics, but the metric is about repeatability, not decoding.
- It does not fabricate data: values are derived statistically from observed flux transitions across revolutions.

## How it’s computed (overview)

Computation lives in `FloppyAI/src/flux_analyzer.py::FluxAnalyzer.analyze()`.

1. For each revolution, build an angular histogram of transitions with `angular_bins` bins (counts per angle).
2. Align revolutions in phase by circular cross‑correlation so similar wedge patterns line up.
3. Compute per‑angle variance across aligned revolutions:
   - Normalize variance to 0..1 → `per_angle_variance`.
   - Set `instability_theta = per_angle_variance`.
4. Compute a scalar `instability_score` by combining:
   - High‑tail of per‑angle variance (e.g., 95th percentile): captures strong angular hot spots.
   - Cross‑revolution incoherence (1 − mean correlation to the mean profile): low agreement means higher instability.
   - Outlier rate: proportion of unusually short/long intervals.
   - Gap rate: proportion of very long intervals (suspected gaps/dropouts).

In the visuals (`FloppyAI/src/rendering.py`):
- Disk‑wide map: `Z(r, θ) = radial_instability(track) × angular_instability_profile(θ)`, combining track‑level instability with side‑level angular weighting.
- Side report / Dashboard panels: use the same angular‑resolved instability logic.
- Colormap is inverted magma (`magma_r`) with percentile contrast stretch for clarity; this does not invent features, it remaps intensities for visibility.

## Why wedge‑like patterns appear

- Sector boundaries and gaps alter local transition statistics. Even on healthy media, these regions often have different repeatability than data fields, so angular bands can appear that align with sector spokes.
- Mechanical factors (RPM wobble, write splice, head/media condition) can also produce specific angular hot spots in instability.

## How to verify on your outputs

- Open `surface_map.json` for the run:
  - Per track/side → `analysis.angular_bins`, `analysis.angular_hist`, `analysis.instability_theta`, `analysis.instability_features`, `analysis.instability_score`.
- Compare to the side report/dashboard:
  - Brighter areas in the instability panels should correspond to angles where `instability_theta` is higher.
- Try a blank vs formatted disk:
  - Blank media typically shows flatter instability; formatted media often exhibits angular structure due to format fields and gaps.

## Reading the panels

- Bright = less consistent across revolutions at that radius/angle.
- Dark = more consistent/repeatable behavior.
- On disks with overlays enabled, spokes mark suspected sector boundaries; they are reference guides only.

## Overlays are non‑synthetic guides only

- Overlays draw angular spokes and labels to mark detected sector boundaries.
- No synthetic fill is applied, and overlays never change the underlying map data.

## FAQ

- “It looks like a waveform along the track — is that real data?”
  - No. Instability is a statistical map of repeatability (variance/incoherence/outliers/gaps), not a time‑domain recording.
- “Why do certain wedges glow even if the disk is fine?”
  - Format structures and small mechanical effects can cause angular repeatability differences. That’s expected and can still be useful for diagnostics.
- “How can I quantify a hot spot?”
  - Inspect `instability_theta` for the track/side in `surface_map.json`, or export auxiliary summaries if needed.
