# FloppyAI CLI Usage Guide

This guide shows how to run FloppyAI from the repository root using Python's module invocation and includes practical examples for all commands.

## Invocation

- Always run from the repository root (the folder that contains `FloppyAI/`).
- Use Python's module syntax so the package and subcommands resolve correctly:

```bash
python -m FloppyAI.src.main --help
```

## Environment

- Python 3.9+ recommended
- `pip install -r requirements.txt`
- Optional: LM Studio (or compatible OpenAI API server) for `--summarize`
- Optional: KryoFlux hardware + DTC for read/write flows

## Global Tips

- Use `--output-dir` to direct outputs to a stable folder; otherwise a timestamped directory under `test_outputs/` is created.
- For PC disks, `--media-type 35HD` (3.5" HD) or `--media-type 525HD` (5.25" HD) is a good starting point.
- For classic Mac disks (400K/800K, Apple GCR), use `--media-type 35DD` and `--overlay-mode gcr`.

---

## Commands

### 1) analyze — single .raw file

Parse a single KryoFlux `.raw` and generate basic statistics and plots.

```bash
python -m FloppyAI.src.main analyze path\to\file.raw --output-dir .\test_outputs\single
```

Outputs: `<stem>_intervals.png`, `<stem>_hist.png`, and `<stem>_heatmap.png` (if multi‑rev).

### 2) read — capture a track/side

```bash
python -m FloppyAI.src.main read <track> <side> [--revs 3] [--simulate] [--analyze] [--output-dir DIR]
```

- `--simulate` avoids real hardware access
- `--analyze` automatically runs `analyze` on the captured `.raw`

### 3) write — write a `.raw` to hardware

```bash
python -m FloppyAI.src.main write <input.raw> <track> <side> [--simulate] [--output-dir DIR]
```

### 4) generate — synthetic stream for testing

```bash
python -m FloppyAI.src.main generate <track> <side> [--revs 1] [--cell 4000] [--analyze] [--output-dir DIR]
```

- Increase `--revs` to fill more revolutions
- Reduce `--cell` (ns) to raise density

### 5) encode — binary → .raw

```bash
python -m FloppyAI.src.main encode <input.bin> <track> <side> \
  [--density 1.0] [--variable] [--revs 1] [--output OUT.raw] [--write] [--simulate] [--analyze] [--output-dir DIR]
```

### 6) decode — recover binary data from `.raw`

```bash
python -m FloppyAI.src.main decode <input.raw> \
  [--density 1.0] [--variable] [--revs 1] [--output out.bin] [--expected orig.bin] [--output-dir DIR]
```

---

## Disk‑level analysis

### 7) analyze_disk — build a full surface map

```bash
python -m FloppyAI.src.main analyze_disk <dir_or_file> \
  [--rpm FLOAT] [--profile 35HD|35DD|525HD|525DD] [--media-type 35HD|35DD|525HD|525DD] \
  [--format-overlay] [--overlay-mode mfm|gcr|auto] [--angular-bins N] \
  [--gcr-candidates CSV] [--overlay-sectors-hint INT] \
  [--overlay-alpha F] [--overlay-color HEX] \
  [--summarize] [--lm-host HOST[:PORT]] [--lm-model NAME] [--lm-temperature F] \
  [--output-dir DIR]
```

- Media override: `--media-type` forces classification, influencing RPM and hints
- RPM: provide `--rpm`, or let `--profile` set one (35HD/35DD=300, 525HD=360, 525DD=300)
- MFM overlays (PC): `--format-overlay --overlay-mode mfm --angular-bins 720`
- GCR overlays (Mac): `--format-overlay --overlay-mode gcr --gcr-candidates "12,10,8,9,11,13" --angular-bins 900`

Outputs include `surface_map.json`, `overlay_debug.json`, composite and surface PNGs, and an instability summary CSV.

### 8) analyze_corpus — aggregate many maps

```bash
python -m FloppyAI.src.main analyze_corpus <root_or_map.json> \
  [--generate-missing] [--rpm FLOAT] [--profile 35HD|35DD|525HD|525DD] [--media-type 35HD|35DD|525HD|525DD] \
  [--format-overlay] [--overlay-mode mfm|gcr|auto] [--angular-bins N] [--gcr-candidates CSV] [--overlay-sectors-hint INT] \
  [--overlay-alpha F] [--overlay-color HEX] \
  [--summarize] [--lm-host HOST[:PORT]] [--lm-model NAME] [--lm-temperature F] \
  [--output-dir DIR]
```

- With `--generate-missing`, the tool will first run `analyze_disk` on directories that contain `.raw` files but no `surface_map.json` and then aggregate
- Writes `corpus_inputs.txt` and `corpus_missing_inputs.txt` for transparency

### 9) compare_reads — compare multiple reads of the same disk

```bash
python -m FloppyAI.src.main compare_reads <path_or_dir1> <path_or_dir2> [<...>] [--output-dir DIR]
```

- Each argument may be a `surface_map.json` path or a directory containing it
- Outputs `diff/diff_summary.json` and `diff/diff_densities.csv`

### 10) classify_surface — quick blank-like vs written-like

```bash
python -m FloppyAI.src.main classify_surface <surface_map.json> [--blank-density-thresh 1000] [--output-dir DIR]
```

### 11) plan_pool — select top‑quality tracks as a bit pool

```bash
python -m FloppyAI.src.main plan_pool <surface_map.json> [--min-density 2000] [--top-percent 0.2] [--output-dir DIR]
```

---

## Overlay Notes

- The overlay block is saved in `surface_map.json` under `global.insights.overlay` and per‑track under `<track>.overlay`.
- If confidence is weak, the tool falls back to either profile defaults, your `--overlay-sectors-hint`, or the median per‑track sector count.
- Increase `--angular-bins` for finer phase resolution; tune `--overlay-color` and `--overlay-alpha` for visibility.

## Troubleshooting

- Compare reads says it found no `surface_map.json`:
  - Run `analyze_disk` for each read first (pointing to the read directory), then re-run `compare_reads` using those run output folders or the direct JSON paths.

- Module cannot be found:
  - Ensure you run commands from the repository root with `python -m FloppyAI.src.main ...`.
