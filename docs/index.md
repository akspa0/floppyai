# FloppyAI Documentation Index

Welcome to FloppyAI. Start here to navigate the docs.

- Getting started and CLI overview
  - README (project overview): ../README.md
  - Usage Guide (all commands and flags): ./usage.md
  - Cheatsheet (common tasks): ./cheatsheet.md

- Project direction and refactor plan
  - Refactor Plan: ./refactor_plan.md
  - Roadmap: ./roadmap.md

- Experiments
  - Experiments Hub: ./experiments/index.md

## Recommended Invocation

Run all commands from the repository root using Python's module syntax:

```bash
python -m FloppyAI.src.main --help
```

## Typical Workflows

- Build a full disk surface map (MFM):
  ```bash
  python -m FloppyAI.src.main analyze_disk path\to\disk_dir \
    --media-type 35HD --format-overlay --overlay-mode mfm --angular-bins 720 \
    --output-dir .\test_outputs\disk_run
  ```

- Build a Mac GCR map (zoned) and compare two reads:
  ```bash
  python -m FloppyAI.src.main analyze_disk path\to\mac_disk \
    --media-type 35DD --format-overlay --overlay-mode gcr \
    --gcr-candidates "12,10,8,9,11,13" --angular-bins 900 \
    --output-dir .\test_outputs\mac_A

  python -m FloppyAI.src.main analyze_disk path\to\mac_disk_b \
    --media-type 35DD --format-overlay --overlay-mode gcr \
    --gcr-candidates "12,10,8,9,11,13" --angular-bins 900 \
    --output-dir .\test_outputs\mac_B

  python -m FloppyAI.src.main compare_reads .\test_outputs\mac_A .\test_outputs\mac_B \
    --output-dir .\test_outputs\mac_diff
  ```

## Outputs at a Glance

- Per run: `surface_map.json`, `overlay_debug.json`, composite and polar PNGs, side PNGs, and instability CSV
- Corpus: `corpus_summary.json`, histograms/boxplots/scatter, inputs/missing manifests, montage grids
- Diff: `diff_summary.json` and `diff_densities.csv`
