# Linux DTC Scripts (FloppyAI)

This folder contains helper scripts to run KryoFlux DTC operations on a Linux host that is physically connected to the board. Use these when FloppyAI is run on Windows for generation/analysis, while hardware I/O runs on Linux.

## Requirements
- Linux machine with KryoFlux installed and `dtc` available in `PATH` (or pass `--dtc-path`)
- Sudo access to run `dtc` (unless your environment is configured otherwise)
- KryoFlux board connected and drive powered

> Important: DTC argument ordering matters. All scripts place `-f<prefix>` FIRST, then `-i0` (for read) or `-wi4` (for write-from-stream), followed by the rest of the flags. Example read: `dtc -ftrack -i0 -d0 -s0 -e82 -g0 -r3`.

## Script: dtc_write_read.sh
Write a `.raw` test pattern to a specified track/side and then read back N revolutions, saving the capture and a log.

Example:
```bash
chmod +x FloppyAI/scripts/linux/dtc_write_read.sh
FloppyAI/scripts/linux/dtc_write_read.sh \
  --write /path/to/generated.raw --track 80 --side 0 --revs 3 \
  --out-dir ./captures --drive 0
```

Options:
- `--write <file>`: Input `.raw` to write (required)
- `--track <N>`: Track index (required)
- `--side <0|1>`: Side index (required)
- `--revs <N>`: Revolutions to read back (default: 3)
- `--drive <N>`: DTC drive index (default: 0)
- `--dtc-path <path>`: Path to `dtc` binary (default: `dtc` in PATH)
- `--out-dir <dir>`: Output directory for captured `.raw` and logs (default: `FloppyAI/output_captures`)
- `--label <name>`: Label prefix for output filenames (optional)
- `--no-sudo`: Do not prefix commands with `sudo` (default: sudo on)
- `--dry-run`: Print commands without executing

Outputs:
- A timestamped `capture_*.raw` in `--out-dir`
- A sibling `capture_*.log` containing `dtc --version` and the exact commands that were run

## Notes
- Use sacrificial media, prefer outer tracks, and allow cool-downs between writes.
- All read commands use `-f<prefix> -i0 ...`; all write-from-stream commands use `-f<basename> -wi4 ...`.
- For cross‑machine workflow details, see `FloppyAI/docs/usage.md` → “Cross‑machine (Linux DTC) workflow”.

---

## Script: capture_forensic_full_disk.sh
Forensic STREAM capture across a full disk.

- Defaults: 83 tracks (0..82), `--revs 3`, outputs under `FloppyAI/output_captures/<label>_<ts>/`
- Flags:
  - `--profile 35HD|35DD|525HD|525DD`
  - `--sides both|0|1`, `--start-track`, `--end-track`, `--step`
  - `--revs N`, `--cooldown`, `--spinup`
  - `--rich`, `--no-p`, `--no-ml` (append extra dtc flags conservatively)
  - `--passes N` writes each whole-disk pass to its own subfolder: `pass_01_side_0/`, `pass_01_side_1/`, ...
  - `--sanity` runs `tools/stream_sanity.py` on outputs
  - `--corpus` runs `tools/compare_corpus.py` to compare passes

Example:
```bash
FloppyAI/scripts/linux/capture_forensic_full_disk.sh \
  --profile 35HD --sides both --revs 3 --passes 2 --sanity --corpus
```

## Script: capture_forensic_repeat_track.sh
Repeat-capture a single track/side several times.

- Each repeat writes into its own subfolder: `read_01/`, `read_02/`, ...
- Supports `--sanity` to run a quick stream check.

Example:
```bash
FloppyAI/scripts/linux/capture_forensic_repeat_track.sh --track 0 --side 0 --repeats 5 --revs 3 --sanity
```

## Script: capture_forensic_sweep.sh
Sweep forward and backward across tracks for multiple sweeps.

- Each sweep writes into direction-specific subfolders: `sweep_01_fwd/`, `sweep_01_bwd/`, ...
- Defaults to 83 tracks (0..82).

## Script: capture_forensic_long_rev.sh
Single track/side with many revolutions (e.g., 16–64) for high-fidelity stats.

---

## Script: dtc_minimal_test.sh
Minimal dtc read test (bare flags) to validate that `dtc` is writing files.

```bash
FloppyAI/scripts/linux/dtc_minimal_test.sh --dir FloppyAI/output_captures/test_min
```

## Script: dtc_write_read_set.sh
Write a whole set of NN.S.raw with `-wi4` and (optionally) read back captures.

```bash
FloppyAI/scripts/linux/dtc_write_read_set.sh \
  --image-dir ./pattern_set --drive 0 --revs 3 --out-dir FloppyAI/output_captures/my_run --read-back
```

## Script: dtc_probe.sh
Probe drive capacity and RPM.

```bash
# Capacity (track reach)
FloppyAI/scripts/linux/dtc_probe.sh --drive 0 --capacity

# RPM (averaged over N seconds; clean exit via SIGINT)
FloppyAI/scripts/linux/dtc_probe.sh --drive 0 --rpm --seconds 6
```

## Script: experiment_write_read_analyze.sh
Turnkey: generate NN.S.raw patterns → write with `-wi4` → read with `-i0` → analyze.

```bash
FloppyAI/scripts/linux/experiment_write_read_analyze.sh \
  --tracks 0-82 --sides 0,1 --revs 3 --drive 0 \
  --label pat_const4us --pattern constant --interval-ns 4000 --rev-time-ns 200000000 --sanity
```
