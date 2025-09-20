# Linux DTC Scripts (FloppyAI)

This folder contains helper scripts to run KryoFlux DTC operations on a Linux host that is physically connected to the board. Use these when FloppyAI is run on Windows for generation/analysis, while hardware I/O runs on Linux.

## Requirements
- Linux machine with KryoFlux installed and `dtc` available in `PATH` (or pass `--dtc-path`)
- Sudo access to run `dtc` (unless your environment is configured otherwise)
- KryoFlux board connected and drive powered

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
- `--out-dir <dir>`: Output directory for captured `.raw` and logs (default: `./captures`)
- `--label <name>`: Label prefix for output filenames (optional)
- `--no-sudo`: Do not prefix commands with `sudo` (default: sudo on)
- `--dry-run`: Print commands without executing

Outputs:
- A timestamped `capture_*.raw` in `--out-dir`
- A sibling `capture_*.log` containing `dtc --version` and the exact commands that were run

## Notes
- Use sacrificial media, prefer outer tracks, and allow cool-downs between writes.
- If your `dtc` version expects different flags, adjust the script (it currently uses `-i 21` for write-from-stream and `-i 0` for hardware read).
- For cross‑machine workflow details, see `FloppyAI/docs/usage.md` → “Cross‑machine (Linux DTC) workflow”.
