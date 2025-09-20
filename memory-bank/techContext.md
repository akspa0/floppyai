# Tech Context — FloppyAI

## Stack
- Python: 3.9+ recommended
- OS: Windows (KryoFlux tooling, PowerShell examples)
- Optional LLM: OpenAI-compatible server (e.g., LM Studio)

## Key Dependencies
- numpy, matplotlib
- (optional) openai — only needed when `--summarize` is used
- Standard library: argparse, json, re, pathlib, datetime, subprocess

Install:
```bash
pip install -r requirements.txt
```

## Repository Layout (relevant)
- `FloppyAI/src/main.py` — CLI entrypoint (module-invoked)
- `FloppyAI/src/cmd_stream_ops.py` — analyze single stream, read, write, generate, encode
- `FloppyAI/src/cmd_corpus.py` — corpus aggregation and optional generation of missing maps
- `FloppyAI/src/cmd_diff.py` — compare reads (diff summary, densities CSV)
- `FloppyAI/src/analysis/analyze_disk.py` — shim now; Phase 3 will move the pipeline here
- `FloppyAI/src/analysis/metrics.py` — (planned) metrics for experiments
- `FloppyAI/src/utils/json_io.py` — `dump_json()` with a custom encoder
- `FloppyAI/src/utils/io_paths.py` — `get_output_dir()` and labeling helpers

## Invocation Pattern
Run from the repository root using module syntax to ensure imports and subprocess flows resolve:
```bash
python -m FloppyAI.src.main --help
```

Notes:
- Some flows (e.g., `analyze_corpus --generate-missing`) spawn `python -m FloppyAI.src.main` via subprocess; CWD must be the repo root.
- `--output-dir` flags are supported by most commands; otherwise a timestamped `test_outputs/<timestamp>/` folder is used.

## Hardware
- KryoFlux board; DTC.exe available (see README notes under `lib/kryoflux_3.50_windows_r2/dtc/`).
- Safety defaults for experiments: sacrificial media, outer tracks, limited revs, cooldown, simulate-first.

## Optional LLM
- Provide `--lm-host`, `--lm-model`, `--lm-temperature` and `--summarize` to enable summaries.
- Do not hardcode secrets; rely on environment or local endpoints.

## Constraints & Considerations
- Windows paths and quoting in examples.
- Keep JSON serialization centralized via `utils/json_io.dump_json` to avoid dtype/Path issues.
- Prefer internal function calls for orchestration; fall back to subprocess where isolation or packaging makes it safer.
