# Experiments

This section hosts structured experiment plans and results for FloppyAI. Each experiment describes:

- Objectives and hypotheses
- Safety and setup
- Protocol and CLI commands
- Metrics and outputs
- How to interpret results

## Index

1. [Extreme Streams](./01-extreme-streams.md) — push recording limits using controlled flux patterns and densities

## Conventions

- Name experiments using a leading number for ordering: `NN-short-name.md`
- Keep experiments self-contained with reproducible CLI examples
- Prefer running from the `FloppyAI/` directory using the main script (module mode also works):
  ```bash
  python src/main.py --help
  ```
- Use sacrificial media and follow safety guidance for any hardware runs
