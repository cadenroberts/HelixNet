# HelixNet

Simple WESTPA/OpenMM pipeline runner for DNA-protein systems on NERSC, with a Streamlit UI.

## What this repo gives you

- One runtime entrypoint: `run.sh`
- One test entrypoint: `test.sh`
- One Python entrypoint: `benchmark.py`

## Requirements

- Python 3.10+
- Bash
- NERSC access (for real runs)
- Slurm tools on runtime host (`squeue`, `sbatch`)
- WESTPA/OpenMM stack in configured micromamba envs

## Quick start

```bash
git clone <repo-url> HelixNet
cd HelixNet
cp config.example.json config.json
```

Edit `config.json` and set at minimum:

- `execution.nersc_user`
- `paths.project_dir`
- `paths.out_dir`
- `paths.micromamba_prefix`
- `paths.westpa_env_prefix`

## Run commands

### UI

```bash
./run.sh ui
```

### Headless

```bash
# Set up one target
./run.sh setup 1ABC

# Set up all IDs from pdb_ids.json then monitor
./run.sh batch

# Monitor and resubmit only
./run.sh run
```

### Smoke check

```bash
./run.sh demo
```

## Test commands

```bash
# Local mocked shell flow
./test.sh mock

# NERSC end-to-end flow
./test.sh e2e [PDB_ID]

# Python test suite (single file)
python -m pytest tests/test.py -v
```

## How it works

- `benchmark.py read-config` reads values from `config.json`.
- `run.sh setup`:
  - runs `benchmark.py preprocess <PDB_ID>`
  - expands templates in `westpa_template/`
  - runs `w_init`
- `run.sh run`:
  - checks each `*_WP/west.h5` iteration count
  - submits `sbatch run.slurm` when target iterations not reached
- `run.sh batch`:
  - reads `pdb_ids.json`
  - runs setup for missing targets
  - runs monitor step

## Minimal file map

```text
benchmark.py
run.sh
test.sh
config.example.json
westpa_template/
tests/
  test.py
```

## Troubleshooting

- `run.sh ui` fails: remove `.venv` and retry.
- `run.sh setup` fails: validate config paths and env activation.
- `run.sh run` submits nothing: verify `west.h5`, `run.slurm`, and target iterations.
- `test.sh e2e` fails: refresh SSH credentials and verify remote path.

## License

MIT
