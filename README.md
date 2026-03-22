# HelixNet

HelixNet is a distributed molecular simulation system for orchestrating WESTPA and OpenMM workloads across GPU clusters.

The system coordinates large-scale simulation jobs on NERSC A100 nodes, enabling concurrent execution, parameter sweeps, and ensemble-based sampling under HPC constraints.

## System Overview

HelixNet operates as a distributed, GPU-backed simulation pipeline:

- Configuration — defines simulation parameters, input structures, and sampling strategy
- Orchestration — schedules and distributes jobs across GPU nodes via Slurm
- Execution — runs WESTPA/OpenMM simulations concurrently across workers
- Aggregation — collects outputs from distributed simulations
- Analysis — processes results for downstream evaluation

The system is designed to scale across multi-node GPU environments while managing scheduling constraints, resource contention, and simulation consistency.

## Architecture

```text
Input Config
     ↓
Slurm Scheduler
     ↓
Distributed GPU Workers (NERSC A100)
     ↓
WESTPA / OpenMM Execution
     ↓
Output Aggregation
     ↓
Analysis Pipeline
```

## Key Properties

- Distributed execution across GPU clusters (NERSC A100)
- Slurm-based job orchestration
- Concurrent simulation pipelines (50+ runs)
- Support for parameter sweeps and ensemble sampling
- Designed for high-throughput molecular simulation workloads

## System Constraints

- HPC scheduling latency (Slurm queue delays and resource availability)
- GPU resource contention across concurrent simulation jobs
- Distributed synchronization across simulation workers
- Large-scale output management and aggregation
- Environment consistency across nodes (micromamba / WESTPA setups)

## Why This Matters

Molecular simulations require significant compute and careful orchestration across distributed GPU resources. HelixNet explores how to structure and scale simulation pipelines across HPC clusters, addressing challenges in scheduling, concurrency, and reproducibility under real-world constraints.

## Entrypoints

- Runtime: `run.sh`
- Testing: `test.sh`
- Python app: `benchmark.py`

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

## Run Commands

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

## Test Commands

```bash
# Local mocked shell flow
./test.sh mock

# NERSC end-to-end flow
./test.sh e2e [PDB_ID]

# Python test suite (single file)
python -m pytest tests/test.py -v
```

## Pipeline Commands

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
