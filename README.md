# HelixNet

WESTPA/OpenMM weighted ensemble molecular dynamics for DNA-protein complexes on NERSC Perlmutter. Includes a Streamlit UI for RCSB PDB search, configuration, pipeline control, and job monitoring.

---

## Prerequisites

### Local machine (Mac/Linux)

- Python 3.10+
- Git
- SSH access to NERSC Perlmutter (`~/.ssh` key or `sshproxy`)

### NERSC Perlmutter

The following must be available on NERSC (via micromamba or module):

| Package | Purpose |
|---------|---------|
| openmm | MD engine |
| pdbfixer | PDB repair |
| rdkit | Ligand chemistry |
| openff-toolkit | Force field toolkit |
| openmmforcefields | GAFF parameterization |
| mdtraj | Trajectory I/O |
| numpy | Numerics |
| westpa | Weighted ensemble |

These are expected in two conda/micromamba environments on NERSC:
- **openmm env** (`paths.micromamba_prefix`): openmm, pdbfixer, rdkit, openff-toolkit, openmmforcefields, numpy, requests
- **westpa env** (`paths.westpa_env_prefix`): westpa, mdtraj, numpy, openmm

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url> HelixNet
cd HelixNet
```

### 2. Create config.json

```bash
cp config.example.json config.json
```

Edit `config.json` with your values. The only required change for a first run is:

```json
{
  "execution": {
    "nersc_user": "your_nersc_username"
  },
  "paths": {
    "project_dir": "/global/cfs/cdirs/<allocation>/<you>/westpa_dna_protein",
    "out_dir": "out",
    "micromamba_prefix": "/path/to/your/micromamba/envs/openmm",
    "westpa_env_prefix": "/path/to/your/micromamba/envs/westpa_env"
  }
}
```

### 3. Launch the UI

```bash
./run_ui.sh
```

This creates a `.venv`, installs `streamlit`, `paramiko`, and `requests`, then opens the UI in your browser.

### 4. Or run headless

```bash
# Single target
./setup_wp.sh 1ABC

# Batch from pdb_ids.json
./batch_wp.sh

# Monitor and resubmit
./run_wp.sh
```

---

## Configuration reference

All parameters live in `config.json`. The UI edits this file directly. Shell scripts read it via `read_config.py`.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `execution` | `nersc_user` | `""` | Your NERSC username (required for SSH mode) |
| `paths` | `project_dir` | | Absolute path to the project on NERSC CFS |
| `paths` | `out_dir` | `"out"` | Subdirectory for `*_WP` output (relative to `project_dir`, or absolute) |
| `paths` | `micromamba_prefix` | | Path to the openmm conda/micromamba environment |
| `paths` | `westpa_env_prefix` | | Path to the WESTPA conda/micromamba environment |
| `rcsb_search` | `keywords` | `["DNA BINDING PROTEIN, DNA", ...]` | RCSB keyword search terms |
| `rcsb_search` | `organism` | `"Homo sapiens"` | Organism filter |
| `rcsb_search` | `max_resolution` | `2.5` | Maximum resolution in angstroms |
| `slurm` | `account` | `"m4229"` | NERSC allocation account |
| `slurm` | `qos` | `"regular"` | Slurm QoS (`regular`, `debug`, `premium`) |
| `slurm` | `walltime` | `"48:00:00"` | Per-job walltime |
| `slurm` | `nodes` | `1` | Nodes per job |
| `slurm` | `ntasks_per_node` | `4` | MPI tasks per node |
| `slurm` | `gpus_per_task` | `1` | GPUs per MPI task |
| `westpa` | `target_iterations` | `12500` | Total WE iterations to reach |
| `westpa` | `pcoord_len` | `11` | Progress coordinate length per segment |
| `westpa` | `nbins` | `9` | Number of MAB bins |
| `openmm` | `temperature` | `300.0` | Simulation temperature (K) |
| `openmm` | `timestep` | `4.0` | Integration timestep (fs) |
| `openmm` | `steps` | `1000` | MD steps per WE segment |
| `openmm` | `forcefield` | `["amber14-all.xml", "amber14/tip3pfb.xml"]` | OpenMM force field XML files |
| `preprocessing` | `padding_nm` | `1.0` | Solvent box padding (nm) |
| `preprocessing` | `ionic_strength_M` | `0.15` | Ionic strength (molar) |
| `preprocessing` | `ph` | `7.0` | pH for hydrogen placement |

---

## Execution modes

Mode is auto-detected from hostname:

| Hostname contains | Mode | How scripts run |
|-------------------|------|-----------------|
| `nersc` or `perlmutter` | Local | `subprocess.Popen(["bash", script])` |
| Anything else | SSH | `paramiko.SSHClient` to `perlmutter.nersc.gov` |

**SSH mode** (running from your laptop):
```bash
./run_ui.sh
# UI opens in browser, all pipeline commands execute over SSH
```

**Local mode** (running on a NERSC login node):
```bash
streamlit run app.py --server.port 8501
# On your laptop, forward the port:
ssh -L 8501:localhost:8501 <user>@perlmutter.nersc.gov
```

---

## Pipeline

### Step 1: Search RCSB for PDB IDs

Use the **RCSB Search** tab in the UI. The search uses the [RCSB Search API v2](https://search.rcsb.org):

- `/v2/query` (POST and GET) -- keyword, organism, resolution filters with auto-pagination
- `/v2/suggest` -- autocomplete suggestions
- `/v2/query/unreleased` -- search upcoming entries
- `/v2/metadata/*/schema` -- browse searchable attributes

Results are saved to `pdb_ids.json`.

### Step 2: Preprocess

For each PDB ID, `preprocess_pdb.py`:

1. Downloads the PDB from RCSB
2. Repairs with PDBFixer (missing residues, atoms, hydrogens)
3. Identifies non-standard ligands, fetches SMILES from RCSB GraphQL, assigns bond orders with RDKit
4. Parameterizes ligands with GAFF via `openmmforcefields`
5. Solvates with explicit water (TIP3P) and ions
6. Writes `{ID}_WP/processed/{ID}_processed.pdb` and `forcefield.json`

### Step 3: WESTPA setup

`setup_wp.sh` expands templates from `westpa_template/` using config values:

| Template | Output | Placeholders |
|----------|--------|--------------|
| `run.slurm.template` | `run.slurm` | `PDB_ID`, `ACCOUNT`, `QOS`, `WALLTIME`, `NODES`, `NTASKS`, `CPUS`, `GPUS` |
| `west.cfg.template` | `west.cfg` | `PDB_ID`, `PROJECT_DIR`, `TARGET_ITERATIONS`, all OpenMM params |
| `b.txt.template` | `b.txt` | `PDB_ID` |
| `env.sh` | `env.sh` | `REPO_DIR` |

Then runs `w_init --bstate-file b.txt` to initialize the WESTPA HDF5 file.

### Step 4: Run simulations

`batch_wp.sh` processes all IDs in `pdb_ids.json`, then calls `run_wp.sh` which:

1. Reads `west.h5` iteration count via `h5ls`
2. Compares against `westpa.target_iterations`
3. Submits `sbatch run.slurm` for any target below the threshold
4. Displays a dashboard with done/running/error status

### Step 5: Monitor

Use the **Status** tab in the UI, or run `./run_wp.sh` directly.

---

## Per-target directory layout

After setup completes for PDB ID `1ABC`:

```
out/1ABC_WP/
  raw/1ABC.pdb                              Downloaded from RCSB
  processed/1ABC_processed.pdb              Solvated, parameterized
  processed/forcefield.json                 Force field files used
  west.cfg                                  WESTPA configuration
  run.slurm                                 Slurm job script
  b.txt                                     Basis states
  env.sh                                    Environment activation
  openmm_explicit_rmsd_p_ca_propagator.py   Propagator
  west.h5                                   WESTPA iteration data
  traj_segs/                                Per-iteration trajectories
    000001/000000/
      seg.dcd                               Solute-only trajectory
      seg.xml                               OpenMM checkpoint
      seg.npz                               Forces, energies, times
```

---

## Propagator

`OpenMMExplicitPropagator` in `westpa_template/openmm_explicit_rmsd_p_ca_propagator.py`:

- **Electrostatics**: PME (Particle Mesh Ewald)
- **Thermostat**: Langevin Middle Integrator (300 K, 1/ps friction)
- **Barostat**: Monte Carlo (1 atm, interval 25)
- **Constraints**: HBonds, rigid water
- **H-mass repartitioning**: 1.5 amu (enables 4 fs timestep)
- **Progress coordinate**: RMSD of phosphorus (P) and alpha-carbon (CA) atoms vs reference
- **Output**: Solute-only DCD (water/ions stripped), XML checkpoint, NPZ (forces/energies)
- **GPU**: CUDA with configurable precision; falls back to CPU if unavailable

---

## Testing

### Run all tests

```bash
pip install pytest pytest-mock responses
python -m pytest tests/ -v
```

### Test breakdown

| File | What it tests |
|------|---------------|
| `tests/test_config.py` | Config load/save, `read_config.py` CLI, PDB ID persistence |
| `tests/test_app_helpers.py` | Execution mode detection, ANSI stripping, payload builder, credential gate |
| `tests/test_rcsb_api.py` | Live RCSB API calls (search, suggest, unreleased, metadata) + mocked error paths |
| `tests/test_preprocess.py` | PDB ID validation, folder creation, ligand SMILES lookup, full preprocessing (requires openmm) |
| `tests/test_propagator.py` | RMSD progress coordinate, solute index extraction, DCD I/O (requires mdtraj) |
| `tests/test_nersc_launch.py` | SSH client (mocked paramiko), `run_script` dispatch, `scan_wp_dirs` |
| `tests/test_template_expansion.py` | Template sed expansion, placeholder verification, bash syntax |
| `tests/test_shell_scripts.py` | `bash -n` on all scripts, `test_wp.sh` execution, `test_pipeline.sh` structure |

Tests requiring `openmm` or `mdtraj` are automatically skipped when those packages are not installed.

### E2E test on NERSC

```bash
# Requires SSH access and sshproxy cert
./test_pipeline.sh [PDB_ID]
```

Runs 6 stages: RCSB API search, preprocessing, WESTPA setup, sbatch, iteration polling, output validation. Default PDB: `1JEY`.

### Local mock test

```bash
./test_wp.sh
```

Runs `batch_wp.sh` and `run_wp.sh` with mocked `h5ls`, `squeue`, and `sbatch` in a temp directory.

---

## Repository layout

```
HelixNet/
  app.py                    Streamlit UI
  preprocess_pdb.py         PDB download, repair, solvation
  read_config.py            Config reader for shell scripts
  config.example.json       Example configuration
  requirements.txt          Python dependencies
  run_ui.sh                 UI launcher (creates venv)
  setup_wp.sh               Per-target WESTPA setup
  batch_wp.sh               Batch setup from pdb_ids.json
  run_wp.sh                 Iteration monitor and resubmit
  test_wp.sh                Local mock test
  test_pipeline.sh          E2E NERSC test
  pytest.ini                Test configuration
  westpa_template/
    west.cfg.template       WESTPA config template
    run.slurm.template      Slurm job template
    b.txt.template          Basis states template
    env.sh                  NERSC environment setup
    openmm_explicit_rmsd_p_ca_propagator.py
  scripts/
    demo.sh                 CI smoke test
  tests/
    conftest.py             Shared fixtures
    test_*.py               Test modules
  .github/workflows/
    ci.yml                  GitHub Actions CI
```

---

## License

MIT
