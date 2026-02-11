# HelixNet

Distributed simulation orchestration system for WESTPA/OpenMM molecular dynamics workloads.

## Purpose

HelixNet automates the setup, execution, and monitoring of weighted ensemble (WESTPA) molecular dynamics simulations across GPU-backed HPC nodes. Given a list of PDB identifiers, the system downloads structures from RCSB, preprocesses them (fix missing atoms, add solvent, parameterize ligands), generates per-target WESTPA configurations from templates, and submits Slurm jobs to NERSC infrastructure. A monitoring loop tracks iteration progress and resubmits incomplete simulations.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams.

The pipeline has three stages:

1. **Preprocessing** (`preprocess_pdb.py`) — Downloads PDB from RCSB, fixes missing residues/atoms via PDBFixer, replaces nonstandard residues, identifies and parameterizes small-molecule ligands using GAFF/OpenFF, adds explicit TIP3P solvent with 1.0 nm padding and 0.15 M ionic strength, validates topology, and writes the processed structure.

2. **Simulation setup** (`setup_wp.sh`) — Creates a per-target working directory (`{PDB_ID}_WP`), applies PDB-specific values to WESTPA configuration templates (west.cfg, run.slurm, b.txt), copies the propagator and environment scripts, initializes WESTPA state (`w_init`), and submits the Slurm job. Failed preprocessing or initialization triggers cleanup.

3. **Monitoring and resubmission** (`run.sh`) — Scans all `*_WP` directories, reads iteration count from `west.h5`, and resubmits jobs that have not reached the target iteration count (12,500).

## Propagator

`OpenMMExplicitPropagator` runs explicit-solvent Langevin dynamics with:
- PME electrostatics
- Monte Carlo barostat (300 K, 1 atm)
- Hydrogen mass repartitioning (1.5 amu, enables 4 fs timestep)
- RMSD progress coordinate on P and CA backbone atoms
- Solute-only DCD trajectory output (strips solvent for storage efficiency)
- XML checkpoint serialization for segment continuation

## Reproducibility

- Deterministic template expansion: `west.cfg`, `run.slurm`, and `b.txt` are generated from version-controlled templates with `{{PDB_ID}}` substitution.
- Force field configuration is serialized to `forcefield.json` per target.
- Ligand parameterization caches GAFF templates to `{PDB_ID}_processed_ligands_cache.json`.
- WESTPA's `west.h5` stores full iteration history, enabling post-hoc analysis of any simulation state.

## Failure modes

| Failure | Behavior |
|---------|----------|
| PDB download fails | `requests.get` raises; `preprocess_pdb.py` exits nonzero |
| Missing residues cannot be resolved | PDBFixer fails; preprocessing exits |
| Unmatched force field residues | RuntimeError raised after ligand fixup; directory not created |
| `w_init` fails | Retries once after cleaning `traj_segs` and `west.h5`; deletes directory on second failure |
| Slurm job times out | `run.sh` detects iteration count below target and resubmits |
| GPU mismatch at runtime | Propagator falls back to CPU platform with warning |
| Node crash mid-segment | WESTPA resumes from last completed iteration via `west.h5` checkpoint |

## Configuration

Key parameters in `west.cfg.template`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `max_total_iterations` | 12,500 | Total WESTPA iterations |
| `max_run_wallclock` | 72:00:00 | Per-submission wall time |
| `bin_target_counts` | 6 | Walkers per bin |
| `nbins` | 9 | MAB adaptive binning |
| `steps` | 1,000 | MD steps per segment |
| `save_steps` | 100 | Frames saved per segment |
| `timestep` | 4.0 fs | Enabled by H-mass repartitioning |
| `temperature` | 300 K | |
| `gpu_precision` | mixed | CUDA mixed precision |

## Usage

```bash
# Single target
./setup_wp.sh 1ABC

# Batch from JSON list
./batch_wp.sh pdb_list.json

# Monitor and resubmit incomplete simulations
./run.sh
```

## Repo structure

```
HelixNet/
├── preprocess_pdb.py                          PDB download, fixup, solvation, parameterization
├── setup_wp.sh                                Per-target WESTPA setup + Slurm submission
├── batch_wp.sh                                Batch submission from JSON PDB list
├── run.sh                                     Iteration monitor + resubmission
└── westpa_template/
    ├── west.cfg.template                      WESTPA master configuration
    ├── run.slurm.template                     Slurm job script
    ├── b.txt.template                         Basis state definitions
    ├── env.sh                                 Environment activation
    └── openmm_explicit_rmsd_p_ca_propagator.py  Explicit-solvent propagator with RMSD pcoord
```

## License

MIT
