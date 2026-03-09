# HelixNet

Distributed simulation orchestration for WESTPA/OpenMM molecular dynamics on NERSC Slurm. Targets DNA-protein complexes with GPU-accelerated weighted ensemble sampling. Includes a Streamlit UI for configuration, RCSB search, and pipeline control.

## What it does

- Downloads PDB structures from RCSB, repairs via PDBFixer, parameterizes ligands (GAFF) from RCSB GraphQL SMILES
- Solvates with TIP3P water and ions, generates per-target WESTPA configs from templates
- Submits GPU simulations to NERSC, monitors iteration progress, resubmits incomplete runs
- Provides a browser-based UI for editing all parameters, running RCSB searches, and launching the pipeline

## Quick start

```bash
# 1. Copy and edit config
cp config.example.json config.json

# 2. Launch the UI (creates .venv, installs deps, runs Streamlit)
./run_ui.sh

# Or run scripts directly:
./batch_wp.sh   # batch setup from pdb_ids.json
./run_wp.sh     # monitor and resubmit
./setup_wp.sh 1ABC  # single target
```

### Running from Mac (SSH mode)

Mode is auto-detected from hostname. On a Mac, the UI uses SSH to connect to NERSC. Fill in `execution.nersc_user` in `config.json` (or via the credentials gate). Uses your default SSH key (`~/.ssh`).

### Running on NERSC login node

```bash
streamlit run app.py --server.port 8501
# On your Mac, forward the port:
# ssh -L 8501:localhost:8501 <user>@perlmutter.nersc.gov
```

## Configuration

All parameters live in a single `config.json` (see `config.example.json` for defaults). Every script and template reads from this file. Sections:

| Section | Controls |
|---------|----------|
| `execution` | NERSC user (mode auto-detected, host=perlmutter.nersc.gov) |
| `paths` | Project directory, out directory (*_WP location), micromamba/WESTPA env prefixes |
| `rcsb_search` | Keywords, organism, resolution, return type |
| `slurm` | Account, constraint, QoS, walltime, nodes, tasks, GPUs |
| `westpa` | Target iterations, wallclock, pcoord, bins |
| `openmm` | Temperature, timestep, friction, pressure, H-mass, forcefield |
| `preprocessing` | Padding, ionic strength, pH |

## Pipeline

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams.

```
PDB ID -> preprocess_pdb.py (RCSB, PDBFixer, GAFF, solvation)
       -> setup_wp.sh (template expansion from config.json, w_init)
       -> GPU nodes (OpenMMExplicitPropagator, PME, Langevin, barostat, P+CA RMSD)
       -> west.h5 + traj_segs (solute-only DCD, NPZ)
       -> run_wp.sh (h5ls, dashboard, resubmit if iter < target)
```

## Entry points

| Script | Purpose |
|--------|---------|
| `app.py` | Streamlit UI: config editor, RCSB search, pipeline control, status |
| `preprocess_pdb.py` | PDB download, fixup, ligand parameterization, solvation |
| `setup_wp.sh` | Per-target setup, template expansion, w_init |
| `batch_wp.sh` | Batch from pdb_ids.json, calls setup then run |
| `run_wp.sh` | Iteration monitor, dashboard, resubmit below target |

## Propagator

`OpenMMExplicitPropagator`: PME electrostatics, Monte Carlo barostat (300 K, 1 atm), H-mass repartitioning (1.5 amu, 4 fs timestep), P+CA RMSD progress coordinate, solute-only DCD output, XML checkpoints.

## Design decisions

- **config.json**: Single source of truth for all parameters; no hardcoded values in scripts
- **Templates**: `sed` on `{{PLACEHOLDER}}` patterns, inspectable, no extra deps
- **Dual execution**: Local subprocess on NERSC, SSH via paramiko from Mac
- **Solute-only DCD**: ~10x storage reduction; solvent positions unavailable post-hoc
- **4 fs timestep**: H-mass repartitioning; validated for equilibrium, not kinetics
- **P+CA RMSD**: 1D progress coordinate; simple, may miss orthogonal motions
- **MAB binning**: Adaptive boundaries, less manual tuning
- **w_init retry**: Cleanup + retry once; second failure deletes directory

## Repository layout

```
HelixNet/
├── app.py                  # Streamlit UI
├── read_config.py          # config.json reader for shell scripts
├── config.example.json     # Checked-in example config
├── requirements.txt        # streamlit, paramiko, requests
├── preprocess_pdb.py
├── setup_wp.sh, batch_wp.sh, run_wp.sh, sync.sh
├── westpa_template/
│   ├── west.cfg.template, run.slurm.template, b.txt.template
│   ├── env.sh
│   └── openmm_explicit_rmsd_p_ca_propagator.py
└── scripts/demo.sh
```

## Dependencies

Python: openmm, pdbfixer, rdkit, openff.toolkit, openmmforcefields, mdtraj, numpy, requests, westpa, streamlit, paramiko. Bash, sed, Slurm, MPI, HDF5, NERSC modules, Micromamba.

## Failure modes

| Failure | Behavior |
|---------|----------|
| PDB download fails | `requests.get` raises |
| Unmatched residues | RuntimeError, directory not created |
| `w_init` fails | Retry once after cleanup; delete on second failure |
| Slurm timeout | `run_wp.sh` resubmits if iter < target |
| GPU unavailable | Propagator falls back to CPU with warning |
| Node crash | WESTPA resumes from `west.h5` |
| SSH connection fails | UI shows paramiko error, no remote side effects |

## License

MIT
