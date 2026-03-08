# HelixNet

Distributed simulation orchestration for WESTPA/OpenMM molecular dynamics on NERSC Slurm. Targets DNA-protein complexes with GPU-accelerated weighted ensemble sampling.

## What it does

- Downloads PDB structures from RCSB, repairs via PDBFixer, parameterizes ligands (GAFF) from RCSB GraphQL SMILES
- Solvates with TIP3P water and ions, generates per-target WESTPA configs from templates
- Submits GPU simulations to NERSC, monitors iteration progress, resubmits incomplete runs

## Quick start

```bash
# Single target
./setup_wp.sh 1ABC

# Batch from JSON list
./batch_wp.sh pdb_list.json

# Monitor and resubmit
./run.sh
```

## Pipeline

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams.

```
PDB ID → preprocess_pdb.py (RCSB, PDBFixer, GAFF, solvation)
       → setup_wp.sh (sed templates, w_init, sbatch)
       → GPU nodes (OpenMMExplicitPropagator, PME, Langevin, barostat, P+CA RMSD)
       → west.h5 + traj_segs (solute-only DCD, NPZ)
       → run.sh (h5ls, resubmit if iter < 12,500)
```

## Entry points

| Script | Purpose |
|--------|---------|
| `preprocess_pdb.py` | PDB download, fixup, ligand parameterization, solvation |
| `setup_wp.sh` | Per-target setup, template expansion, w_init, Slurm submit |
| `batch_wp.sh` | Batch from JSON PDB list |
| `run.sh` | Iteration monitor, resubmit below target |

## Propagator

`OpenMMExplicitPropagator`: PME electrostatics, Monte Carlo barostat (300 K, 1 atm), H-mass repartitioning (1.5 amu, 4 fs timestep), P+CA RMSD progress coordinate, solute-only DCD output, XML checkpoints.

## Design decisions

- **Templates**: `sed` on `{{PDB_ID}}` placeholders — inspectable, no extra deps
- **Solute-only DCD**: ~10× storage reduction; solvent positions unavailable post-hoc
- **4 fs timestep**: H-mass repartitioning; validated for equilibrium, not kinetics
- **P+CA RMSD**: 1D progress coordinate; simple, may miss orthogonal motions
- **MAB binning**: Adaptive boundaries, less manual tuning
- **w_init retry**: Cleanup + retry once; second failure deletes directory

## Evaluation

**Correctness**: Preprocessing exit 0, `west.h5` valid, first iteration non-zero pcoord, monitoring detects incomplete runs.

**Metrics**: Simulation reproducibility (fixed seeds), iteration throughput, walker convergence, preprocessing success rate, storage efficiency.

**Scaling**: Targets linear; iterations linear in wall time; system size O(N log N) for PME; GPU count near-linear.

**Storage** (12.5k iter, 6 walkers/bin, 9 bins): `west.h5` 5–50 GB, `traj_segs` 10–100 GB, `seg.npz` 1–10 GB.

## Demo

**Smoke test** (local, needs OpenMM/PDBFixer/RDKit/openmmforcefields + network):

```bash
./preprocess_pdb.py 1L2Y
# Exit 0, 1L2Y_WP/raw/, 1L2Y_WP/processed/, forcefield.json
./scripts/demo.sh   # SMOKE_OK
```

**Full demo** (NERSC only): `./setup_wp.sh 1L2Y`, `watch -n 60 './run.sh'`, `h5ls west.h5/iterations`.

## Repository layout

```
HelixNet/
├── preprocess_pdb.py
├── setup_wp.sh, batch_wp.sh, run.sh, sync.sh
├── westpa_template/
│   ├── west.cfg.template, run.slurm.template, b.txt.template
│   ├── env.sh
│   └── openmm_explicit_rmsd_p_ca_propagator.py
└── scripts/demo.sh
```

## Dependencies

Python: openmm, pdbfixer, rdkit, openff.toolkit, openmmforcefields, mdtraj, numpy, requests, westpa. Bash, sed, Slurm, MPI, HDF5, NERSC modules, Micromamba.

## Configuration

Hardcoded in templates: NERSC paths (`/global/cfs/cdirs/m4229/caden/...`), Slurm account `m4229`, `max_total_iterations` 12,500, `nbins` 9, `steps` 1,000, `timestep` 4 fs, `hydrogenMass` 1.5 amu.

## Failure modes

| Failure | Behavior |
|---------|----------|
| PDB download fails | `requests.get` raises |
| Unmatched residues | RuntimeError, directory not created |
| `w_init` fails | Retry once after cleanup; delete on second failure |
| Slurm timeout | `run.sh` resubmits if iter < 12,500 |
| GPU unavailable | Propagator falls back to CPU with warning |
| Node crash | WESTPA resumes from `west.h5` |

## Limitations

- Hardcoded NERSC paths; no portable execution
- No offline mode (RCSB required)
- No dependency pinning (no requirements.txt/environment.yml)
- Sequential batch preprocessing
- PDB ID unsanitized (potential path traversal)
- `run.sh` line 34: submission may be commented out
- No automated convergence detection

## Improvements (from audit)

**P0**: Sanitize PDB ID (4-char alphanumeric); CI (`.github/workflows/ci.yml` present).

**P1**: Remove hardcoded paths; add requirements.txt/environment.yml; parameterize Slurm; document offline path.

**P2**: Structured logging; `--help` flags; progress reporting in `run.sh`.

## License

MIT
