# HelixNet

Distributed simulation orchestration system for WESTPA/OpenMM molecular dynamics workloads.

## What it does

- Downloads PDB structures from RCSB and repairs missing atoms/residues via PDBFixer
- Identifies small-molecule ligands, retrieves canonical SMILES from RCSB GraphQL, and parameterizes with GAFF
- Solvates structures with explicit TIP3P water and ions
- Generates per-target WESTPA configurations from version-controlled templates
- Submits GPU-accelerated weighted ensemble simulations to NERSC Slurm infrastructure
- Monitors iteration progress and resubmits incomplete simulations

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full diagrams.

### Pipeline stages

```
PDB ID
  ↓
preprocess_pdb.py
  ├→ RCSB HTTP GET → raw PDB
  ├→ PDBFixer → structural repair
  ├→ RDKit + RCSB GraphQL → ligand SMILES
  ├→ GAFF template generation → ligand cache
  └→ OpenMM solvation → processed PDB
  ↓
setup_wp.sh
  ├→ sed {{PDB_ID}} → west.cfg, run.slurm, b.txt
  ├→ w_init (WESTPA initialization)
  └→ sbatch (Slurm submission)
  ↓
GPU nodes (MPI-parallel w_run)
  ├→ OpenMMExplicitPropagator
  │    ├→ PME electrostatics
  │    ├→ Langevin integrator (300 K, 4 fs timestep)
  │    ├→ Monte Carlo barostat (1 atm)
  │    ├→ H-mass repartitioning (1.5 amu)
  │    └→ Solute-only DCD + NPZ output
  ├→ RMSD calculation (P + CA atoms)
  └→ west.h5 (HDF5 iteration data)
  ↓
run.sh
  ├→ h5ls west.h5/iterations
  └→ resubmit if iter < 12,500
```

### Propagator

`OpenMMExplicitPropagator` runs explicit-solvent Langevin dynamics with:
- PME electrostatics
- Monte Carlo barostat (300 K, 1 atm)
- Hydrogen mass repartitioning (1.5 amu, enables 4 fs timestep)
- RMSD progress coordinate on P and CA backbone atoms
- Solute-only DCD trajectory output (strips solvent for storage efficiency)
- XML checkpoint serialization for segment continuation

### Failure modes

| Failure | Behavior |
|---------|----------|
| PDB download fails | `requests.get` raises; `preprocess_pdb.py` exits nonzero |
| Missing residues cannot be resolved | PDBFixer fails; preprocessing exits |
| Unmatched force field residues | RuntimeError raised after ligand fixup; directory not created |
| `w_init` fails | Retries once after cleaning `traj_segs` and `west.h5`; deletes directory on second failure |
| Slurm job times out | `run.sh` detects iteration count below target and resubmits |
| GPU mismatch at runtime | Propagator falls back to CPU platform with warning |
| Node crash mid-segment | WESTPA resumes from last completed iteration via `west.h5` checkpoint |

## Design tradeoffs

**Template-based configuration**: Shell `sed` substitution on `{{PDB_ID}}` placeholders rather than Python config generation. Templates are directly inspectable and diffable, require zero dependencies beyond `sed`, but less flexible than programmatic generation.

**Solute-only trajectory storage**: DCD files contain only solute atoms, reducing storage by ~10× for typical protein-water systems. Full-system forces/energies stored separately in compressed NPZ. Post-hoc analysis requiring solvent positions is impossible without re-running.

**4 fs timestep via hydrogen mass repartitioning**: Setting `hydrogenMass=1.5` amu enables 4 fs integration, halving compute cost. Slightly alters H-bond vibrational frequencies but well-validated for equilibrium sampling. Not appropriate for kinetic properties.

**P+CA RMSD progress coordinate**: Combined phosphorus and alpha-carbon RMSD collapses conformational change into one number. Simple and fast to compute, but potentially misses orthogonal motions. Multi-dimensional coordinates increase bin space exponentially.

**MAB adaptive binning**: Bin boundaries adapt during simulation, reducing manual tuning. Adds recalculation overhead but eliminates risk of poorly placed fixed bins wasting walkers in uninteresting regions.

**Retry-once-then-delete**: On `w_init` failure, cleanup and retry once. Second failure deletes directory entirely. Aggressive but resolves the most common failure mode (stale HDF5 locks). Corrupted directories that pass initialization cause silent downstream errors.

## Evaluation

See [EVAL.md](EVAL.md) for detailed metrics.

### Correctness definition

A correct execution satisfies:
1. Preprocessing produces a solvated structure with all force field residues matched
2. WESTPA initialization creates valid `west.h5` structure
3. First iteration completes with non-zero progress coordinates
4. Monitoring correctly identifies incomplete simulations (iter < 12,500)
5. Topology validation: processed PDB atom/residue/bond counts match Modeller topology

### Commands

```bash
# Preprocessing correctness
./preprocess_pdb.py 1L2Y
# Exit code 0 + assertions pass (lines 232-238 in script)

# WESTPA initialization
cd 1L2Y_WP && source env.sh && w_init --bstate-file b.txt
# Exit code 0 + west.h5 created

# Progress coordinate validation
h5ls 1L2Y_WP/west.h5/iterations/iter_000001/pcoord
# Dataset shows non-zero RMSD values

# Monitoring logic
./run.sh
# Correctly reports iteration count and submission decision
```

### Pass/fail criteria

- Preprocessing: exit code 0, no RuntimeError on unmatched residues
- WESTPA init: `west.h5` exists and readable by `h5ls`
- Propagation: `seg.dcd`, `seg.npz`, `seg.xml` present after iteration 1
- Monitoring: correct iteration count parsed, resubmission triggered if below target

## Demo

See [DEMO.md](DEMO.md) for full instructions.

### Quick start

```bash
# Single target
./setup_wp.sh 1ABC

# Batch from JSON list
./batch_wp.sh pdb_list.json

# Monitor and resubmit incomplete simulations
./run.sh
```

### Expected output (preprocessing)

```
Folder created: 1ABC_WP
Missing residues: {...}
Missing terminals: {...}
Missing atoms: {...}
After the process
Missing residues: {}
Missing terminals: {}
Missing atoms: {}
```

### Expected output (monitoring)

```
Checking 1ABC_WP ...
  → Found last iteration = 5432
Below 12500 — submitting
```

## Repository layout

```
HelixNet/
├── preprocess_pdb.py                          PDB download, fixup, solvation, parameterization
├── setup_wp.sh                                Per-target WESTPA setup + Slurm submission
├── batch_wp.sh                                Batch submission from JSON PDB list
├── run.sh                                     Iteration monitor + resubmission
├── sync.sh                                    Git commit and push helper
├── ARCHITECTURE.md                            Full pipeline diagrams and data flow
├── DESIGN_DECISIONS.md                        ADR entries for key tradeoffs
├── EVAL.md                                    Metrics, scaling, validation procedures
├── DEMO.md                                    Smoke test and full demo instructions
├── REPO_AUDIT.md                              Technical audit and improvement roadmap
└── westpa_template/
    ├── west.cfg.template                      WESTPA master configuration
    ├── run.slurm.template                     Slurm job script
    ├── b.txt.template                         Basis state definitions
    ├── env.sh                                 Environment activation
    └── openmm_explicit_rmsd_p_ca_propagator.py  Explicit-solvent propagator with RMSD pcoord
```

## Limitations

- Hardcoded NERSC paths (`/global/cfs/cdirs/m4229/caden/...`) prevent portable execution
- No offline mode: requires RCSB network access for PDB download and ligand SMILES queries
- No dependency version pinning: no `requirements.txt`, `environment.yml`, or lockfile
- Single-target preprocessing is sequential in `batch_wp.sh` (no parallelization)
- Progress coordinate (P+CA RMSD) is a 1D reduction of high-dimensional conformational space
- No automated convergence detection: iteration target (12,500) set manually
- PDB ID input unsanitized: potential directory traversal vulnerability
- Monitoring loop (`run.sh`) has commented-out submission line (line 34)

## License

MIT
