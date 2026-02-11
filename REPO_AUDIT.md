# Repository Audit

## 1. Purpose

HelixNet is a distributed simulation orchestration system for weighted ensemble molecular dynamics. Given a list of PDB identifiers, it automates the full pipeline: downloading structures from RCSB, structural preprocessing (missing atom reconstruction, ligand parameterization, solvation), WESTPA configuration generation from templates, Slurm job submission to HPC infrastructure, and automated monitoring with resubmission of incomplete simulations.

The system targets DNA-protein complexes on NERSC GPU nodes, using OpenMM for propagation and WESTPA for weighted ensemble sampling with adaptive binning.

## 2. Entry points

| Entry point | Type | Purpose |
|-------------|------|---------|
| `preprocess_pdb.py` | Python script | PDB download, PDBFixer structural repair, ligand identification and GAFF parameterization, explicit solvation |
| `setup_wp.sh` | Bash script | Per-target orchestration: calls preprocessing, expands templates, initializes WESTPA, submits Slurm job |
| `batch_wp.sh` | Bash script | Sequential batch processing from JSON PDB list |
| `run.sh` | Bash script | Iteration monitor: scans `west.h5` files, resubmits jobs below iteration target |

All entry points are invoked by the user directly or via batch submission. No daemon processes.

## 3. Dependency surface

### Runtime dependencies

- Python 3.x
  - openmm (GPU-accelerated MD)
  - pdbfixer (structural repair)
  - rdkit (ligand bond order assignment)
  - openff.toolkit (ligand molecule representation)
  - openmmforcefields (GAFF template generation)
  - mdtraj (trajectory I/O and RMSD calculation)
  - numpy
  - requests (RCSB PDB and GraphQL queries)
  - westpa (weighted ensemble framework)
- Bash + sed (template expansion)
- Slurm (job submission: sbatch, srun)
- MPI (mpiexec, pmix)
- HDF5 tools (h5ls for iteration counting)
- NERSC-specific modules (openmpi, cudatoolkit)
- Micromamba (environment management)

### Development dependencies

None explicitly managed. No test framework, linter config, or build tooling present.

## 4. Configuration surface

### Environment-specific

- `westpa_template/env.sh`
  - Micromamba paths: `/global/homes/c/cawrober/micromamba` (hardcoded username)
  - Environment path: `/global/cfs/cdirs/m4229/caden/micromamba_root/envs/westpa_env`
  - NERSC module paths: `/global/common/software/m3169/perlmutter/modulefiles`
  - TMPDIR set to `$PSCRATCH`

- `westpa_template/west.cfg.template`
  - Topology path: `/global/cfs/cdirs/m4229/caden/westpa_dna_protein/{{PDB_ID}}_WP/processed/{{PDB_ID}}_processed.pdb` (hardcoded project path)

- `westpa_template/run.slurm.template`
  - Slurm account: `m4229`
  - GPU constraint: `-C gpu`, queue: `regular`

- `setup_wp.sh`
  - Hardcoded Micromamba activation: `/global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm`

### Tunable parameters (west.cfg.template)

| Parameter | Value | Location |
|-----------|-------|----------|
| `max_total_iterations` | 12,500 | `west.propagation` |
| `max_run_wallclock` | 72:00:00 | `west.propagation` |
| `bin_target_counts` | 6 | `west.system.system_options` |
| `nbins` | 9 | `west.system.system_options.bins` |
| `steps` | 1,000 | `west.openmm` |
| `save_steps` | 100 | `west.openmm` |
| `timestep` | 4.0 fs | `west.openmm` |
| `temperature` | 300 K | `west.openmm` |
| `pressure` | 1.0 atm | `west.openmm` |
| `hydrogenMass` | 1.5 amu | `west.openmm` |
| `gpu_precision` | mixed | `west.openmm` |

### No .env, flags, or external config files

Configuration is embedded in templates and scripts. No command-line flag parsing.

## 5. Data flow

```
User PDB ID
    ↓
setup_wp.sh
    ↓
preprocess_pdb.py
    ├→ RCSB HTTP GET → raw PDB
    ├→ PDBFixer → structural repair
    ├→ RDKit + RCSB GraphQL → ligand SMILES
    ├→ GAFF template generation → ligand cache JSON
    ├→ OpenMM solvation → processed PDB
    └→ {PDB_ID}_WP/processed/
         ├ {PDB_ID}_processed.pdb
         ├ forcefield.json
         └ {PDB_ID}_processed_ligands_smiles.json (if ligands)
    ↓
setup_wp.sh (template expansion)
    ├→ sed {{PDB_ID}} → west.cfg, run.slurm, b.txt
    └→ {PDB_ID}_WP/
    ↓
w_init (WESTPA initialization)
    ↓
sbatch run.slurm (Slurm submission)
    ↓
GPU nodes (MPI-parallel WESTPA w_run)
    ├→ Propagator: OpenMMExplicitPropagator
    │    ├→ Load parent XML checkpoint (or minimize from PDB)
    │    ├→ Run MD for N steps
    │    ├→ Write solute-only DCD + NPZ (forces, energies)
    │    └→ Save XML checkpoint
    ├→ Calculate RMSD on P + CA atoms
    └→ Write to west.h5 (HDF5)
         ├ iteration metadata
         ├ walker weights
         ├ progress coordinates
         └ bin assignments
    ↓
run.sh (monitoring loop)
    ├→ h5ls {PDB_ID}_WP/west.h5/iterations
    ├→ Parse last iteration count
    └→ sbatch run.slurm (if iter < 12,500)
```

## 6. Determinism risks

| Risk | Source | Mitigation |
|------|--------|------------|
| Stochastic MD trajectories | LangevinMiddleIntegrator | Random seed set per segment; ensemble statistics reproducible |
| External network calls | RCSB PDB download, RCSB GraphQL ligand queries | Cached after first preprocessing; offline runs impossible without cache |
| HDF5 file locking | Multi-node writes to `west.h5` | `HDF5_USE_FILE_LOCKING=0` set in `env.sh` |
| GPU floating-point variance | CUDA mixed precision | Fixed by `gpu_precision: mixed`; no cross-platform reproducibility guaranteed |
| Template path substitution | `sed` with `{{PDB_ID}}` | Deterministic string replacement; no escaping issues for 4-char PDB IDs |
| Micromamba environment drift | No lockfile | Environment paths hardcoded; version drift possible across installs |

## 7. Observability

### Logs

- Slurm output: `{PDB_ID}_WP.out` (stdout/stderr from `run.slurm`)
- No structured logging framework
- `print()` statements in Python (preprocessing, propagator)
- Script output to terminal (setup, batch, monitor)

### Metrics

- WESTPA iteration count: `west.h5/iterations` (queryable via `h5ls`)
- Walker weights, progress coordinates: `west.h5` HDF5 structure
- Forces, energies, times: `traj_segs/{iter}/{seg}/seg.npz`
- Segment wall time: stored in `Segment.walltime` (not easily queryable post-run)

### Error handling

- Exit codes used consistently (preprocessing, w_init)
- `setup_wp.sh` retries `w_init` once, then deletes directory on second failure
- No error recovery in propagator (segment-level failures kill the job)
- `run.sh` silently skips directories missing `west.h5` or `run.slurm`

## 8. Test state

- No test suite
- No test files
- No CI configuration
- No coverage measurement

Validation is implicit:
- Assertions in `preprocess_pdb.py` (line 232-238): topology residue/atom/bond counts after write/read
- WESTPA's internal validation (w_init fails on malformed config)

## 9. Reproducibility

| Component | Status | Notes |
|-----------|--------|-------|
| Dependency versions | Not pinned | No `requirements.txt`, `environment.yml`, or lockfile |
| Force field parameters | Pinned | `forcefield.json` per target: `["amber14-all.xml", "amber14/tip3pfb.xml"]` |
| Ligand parameterization | Cached | `{PDB_ID}_processed_ligands_cache.json` after first run |
| Template files | Version-controlled | `westpa_template/*.template` committed |
| Random seeds | Per-segment | `integrator.setRandomNumberSeed(random.randint(1, 1000000))` |
| Build steps | None | No compilation or build process |

## 10. Security surface

| Surface | Exposure |
|---------|----------|
| PDB ID input | Unsanitized; directly interpolated into paths and URLs |
| RCSB HTTP requests | No TLS verification shown; `requests.get()` raises on non-200 |
| RCSB GraphQL queries | Unsanitized comp_id in query string (line 63) |
| File writes | `{PDB_ID}_WP/` directory created with user permissions; no validation of PDB ID format before filesystem ops |
| Environment variables | Hardcoded paths to user home directories (usernames visible) |
| Secrets | None (no API keys, tokens, or credentials) |
| External command execution | `sed`, `sbatch`, `w_init` called with user input |

Potential injection: malformed PDB IDs (e.g., `../../etc/passwd`) could escape intended directory structure. No validation beyond implicit 4-character assumption.

## 11. Ranked improvement list

### P0 (Blocks demo/verification)

1. Create `DEMO.md` with concrete demo path
2. Create `scripts/demo.sh` with reproducible verification procedure
3. Add `.github/workflows/ci.yml` for CI integration
4. Sanitize PDB ID input: validate 4-character alphanumeric format before filesystem operations

### P1 (Critical for portability)

5. Remove hardcoded NERSC paths from templates and scripts
6. Add `requirements.txt` or `environment.yml` with pinned dependency versions
7. Parameterize Slurm account, queue, and resource requests (currently hardcoded to `m4229`)
8. Document offline preprocessing path (cached SMILES required for no-network runs)

### P2 (Quality of life)

9. Add basic smoke test: preprocess a small PDB (e.g., 1L2Y), initialize WESTPA, validate `west.h5` structure
10. Structured logging (replace `print()` with `logging` module)
11. Add `--help` / `-h` flags to scripts
12. Progress bar or iteration throughput reporting in `run.sh`
