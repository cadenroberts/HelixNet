# Patchset Summary

## PHASE 0 — BASELINE SNAPSHOT

**Repository**: HelixNet  
**Branch**: main  
**Baseline commit**: 9bb5cea535741b0024cb4f7c714fee3644babfd4  
**Tracked files**: 10

### Current entry points

- `preprocess_pdb.py` — PDB download, structural fixup, ligand parameterization, solvation
- `setup_wp.sh` — Per-target WESTPA setup and Slurm submission
- `batch_wp.sh` — Batch submission from JSON PDB list
- `run.sh` — Iteration monitor and resubmission loop

### How the project runs

1. User provides a PDB ID (4-character code)
2. `setup_wp.sh <PDB_ID>` calls `preprocess_pdb.py` to download and prepare the structure
3. Template files in `westpa_template/` are expanded with PDB-specific values
4. WESTPA is initialized (`w_init`) and a Slurm job is submitted
5. `run.sh` monitors `west.h5` iteration counts and resubmits incomplete simulations

### Existing documentation artifacts

- README.md (present)
- ARCHITECTURE.md (present)
- DESIGN_DECISIONS.md (present, 6 ADR entries)
- EVAL.md (present)

### Missing required artifacts

- DEMO.md
- REPO_AUDIT.md
- scripts/ directory
- scripts/demo.sh
- .github/workflows/ci.yml

---

## PHASE 1 — TECHNICAL AUDIT

(To be completed)

---

## PHASE 2 — CLEANING

(To be completed)

---

## PHASE 3 — DOCUMENTATION REBUILD

(To be completed)

---

## PHASE 4 — VERIFICATION IMPLEMENTATION

Created `scripts/demo.sh` smoke test:

- Validates preprocessing (PDB download, PDBFixer, solvation)
- Verifies directory structure (`{PDB_ID}_WP/raw/`, `{PDB_ID}_WP/processed/`)
- Checks raw PDB atom count (>100 atoms)
- Checks processed PDB atom count (>5000 atoms for solvated system)
- Validates forcefield configuration (amber14-all.xml present)
- Tests template expansion (sed substitution of `{{PDB_ID}}`)
- Cleans up test artifacts

**Verification limitation**: Smoke test requires OpenMM, PDBFixer, RDKit, openmmforcefields, and network access (RCSB). Cannot run in standard CI without these dependencies. Script exits with `SMOKE_OK` on success.

**Local execution blocked by**:
- Missing OpenMM/WESTPA environment
- RCSB network dependency
- Large conda/pip dependency surface

Full demo requires NERSC infrastructure (Slurm, GPU nodes, MPI).

---

## PHASE 5 — CI

Created `.github/workflows/ci.yml`:

- Triggers on push and pull_request to main
- Uses ubuntu-latest runner
- Sets up Miniconda with Python 3.10
- Installs dependencies via mamba: openmm, pdbfixer, rdkit, openff-toolkit, openmmforcefields, numpy, requests
- Executes `scripts/demo.sh`
- Fails on non-zero exit code

CI validates:
- PDB download and preprocessing pipeline
- Template expansion logic
- Directory structure creation
- Force field configuration

CI does NOT validate:
- WESTPA initialization (requires WESTPA package)
- Slurm submission (requires HPC infrastructure)
- GPU propagation (requires CUDA)
- Full simulation execution

---

## PHASE 6 — FINALIZE

(To be completed)
