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

(To be completed)

---

## PHASE 5 — CI

(To be completed)

---

## PHASE 6 — FINALIZE

(To be completed)
