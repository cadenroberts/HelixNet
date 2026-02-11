# Demo

## Scope

HelixNet targets NERSC HPC infrastructure with:
- Slurm job scheduler
- GPU nodes (A100)
- Micromamba environment with OpenMM, WESTPA, RDKit
- External network access (RCSB PDB, GraphQL)
- Multi-node MPI execution

**Full demo is not feasible on standard local machines** due to:
1. Hardcoded NERSC paths (`/global/cfs/cdirs/m4229/caden/...`)
2. Slurm-specific submission (`sbatch`, `srun`)
3. GPU requirements (CUDA)
4. Large-scale HDF5 writes (multi-GB `west.h5`)
5. Micromamba environment dependencies (OpenMM with CUDA, WESTPA)

## Smoke path

A local smoke test validates:
1. PDB download and parsing
2. Structural preprocessing (PDBFixer, ligand handling)
3. Template expansion (sed substitution)
4. File structure creation

This does NOT validate:
- WESTPA initialization (`w_init`)
- Slurm submission
- GPU propagation
- HDF5 iteration writes
- Monitoring/resubmission

## Prerequisites

- Python 3.8+
- OpenMM (`conda install -c conda-forge openmm`)
- PDBFixer (`conda install -c conda-forge pdbfixer`)
- RDKit (`conda install -c conda-forge rdkit`)
- OpenFF Toolkit (`conda install -c conda-forge openff-toolkit`)
- openmmforcefields (`conda install -c conda-forge openmmforcefields`)
- requests, numpy

Network access required for:
- RCSB PDB download (`https://files.rcsb.org/download/`)
- RCSB GraphQL ligand queries (`https://data.rcsb.org/graphql`)

## Smoke test commands

```bash
# 1. Download and preprocess a small PDB (1L2Y: 20-residue DNA hairpin)
./preprocess_pdb.py 1L2Y

# Expected output:
#   Folder created: 1L2Y_WP
#   Missing residues: {...}
#   Missing terminals: {...}
#   Missing atoms: {...}
#   After the process
#   Missing residues: {}
#   Missing terminals: {}
#   Missing atoms: {}
#   (No errors, exit code 0)

# 2. Verify directory structure
ls -R 1L2Y_WP/

# Expected:
#   1L2Y_WP/:
#   processed  raw
#
#   1L2Y_WP/processed:
#   1L2Y_processed.pdb  forcefield.json
#
#   1L2Y_WP/raw:
#   1L2Y.pdb

# 3. Validate processed PDB
grep "^ATOM" 1L2Y_WP/processed/1L2Y_processed.pdb | wc -l

# Expected:
#   >5000 atoms (solvated system with explicit water and ions)

# 4. Check forcefield config
cat 1L2Y_WP/processed/forcefield.json

# Expected:
#   ["amber14-all.xml", "amber14/tip3pfb.xml"]

# 5. Template expansion (requires sed)
sed "s/{{PDB_ID}}/1L2Y/g" westpa_template/west.cfg.template > /tmp/test_west.cfg
grep "topology_path" /tmp/test_west.cfg

# Expected:
#   topology_path: /global/cfs/cdirs/m4229/caden/westpa_dna_protein/1L2Y_WP/processed/1L2Y_processed.pdb

# 6. Cleanup
rm -rf 1L2Y_WP /tmp/test_west.cfg
```

## Expected smoke test result

All commands complete with exit code 0. Final line of smoke test output:

```
SMOKE_OK
```

## Full demo (NERSC only)

On NERSC Perlmutter with configured environment:

```bash
# 1. Single target setup and submission
./setup_wp.sh 1L2Y

# Expected:
#   Preprocessing completes
#   Templates expanded
#   w_init succeeds
#   Slurm job submitted
#   Job ID: <SLURM_JOB_ID>

# 2. Monitor iteration progress
watch -n 60 './run.sh'

# Expected:
#   Checking 1L2Y_WP ...
#   → Found last iteration = N
#   Below 12500 — submitting  (if N < 12500)
#   Already over 12500 — skipping  (if N >= 12500)

# 3. Query WESTPA state
h5ls 1L2Y_WP/west.h5/iterations | tail -n 5

# Expected:
#   iter_012496          Group
#   iter_012497          Group
#   iter_012498          Group
#   iter_012499          Group
#   iter_012500          Group

# 4. Verify trajectory output
ls 1L2Y_WP/traj_segs/000001/000000/

# Expected:
#   seg.dcd  seg.npz  seg.xml
```

Full demo completion marker:

```
DEMO_OK
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `requests.exceptions.HTTPError: 404` | Invalid PDB ID or network failure | Verify PDB exists at RCSB; check network |
| `RuntimeError: Structure still contains unmatched residues` | Force field cannot parameterize residue | Ligand SMILES lookup failed; check RCSB GraphQL availability |
| `OSError: Unable to create folder` | Permission error or invalid path | Check write permissions in current directory |
| `w_init` exits with nonzero | Malformed template or stale HDF5 | `setup_wp.sh` retries once with cleanup; check Slurm output for details |
| Propagator falls back to CPU | CUDA not available | Expected on non-GPU nodes; performance degraded |

## Limitations

- Hardcoded NERSC paths prevent portable execution
- No offline mode (requires RCSB network access)
- No deterministic MD (random seed per segment)
- No automated convergence detection
- Monitoring loop (`run.sh`) has commented-out submission line (line 34)
