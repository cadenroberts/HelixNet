# Evaluation

## Metrics

| Metric | Definition | How measured |
|--------|-----------|-------------|
| Simulation reproducibility | Identical inputs produce identical trajectories | Fixed random seeds + XML checkpoint comparison across restarts |
| Iteration throughput | Iterations completed per wall-clock hour | `h5ls west.h5/iterations` count vs. Slurm elapsed time |
| Walker convergence | RMSD distribution stabilization across iterations | Progress coordinate histograms from `west.h5` |
| Preprocessing success rate | Fraction of PDB targets that complete preprocessing | Exit codes from `preprocess_pdb.py` across batch runs |
| Storage efficiency | Bytes per iteration per target | Solute-only DCD size vs. full-system DCD |

## Scaling characteristics

| Dimension | Scaling behavior |
|-----------|-----------------|
| Targets (PDB count) | Linear — each target runs as an independent Slurm job |
| Iterations per target | Linear in wall time — each iteration runs a fixed number of MD steps |
| System size (atoms) | Superlinear — PME electrostatics scales as O(N log N); larger solvated systems increase per-step cost |
| GPU count per node | Near-linear speedup for multi-walker segments via `CudaDeviceIndex` round-robin |

## Simulation reproducibility

WESTPA stores all iteration data in `west.h5` (HDF5):
- Per-iteration walker weights, progress coordinates, and parent segment IDs
- Basis state definitions
- Bin boundaries (MAB adaptive)

Given the same processed PDB, force field parameters, and WESTPA configuration, the ensemble evolution is reproducible. Individual trajectories are stochastic (Langevin thermostat), but the weighted ensemble statistics converge.

## Storage profile

Per target with 12,500 iterations, 6 walkers/bin, 9 bins:

| Component | Approximate size |
|-----------|-----------------|
| `west.h5` | 5–50 GB (depends on system size) |
| `traj_segs/` (solute DCD) | 10–100 GB |
| `seg.npz` (forces, energies) | 1–10 GB |
| Processed PDB + configs | <10 MB |

Solute-only DCD reduces trajectory storage by approximately 10× compared to full-system output for typical protein-solvent systems.

## Validation procedure

1. **Preprocessing**: Verify processed PDB atom/residue/bond counts match the Modeller topology (assertions in `preprocess_pdb.py`).
2. **Initialization**: Confirm `w_init` creates valid `west.h5` and `traj_segs/` structure.
3. **Propagation**: Check first iteration completes with non-zero progress coordinates.
4. **Monitoring**: Verify `run.sh` correctly identifies incomplete simulations and resubmits.
5. **Convergence**: Inspect RMSD distributions across late iterations for stabilization.

## Known limitations

- Single-target Slurm submission is sequential in `batch_wp.sh` (no parallel preprocessing).
- Ligand parameterization depends on RCSB GraphQL availability; offline runs require cached SMILES.
- Progress coordinate (P+CA RMSD) is a 1D reduction of high-dimensional conformational space.
- No automated convergence detection — iteration target (12,500) is set manually per study.
