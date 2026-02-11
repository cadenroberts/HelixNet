# Design Decisions

## ADR-1: Template-based configuration over programmatic generation

**Context**: Each PDB target requires a unique WESTPA configuration, Slurm script, and basis state file.

**Decision**: Use shell-based `sed` substitution on `{{PDB_ID}}` placeholders in version-controlled templates.

**Tradeoff**: Less flexible than a Python config generator, but templates are directly inspectable, diffable, and require zero dependencies beyond `sed`. Adding a new parameter requires editing one template rather than a code path.

**Rejected alternative**: Jinja2 template engine — adds a Python dependency to what is otherwise a pure shell orchestration layer.

## ADR-2: Solute-only trajectory storage

**Context**: Explicit-solvent simulations produce trajectories dominated by water atoms (often >90% of atoms). Full-system DCD files consume substantial storage across thousands of iterations.

**Decision**: `SoluteDCDReporter` filters positions to solute atoms before writing DCD. Full-system data (forces, energies) is stored in compressed NPZ.

**Tradeoff**: Post-hoc analysis requiring solvent positions is impossible without re-running simulations. Solute-only trajectories reduce storage by ~10× for typical protein-water systems.

**Rejected alternative**: Full-system DCD with post-hoc stripping — defers the storage problem and requires a second processing pass.

## ADR-3: Retry-once-then-delete failure policy for w_init

**Context**: WESTPA initialization (`w_init`) occasionally fails due to stale HDF5 locks or corrupted segment directories, particularly after node crashes.

**Decision**: On `w_init` failure, delete `traj_segs/` and `west.h5`, retry once. On second failure, delete the entire `{PDB_ID}_WP` directory and exit with error.

**Tradeoff**: Aggressive cleanup risks losing partial data, but WESTPA initialization is fast (<1 min) and stale state is the most common failure mode. A corrupted directory that passes `w_init` causes silent downstream errors.

**Rejected alternative**: Manual intervention queue — increases operator burden for a failure mode that is almost always resolved by clean restart.

## ADR-4: RMSD on P and CA backbone atoms as progress coordinate

**Context**: WESTPA requires a progress coordinate for adaptive binning. DNA-protein systems contain both phosphorus (P) backbone atoms and protein alpha-carbon (CA) atoms.

**Decision**: Use combined P + CA RMSD relative to the initial (crystal) structure as the 1D progress coordinate.

**Tradeoff**: A single RMSD coordinate collapses all conformational change into one number, potentially missing orthogonal motions. Multi-dimensional progress coordinates (e.g., separate protein and DNA RMSD) increase bin space exponentially.

**Rejected alternative**: Contact-map-based coordinates — more informative but computationally expensive to evaluate at every segment boundary.

## ADR-5: Hydrogen mass repartitioning for 4 fs timestep

**Context**: Standard integration timesteps for biomolecular MD are 2 fs with constrained bonds. Doubling the timestep halves wall time per iteration.

**Decision**: Set `hydrogenMass=1.5` amu, redistributing mass from heavy atoms to bonded hydrogens. This enables a 4 fs timestep with `LangevinMiddleIntegrator` and `HBonds` constraints.

**Tradeoff**: Slightly alters vibrational frequencies of hydrogen-containing bonds. Well-validated for equilibrium sampling in the literature (Hopkins et al., JCTC 2015). Not appropriate for kinetic property calculations.

**Rejected alternative**: Standard 2 fs timestep — doubles compute cost with no benefit for the equilibrium conformational sampling targeted by WESTPA.

## ADR-6: MAB adaptive binning over fixed bins

**Context**: WESTPA's binning scheme determines how walkers are distributed across progress coordinate space. Fixed bins require manual tuning per system.

**Decision**: Use `MABBinMapper` (Minimum Adaptive Binning) with 9 bins and logarithmic output.

**Tradeoff**: MAB adapts bin boundaries during the simulation, reducing the need for per-system tuning. It adds overhead for bin boundary recalculation but eliminates the risk of poorly placed fixed bins that waste walkers in uninteresting regions.

**Rejected alternative**: Fixed uniform bins — simpler but requires manual calibration of bin boundaries for each PDB target.
