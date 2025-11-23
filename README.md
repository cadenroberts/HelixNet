# 🧬 HelixNet

---

## 🧱 Repository Structure

---
```yaml
batch_wp.sh
preprocess_pdb.py
run.sh
setup_wp.sh
it.sh
westpa_template/
  ├── west.cfg.template
  ├── run.slurm.template
  ├── b.txt.template
  └── openmm_explicit_rmsd_p_ca_propagator.py
```

### batch_wp.sh
Script to submit a .json of PDB_IDs to setup_wp.sh

### preprocess_pdb.py
Script to preprocess a pdb for openmm explicit solvent.

### run.sh
Script to check all *_WP folders in the directory and resubmit them if they are beneath a goal # of iterations.

### setup_wp.sh
Script to preprocess a pdb file, copy templated files to a new directory {PDB_ID}_WP, initialize a new WESTPA simulation, and submit a SLURM job to run the simulation if everything is successful (delete directory otherwise).

### it.sh
Script to check all *_WP folders # of iterations.

### west_template/
- **`openmm_explicit_rmsd_p_ca_propagator.py`** — Implements explicit-solvent OpenMM simulations. Includes barostat control, solvent boxes, and hydrogen constraints. Implements **RMSD (Root-Mean Square Deviation)** progress coordinate on P and CA backbones.
  
- **`west.cfg.template`** - The `west.cfg` file is the **master configuration file** controlling WESTPA’s runtime behavior, binning scheme, and propagator settings.

- **`run.slurm.template`** - Slurm deployment on Nersc.

- **`b.txt.template`** - Define the states the WESTPA simulation can start.
