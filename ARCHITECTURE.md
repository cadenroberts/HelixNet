# Architecture

## Configuration flow

All parameters live in `config.json`. The Streamlit UI (`app.py`) reads and writes this file. Shell scripts use `read_config.py` to extract values. Templates are expanded by `setup_wp.sh` using values from config.

```
┌─────────────────────────┐
│  Streamlit UI (app.py)  │
│  ┌───────────────────┐  │
│  │  Config Editor    │  │──── reads/writes ──── config.json
│  │  RCSB Search      │  │──── POST ──────────── RCSB API
│  │  Pipeline Control │  │──── subprocess/SSH ── batch_wp.sh / run_wp.sh
│  │  Status Dashboard │  │──── scan *_WP dirs
│  └───────────────────┘  │
└─────────────────────────┘
         │
    local or SSH
         │
         ▼
┌─────────────────────────┐
│  NERSC / local shell    │
│  scripts read config    │
│  via read_config.py     │
└─────────────────────────┘
```

## Execution modes (auto-detected from hostname)

```
Local (hostname contains "nersc" or "perlmutter"):
  browser ──SSH tunnel──▶ streamlit ──▶ subprocess.run(["./batch_wp.sh"])

SSH (e.g. Mac):
  browser ──▶ streamlit ──▶ paramiko.SSHClient ──▶ NERSC login node ──▶ scripts
```

## Pipeline overview

```
                    ┌──────────────────┐
                    │  config.json     │
                    │  (all params)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  pdb_ids.json    │
                    │  (from RCSB or   │
                    │   manual entry)  │
                    └────────┬─────────┘
                             │
                      batch_wp.sh
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ setup_wp.sh │  │ setup_wp.sh │  │ setup_wp.sh │  (one per PDB)
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
 ┌──────────┼──────────┐    │                │
 │          ▼          │    ▼                ▼
 │  preprocess_pdb.py  │
 │  (reads config.json │
 │   for pH, padding,  │
 │   ionic strength,   │
 │   forcefield)       │
 │  ┌────────────────┐ │
 │  │ 1. Download PDB│ │
 │  │ 2. PDBFixer    │ │
 │  │ 3. Ligand GAFF │ │
 │  │ 4. Solvate     │ │
 │  │ 5. Validate    │ │
 │  └────────────────┘ │
 │          │          │
 │          ▼          │
 │  Template expansion │
 │  (all placeholders  │
 │   from config.json) │
 │  west.cfg           │
 │  run.slurm          │
 │  b.txt              │
 │          │          │
 │          ▼          │
 │  w_init (WESTPA)    │
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────┐
 │   NERSC GPU Nodes   │
 │   (A100 / Slurm)    │
 │                     │
 │   WESTPA w_run      │
 │   └─ Propagator     │
 │      └─ OpenMM      │
 │         └─ CUDA     │
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │   west.h5           │
 │   traj_segs/        │
 │   (iteration data)  │
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │   run_wp.sh         │
 │   (reads target     │
 │    from config.json) │
 │   Reads west.h5     │
 │   Resubmits if      │
 │   iter < target     │
 └─────────────────────┘
```

## config.json sections

| Section | Used by | Key values |
|---------|---------|------------|
| `execution` | app.py | nersc_user (mode auto-detected, host=perlmutter.nersc.gov) |
| `paths` | setup_wp.sh, env.sh, batch_wp.sh | project_dir, out_dir (*_WP location), micromamba_prefix, westpa_env_prefix |
| `rcsb_search` | app.py | keywords, organism, max_resolution |
| `slurm` | run.slurm.template | account, constraint, qos, walltime, nodes, tasks, gpus |
| `westpa` | west.cfg.template, run_wp.sh, batch_wp.sh | target_iterations, pcoord, bins |
| `openmm` | west.cfg.template, preprocess_pdb.py | temperature, timestep, forcefield, etc. |
| `preprocessing` | preprocess_pdb.py | padding_nm, ionic_strength_M, ph |

## Preprocessing pipeline

`preprocess_pdb.py` transforms a raw RCSB PDB into a simulation-ready system:

```
Raw PDB (RCSB)
    │
    ├─ PDBFixer
    │  ├─ findMissingResidues() - skip terminal insertions
    │  ├─ findNonstandardResidues() - replaceNonstandardResidues()
    │  ├─ findMissingAtoms() - addMissingAtoms()
    │  └─ addMissingHydrogens(pH from config.json)
    │
    ├─ Ligand handling
    │  ├─ RDKit: fragment PDB, identify non-protein residues
    │  ├─ RCSB GraphQL: fetch canonical SMILES per comp_id
    │  ├─ AssignBondOrdersFromTemplate() - add hydrogens
    │  ├─ Remove old ligand residues from Modeller
    │  ├─ Add corrected ligand topologies back
    │  └─ GAFFTemplateGenerator: parameterize for forcefield from config.json
    │
    ├─ Solvation
    │  ├─ Forcefield from config.json
    │  ├─ addSolvent(padding, ionicStrength from config.json)
    │  └─ Validate topology (residues, atoms, bonds match)
    │
    └─ Output
       ├─ {PDB_ID}_processed.pdb
       ├─ forcefield.json
       └─ {PDB_ID}_processed_ligands_smiles.json (if ligands present)
```

## Propagator hierarchy

```
WESTPropagator (WESTPA base)
    └─ BasePropagator
       ├─ _load_config()          [abstract]
       ├─ _init_pcoord_calculator()  - RMSDProgressCoordinate(P + CA)
       ├─ propagate()             [abstract]
       └─ OpenMMPropagator
          ├─ _load_config()       - reads west.cfg [west][openmm]
          ├─ _create_simulation() [abstract system creation]
          ├─ _init_segment_state()- load parent XML or minimize from PDB
          ├─ _run_simulation()    - step loop with state snapshots
          ├─ _save_final_state()  - XML checkpoint
          └─ OpenMMExplicitPropagator
             ├─ _create_system()  - PME, HBonds, barostat, H-mass
             ├─ _setup_reporters()- SoluteDCDReporter (solute-only)
             └─ _calculate_pcoord() - RMSD on P/CA atoms
```

## Per-target directory layout

After `setup_wp.sh` completes for PDB ID `1ABC`:

```
1ABC_WP/
├── raw/
│   └── 1ABC.pdb                 Downloaded from RCSB
├── processed/
│   ├── 1ABC_processed.pdb       Solvated, parameterized structure
│   ├── forcefield.json          From config.json openmm.forcefield
│   └── 1ABC_processed_ligands_smiles.json  (if ligands)
├── west.cfg                     Expanded from config.json
├── run.slurm                    Expanded from config.json
├── b.txt                        Basis states
├── env.sh                       Environment activation
├── openmm_explicit_rmsd_p_ca_propagator.py
├── west.h5                      WESTPA iteration data (HDF5)
├── traj_segs/                   Per-iteration trajectory segments
│   ├── 000001/
│   │   ├── 000000/
│   │   │   ├── seg.dcd          Solute trajectory
│   │   │   ├── seg.xml          Checkpoint state
│   │   │   └── seg.npz          Forces, energies, times
│   │   └── ...
│   └── ...
└── istates/                     Initial states
```
