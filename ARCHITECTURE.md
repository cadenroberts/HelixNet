# Architecture

## Pipeline overview

```
                         ┌──────────────┐
                         │  PDB ID(s)   │
                         │  (JSON list) │
                         └──────┬───────┘
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
    ┌──────────┼──────────┐     │                │
    │          ▼          │     ▼                ▼
    │  preprocess_pdb.py  │
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
    │  west.cfg           │
    │  run.slurm          │
    │  b.txt              │
    │          │          │
    │          ▼          │
    │  w_init (WESTPA)    │
    │          │          │
    │          ▼          │
    │  sbatch run.slurm   │
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
    │   run.sh            │
    │   (monitor loop)    │
    │   Reads west.h5     │
    │   Resubmits if      │
    │   iter < 12,500     │
    └─────────────────────┘
```

## Preprocessing pipeline

`preprocess_pdb.py` transforms a raw RCSB PDB into a simulation-ready system:

```
Raw PDB (RCSB)
    │
    ├─ PDBFixer
    │  ├─ findMissingResidues() → skip terminal insertions
    │  ├─ findNonstandardResidues() → replaceNonstandardResidues()
    │  ├─ findMissingAtoms() → addMissingAtoms()
    │  └─ addMissingHydrogens(pH=7.0)
    │
    ├─ Ligand handling
    │  ├─ RDKit: fragment PDB, identify non-protein residues
    │  ├─ RCSB GraphQL: fetch canonical SMILES per comp_id
    │  ├─ AssignBondOrdersFromTemplate() → add hydrogens
    │  ├─ Remove old ligand residues from Modeller
    │  ├─ Add corrected ligand topologies back
    │  └─ GAFFTemplateGenerator: parameterize for Amber14
    │
    ├─ Solvation
    │  ├─ Amber14/TIP3P-FB force field
    │  ├─ addSolvent(padding=1.0 nm, ionicStrength=0.15 M)
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
       ├─ _init_pcoord_calculator()  → RMSDProgressCoordinate(P + CA)
       ├─ propagate()             [abstract]
       └─ OpenMMPropagator
          ├─ _load_config()       → reads west.cfg [west][openmm]
          ├─ _create_simulation() [abstract system creation]
          ├─ _init_segment_state()→ load parent XML or minimize from PDB
          ├─ _run_simulation()    → step loop with state snapshots
          ├─ _save_final_state()  → XML checkpoint
          └─ OpenMMExplicitPropagator
             ├─ _create_system()  → PME, HBonds, barostat, H-mass
             ├─ _setup_reporters()→ SoluteDCDReporter (solute-only)
             └─ _calculate_pcoord() → RMSD on P/CA atoms
```

## Per-target directory layout

After `setup_wp.sh` completes for PDB ID `1ABC`:

```
1ABC_WP/
├── raw/
│   └── 1ABC.pdb                 Downloaded from RCSB
├── processed/
│   ├── 1ABC_processed.pdb       Solvated, parameterized structure
│   ├── forcefield.json          ["amber14-all.xml", "amber14/tip3pfb.xml"]
│   └── 1ABC_processed_ligands_smiles.json  (if ligands)
├── west.cfg                     Expanded WESTPA config
├── run.slurm                    Expanded Slurm script
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
