#!/bin/bash

pdb_id="$1"
if [ -z "$pdb_id" ]; then
    echo "Usage: ./setup_wp.sh <PDB_ID>"
    exit 1
fi


eval "$(micromamba shell hook --shell bash)"

micromamba activate HelixNet/micromamba_root/envs/openmm

./preprocess_pdb.py "$pdb_id"
# output directory

# copy + substitute in all template files
sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/run.slurm.template > "${pdb_id}_WP/run.slurm"

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/west.cfg.template > "${pdb_id}_WP/west.cfg"

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/b.txt.template > "${pdb_id}_WP/b.txt"

cp westpa_template/openmm_implicit_p_ca_propagator.py "${pdb_id}_WP/openmm_implicit_p_ca_propagator.py"
cp westpa_template/openmm_p_ca_propagator.py "${pdb_id}_WP/openmm_p_ca_propagator.py"
cp westpa_template/base_propagator.py "${pdb_id}_WP/base_propagator.py"
cp westpa_template/rmsd_p_ca_progress_coordinate.py "${pdb_id}_WP/rmsd_p_ca_progress_coordinate.py"
cp westpa_template/base_progress_coordinate.py "${pdb_id}_WP/base_progress_coordinate.py"
cp westpa_template/env.sh "${pdb_id}_WP/env.sh"
cp westpa_template/save_npz.py "${pdb_id}_WP/save_npz.py"
cp westpa_template/save_dcd.py "${pdb_id}_WP/save_dcd.py"

cd "${pdb_id}_WP"

chmod +x env.sh
source env.sh
w_init --bstate-file b.txt
sbatch run.slurm
