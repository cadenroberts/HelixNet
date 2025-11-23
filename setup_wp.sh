#!/bin/bash

pdb_id="$1"
if [ -z "$pdb_id" ]; then
    echo "Usage: ./setup_wp.sh <PDB_ID>"
    exit 1
fi


eval "$(micromamba shell hook --shell bash)"

micromamba activate /global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm

./preprocess_pdb.py "$pdb_id"
preprocess_rc=$?

if [ $preprocess_rc -ne 0 ]; then
    echo "!!! Preprocessing FAILED for $pdb_id (exit code $preprocess_rc)"
    echo "!!! Giving up. Not runnning w_init. Cleaning directory ${pdb_id}_WP."
    rm -rf "${pdb_id}_WP"
    exit $preprocess_rc
fi

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/run.slurm.template > "${pdb_id}_WP/run.slurm"

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/west.cfg.template > "${pdb_id}_WP/west.cfg"

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/b.txt.template > "${pdb_id}_WP/b.txt"

cp westpa_template/openmm_explicit_rmsd_p_ca_propagator.py "${pdb_id}_WP/openmm_explicit_rmsd_p_ca_propagator.py"
cp westpa_template/env.sh "${pdb_id}_WP/env.sh"

cd "${pdb_id}_WP"

chmod +x env.sh
source env.sh
w_init --bstate-file b.txt
winit_rc=$?
if [ $winit_rc -ne 0 ]; then
    echo "!!! w_init FAILED for $pdb_id (exit code $winit_rc)"
    echo "!!! Cleaning ${pdb_id}_WP/traj_segs and ${pdb_id}_WP/west.h5."
    rm -rf "traj_segs"
    rm -f  "west.h5"
    w_init --bstate-file b.txt
    winit_rc=$?
    if [ $winit_rc -ne 0 ]; then
        echo "!!! w_init FAILED AGAIN for $pdb_id (attempt 2, exit code $winit_rc)"
        echo "!!! Giving up. Not submitting WESTPA job. Cleaning directory ${pdb_id}_WP."
        rm -rf "${pdb_id}_WP"
        exit $winit_rc
    fi
fi
sbatch run.slurm
