#!/bin/bash
# setup_wp.sh - per-target WESTPA setup and Slurm submission
# Usage: ./setup_wp.sh <PDB_ID>
# With SETUP_STATUS_ONLY=1: outputs OK or FAIL:<step> on last line (for batch_wp.sh)

pdb_id="${1:-}"
if [[ -z "$pdb_id" ]]; then
    [ -z "$SETUP_STATUS_ONLY" ] && echo "Usage: ./setup_wp.sh <PDB_ID>"
    exit 1
fi

# PDB ID sanitization: 4 alphanumeric chars
if [[ ! "$pdb_id" =~ ^[A-Za-z0-9]{4}$ ]]; then
    [ -z "$SETUP_STATUS_ONLY" ] && echo "Error: PDB ID must be exactly 4 alphanumeric characters"
    [ -n "$SETUP_STATUS_ONLY" ] && echo "FAIL:invalid_pdb_id"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$SCRIPT_DIR/read_config.py"
MAMBA_PREFIX=$(python3 "$CFG" paths.micromamba_prefix 2>/dev/null || echo "/global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm")

PROJECT_DIR=$(python3 "$CFG" paths.project_dir 2>/dev/null || echo "$SCRIPT_DIR")
OUT_DIR_RAW=$(python3 "$CFG" paths.out_dir 2>/dev/null || echo "")
if [[ -z "$OUT_DIR_RAW" ]]; then
    OUT_DIR="$PROJECT_DIR"
elif [[ "$OUT_DIR_RAW" == /* ]]; then
    OUT_DIR="$OUT_DIR_RAW"
else
    OUT_DIR="$PROJECT_DIR/$OUT_DIR_RAW"
fi
mkdir -p "$OUT_DIR"

eval "$(micromamba shell hook --shell bash 2>/dev/null)" || true
micromamba activate "$MAMBA_PREFIX" 2>/dev/null || true

cd "$OUT_DIR"
"$SCRIPT_DIR/preprocess_pdb.py" "$pdb_id"
preprocess_rc=$?
if [[ $preprocess_rc -ne 0 ]]; then
    [ -n "$SETUP_STATUS_ONLY" ] && echo "FAIL:preprocess_pdb.py"
    rm -rf "${pdb_id}_WP"
    exit $preprocess_rc
fi

ACCOUNT=$(python3 "$CFG" slurm.account)
CONSTRAINT=$(python3 "$CFG" slurm.constraint)
QOS=$(python3 "$CFG" slurm.qos)
WALLTIME=$(python3 "$CFG" slurm.walltime)
NODES=$(python3 "$CFG" slurm.nodes)
NTASKS=$(python3 "$CFG" slurm.ntasks_per_node)
CPUS=$(python3 "$CFG" slurm.cpus_per_task)
GPUS=$(python3 "$CFG" slurm.gpus_per_task)
TARGET_ITERS=$(python3 "$CFG" westpa.target_iterations)
MAX_WALLCLOCK=$(python3 "$CFG" westpa.max_run_wallclock)
PCOORD_NDIM=$(python3 "$CFG" westpa.pcoord_ndim)
PCOORD_LEN=$(python3 "$CFG" westpa.pcoord_len)
NBINS=$(python3 "$CFG" westpa.nbins)
BIN_TARGET=$(python3 "$CFG" westpa.bin_target_counts)
TEMPERATURE=$(python3 "$CFG" openmm.temperature)
TIMESTEP=$(python3 "$CFG" openmm.timestep)
FRICTION=$(python3 "$CFG" openmm.friction)
PRESSURE=$(python3 "$CFG" openmm.pressure)
BAROSTAT_INT=$(python3 "$CFG" openmm.barostat_interval)
CONST_TOL=$(python3 "$CFG" openmm.constraint_tolerance)
HMASS=$(python3 "$CFG" openmm.hydrogen_mass)
STEPS=$(python3 "$CFG" openmm.steps)
SAVE_STEPS=$(python3 "$CFG" openmm.save_steps)
GPU_PREC=$(python3 "$CFG" openmm.gpu_precision)
FF_RAW=$(python3 "$CFG" openmm.forcefield)
FF_0=$(echo "$FF_RAW" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())[0])")
FF_1=$(echo "$FF_RAW" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())[1])")

sed "s|{{PDB_ID}}|$pdb_id|g; s|{{ACCOUNT}}|$ACCOUNT|g; s|{{CONSTRAINT}}|$CONSTRAINT|g; s|{{QOS}}|$QOS|g; s|{{WALLTIME}}|$WALLTIME|g; s|{{NODES}}|$NODES|g; s|{{NTASKS}}|$NTASKS|g; s|{{CPUS}}|$CPUS|g; s|{{GPUS}}|$GPUS|g" \
    "$SCRIPT_DIR/westpa_template/run.slurm.template" > "${pdb_id}_WP/run.slurm"

sed "s|{{PDB_ID}}|$pdb_id|g; s|{{PROJECT_DIR}}|$OUT_DIR|g; s|{{TARGET_ITERATIONS}}|$TARGET_ITERS|g; s|{{MAX_RUN_WALLCLOCK}}|$MAX_WALLCLOCK|g; s|{{PCOORD_NDIM}}|$PCOORD_NDIM|g; s|{{PCOORD_LEN}}|$PCOORD_LEN|g; s|{{NBINS}}|$NBINS|g; s|{{BIN_TARGET_COUNTS}}|$BIN_TARGET|g; s|{{NUM_GPUS}}|$GPUS|g; s|{{GPU_PRECISION}}|$GPU_PREC|g; s|{{FF_0}}|$FF_0|g; s|{{FF_1}}|$FF_1|g; s|{{TEMPERATURE}}|$TEMPERATURE|g; s|{{TIMESTEP}}|$TIMESTEP|g; s|{{FRICTION}}|$FRICTION|g; s|{{PRESSURE}}|$PRESSURE|g; s|{{BAROSTAT_INTERVAL}}|$BAROSTAT_INT|g; s|{{CONSTRAINT_TOLERANCE}}|$CONST_TOL|g; s|{{HYDROGEN_MASS}}|$HMASS|g; s|{{STEPS}}|$STEPS|g; s|{{SAVE_STEPS}}|$SAVE_STEPS|g" \
    "$SCRIPT_DIR/westpa_template/west.cfg.template" > "${pdb_id}_WP/west.cfg"

sed "s|{{PDB_ID}}|$pdb_id|g" "$SCRIPT_DIR/westpa_template/b.txt.template" > "${pdb_id}_WP/b.txt"
cp "$SCRIPT_DIR/westpa_template/openmm_explicit_rmsd_p_ca_propagator.py" "${pdb_id}_WP/"
sed "s|{{REPO_DIR}}|$SCRIPT_DIR|g" "$SCRIPT_DIR/westpa_template/env.sh" > "${pdb_id}_WP/env.sh"

cd "${pdb_id}_WP"
chmod +x env.sh
source env.sh

w_init --bstate-file b.txt >/dev/null 2>&1
winit_rc=$?
if [[ $winit_rc -ne 0 ]]; then
    rm -rf traj_segs west.h5
    w_init --bstate-file b.txt >/dev/null 2>&1
    winit_rc=$?
    if [[ $winit_rc -ne 0 ]]; then
        [ -n "$SETUP_STATUS_ONLY" ] && echo "FAIL:w_init"
        cd ..
        rm -rf "${pdb_id}_WP"
        exit $winit_rc
    fi
fi
cd ..

[ -n "$SETUP_STATUS_ONLY" ] && echo "OK"
