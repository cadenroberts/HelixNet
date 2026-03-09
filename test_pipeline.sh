#!/bin/bash
# test_pipeline.sh - E2E pipeline test (runs on Mac, SSHs to NERSC)
#
# Prerequisites:
#   1. Run `sshproxy -u cawrober` or `na` first to get a valid proxy cert
#   2. Ensure ~/.ssh/nersc exists
#
# Usage: ./test_pipeline.sh [PDB_ID]
#   Default PDB_ID: 1JEY (87 residues, fast preprocessing)

set -euo pipefail

PDB_ID="${1:-1JEY}"
SSH_KEY=~/.ssh/nersc
NERSC_USER=cawrober
NERSC_HOST=perlmutter.nersc.gov
PROJECT_DIR=/global/cfs/cdirs/m4229/caden/westpa_dna_protein
TEST_OUT_DIR="test_e2e_$$"
REPO_DIR="$PROJECT_DIR"
POLL_INTERVAL=30
POLL_TIMEOUT=900   # 15 minutes
TARGET_ITERS=10

PASS=0
FAIL=0
STAGES=()

ssh_cmd() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 \
        -l "$NERSC_USER" -i "$SSH_KEY" "$NERSC_HOST" "$@"
}

stage_start() {
    STAGE_NAME="$1"
    STAGE_START=$(date +%s)
    echo ""
    echo "=== Stage: $STAGE_NAME ==="
}

stage_pass() {
    local elapsed=$(( $(date +%s) - STAGE_START ))
    echo "  PASS (${elapsed}s)"
    STAGES+=("PASS ${elapsed}s  $STAGE_NAME")
    ((PASS++))
}

stage_fail() {
    local elapsed=$(( $(date +%s) - STAGE_START ))
    echo "  FAIL: $1 (${elapsed}s)"
    STAGES+=("FAIL ${elapsed}s  $STAGE_NAME - $1")
    ((FAIL++))
}

cleanup() {
    echo ""
    echo "=== Cleanup ==="
    ssh_cmd "rm -rf $PROJECT_DIR/$TEST_OUT_DIR" 2>/dev/null || true
    echo "  Removed $PROJECT_DIR/$TEST_OUT_DIR on NERSC"
}

print_summary() {
    echo ""
    echo "========================================"
    echo "  SUMMARY: $PASS pass, $FAIL fail"
    echo "========================================"
    for s in "${STAGES[@]}"; do
        echo "  $s"
    done
    echo ""
}

# -------------------------------------------------------------------
# Verify SSH connectivity
# -------------------------------------------------------------------
echo "Checking SSH to $NERSC_HOST..."
if ! ssh_cmd "echo ok" >/dev/null 2>&1; then
    echo "ERROR: SSH failed. Run sshproxy or na first."
    exit 1
fi
echo "SSH OK"

# -------------------------------------------------------------------
# Create test config on NERSC
# -------------------------------------------------------------------
echo "Creating test config ($TEST_OUT_DIR)..."
ssh_cmd "bash -l" <<REMOTE_SETUP
set -e
mkdir -p "$PROJECT_DIR/$TEST_OUT_DIR"
cat > "$PROJECT_DIR/$TEST_OUT_DIR/config.json" <<'CFGEOF'
{
  "execution": {
    "nersc_user": "$NERSC_USER"
  },
  "paths": {
    "project_dir": "$PROJECT_DIR",
    "out_dir": "$TEST_OUT_DIR",
    "micromamba_prefix": "/global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm",
    "westpa_env_prefix": "/global/cfs/cdirs/m4229/caden/micromamba_root/envs/westpa_env"
  },
  "rcsb_search": {
    "keywords": ["DNA"],
    "keyword_operator": "contains_phrase",
    "organism": "Homo sapiens",
    "max_resolution": 2.5,
    "return_type": "entry"
  },
  "slurm": {
    "account": "m4229",
    "constraint": "gpu",
    "qos": "debug",
    "walltime": "00:15:00",
    "nodes": 1,
    "ntasks_per_node": 1,
    "cpus_per_task": 8,
    "gpus_per_task": 1
  },
  "westpa": {
    "target_iterations": $TARGET_ITERS,
    "max_run_wallclock": "00:15:00",
    "pcoord_ndim": 1,
    "pcoord_len": 11,
    "nbins": 9,
    "bin_target_counts": 6
  },
  "openmm": {
    "temperature": 300.0,
    "timestep": 4.0,
    "friction": 1.0,
    "pressure": 1.0,
    "barostat_interval": 25,
    "constraint_tolerance": 1e-6,
    "hydrogen_mass": 1.5,
    "steps": 1000,
    "save_steps": 100,
    "gpu_precision": "mixed",
    "forcefield": ["amber14-all.xml", "amber14/tip3pfb.xml"]
  },
  "preprocessing": {
    "padding_nm": 1.0,
    "ionic_strength_M": 0.15,
    "ph": 7.0
  }
}
CFGEOF
REMOTE_SETUP
echo "Test config created"

HELIXNET_CFG_DIR="$PROJECT_DIR/$TEST_OUT_DIR"
OUT_DIR="$PROJECT_DIR/$TEST_OUT_DIR"

# -------------------------------------------------------------------
# Stage 0: RCSB API search
# -------------------------------------------------------------------
stage_start "RCSB API search"
RCSB_RESULT=$(ssh_cmd "bash -l" <<'RCSB_EOF'
python3 -c "
import requests, json
payload = {
    'query': {
        'type': 'group',
        'logical_operator': 'and',
        'nodes': [
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.struct_keywords',
                'operator': 'contains_phrase', 'value': 'DNA'}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entity_source_organism.scientific_name',
                'operator': 'exact_match', 'value': 'Homo sapiens'}},
            {'type': 'terminal', 'service': 'numeric', 'parameters': {
                'attribute': 'rcsb_entry_info.resolution_combined',
                'operator': 'less_or_equal', 'value': 2.5}}
        ]
    },
    'return_type': 'entry',
    'request_options': {'paginate': {'start': 0, 'rows': 5}}
}
r = requests.post('https://search.rcsb.org/rcsbsearch/v2/query', json=payload, timeout=30)
if r.status_code == 200:
    d = r.json()
    tc = d.get('total_count', 0)
    ids = [x['identifier'] for x in d.get('result_set', [])]
    print(f'OK total_count={tc} first={ids[:3]}')
elif r.status_code == 204:
    print('FAIL: 204 no content')
else:
    print(f'FAIL: HTTP {r.status_code}')
"
RCSB_EOF
) || true
echo "  $RCSB_RESULT"
if [[ "$RCSB_RESULT" == OK* ]]; then
    stage_pass
else
    stage_fail "$RCSB_RESULT"
fi

# -------------------------------------------------------------------
# Stage 1: preprocess_pdb.py
# -------------------------------------------------------------------
stage_start "preprocess_pdb.py $PDB_ID"
PREPROCESS_RC=0
ssh_cmd "bash -l" <<PREPROCESS_EOF || PREPROCESS_RC=\$?
set -e
export HELIXNET_CONFIG_DIR="$HELIXNET_CFG_DIR"
cd "$PROJECT_DIR"
eval "\$(micromamba shell hook --shell bash 2>/dev/null)" || true
micromamba activate /global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm 2>/dev/null || true
cd "$OUT_DIR"
python3 "$PROJECT_DIR/preprocess_pdb.py" "$PDB_ID"
PREPROCESS_EOF

if [[ $PREPROCESS_RC -ne 0 ]]; then
    stage_fail "preprocess exit code $PREPROCESS_RC"
else
    VERIFY=$(ssh_cmd "bash -c '
        raw=\"$OUT_DIR/${PDB_ID}_WP/raw/${PDB_ID}.pdb\"
        proc=\"$OUT_DIR/${PDB_ID}_WP/processed/${PDB_ID}_processed.pdb\"
        ff=\"$OUT_DIR/${PDB_ID}_WP/processed/forcefield.json\"
        ok=true
        [ ! -f \"\$raw\" ] && echo \"missing raw pdb\" && ok=false
        [ ! -f \"\$proc\" ] && echo \"missing processed pdb\" && ok=false
        [ ! -f \"\$ff\" ] && echo \"missing forcefield.json\" && ok=false
        if \$ok; then
            raw_atoms=\$(grep -c \"^ATOM\" \"\$raw\" 2>/dev/null || echo 0)
            proc_atoms=\$(grep -c \"^ATOM\" \"\$proc\" 2>/dev/null || echo 0)
            echo \"OK raw_atoms=\$raw_atoms proc_atoms=\$proc_atoms\"
        fi
    '") || true
    echo "  $VERIFY"
    if [[ "$VERIFY" == OK* ]]; then
        stage_pass
    else
        stage_fail "$VERIFY"
    fi
fi

# -------------------------------------------------------------------
# Stage 2: setup_wp.sh
# -------------------------------------------------------------------
stage_start "setup_wp.sh $PDB_ID"
SETUP_RC=0
ssh_cmd "bash -l" <<SETUP_EOF || SETUP_RC=\$?
set -e
export HELIXNET_CONFIG_DIR="$HELIXNET_CFG_DIR"
cd "$PROJECT_DIR"
./setup_wp.sh "$PDB_ID"
SETUP_EOF

if [[ $SETUP_RC -ne 0 ]]; then
    stage_fail "setup_wp.sh exit code $SETUP_RC"
else
    VERIFY=$(ssh_cmd "bash -c '
        cfg=\"$OUT_DIR/${PDB_ID}_WP/west.cfg\"
        slurm=\"$OUT_DIR/${PDB_ID}_WP/run.slurm\"
        h5=\"$OUT_DIR/${PDB_ID}_WP/west.h5\"
        env=\"$OUT_DIR/${PDB_ID}_WP/env.sh\"
        ok=true
        [ ! -f \"\$cfg\" ] && echo \"missing west.cfg\" && ok=false
        [ ! -f \"\$slurm\" ] && echo \"missing run.slurm\" && ok=false
        [ ! -s \"\$h5\" ] && echo \"missing/empty west.h5\" && ok=false
        [ ! -f \"\$env\" ] && echo \"missing env.sh\" && ok=false
        if \$ok; then
            iters=\$(grep \"max_total_iterations\" \"\$cfg\" | head -1)
            account=\$(grep \"SBATCH -A\" \"\$slurm\" | head -1)
            has_repo=\$(grep -c \"REPO_DIR\" \"\$env\" 2>/dev/null || echo 0)
            echo \"OK iters=[\$iters] account=[\$account] repo_placeholder=\$has_repo\"
        fi
    '") || true
    echo "  $VERIFY"
    if [[ "$VERIFY" == OK* ]]; then
        stage_pass
    else
        stage_fail "$VERIFY"
    fi
fi

# -------------------------------------------------------------------
# Stage 3: sbatch
# -------------------------------------------------------------------
stage_start "sbatch run.slurm"
JOBID=""
SBATCH_OUT=$(ssh_cmd "bash -l" <<SBATCH_EOF
set -e
cd "$OUT_DIR/${PDB_ID}_WP"
sbatch run.slurm
SBATCH_EOF
) || true
echo "  $SBATCH_OUT"

JOBID=$(echo "$SBATCH_OUT" | grep -oE '[0-9]{5,}' | tail -1)
if [[ -z "$JOBID" ]]; then
    stage_fail "no job ID from sbatch"
else
    SQUEUE_CHECK=$(ssh_cmd "squeue -j $JOBID --noheader" 2>/dev/null) || true
    if [[ -n "$SQUEUE_CHECK" ]]; then
        echo "  Job $JOBID in queue"
        stage_pass
    else
        echo "  Job $JOBID not found in queue (may have started and finished already)"
        stage_pass
    fi
fi

# -------------------------------------------------------------------
# Stage 4: poll for iterations
# -------------------------------------------------------------------
stage_start "Wait for $TARGET_ITERS iterations (timeout ${POLL_TIMEOUT}s)"
if [[ -z "$JOBID" ]]; then
    stage_fail "no job to poll"
else
    ELAPSED=0
    LAST_ITER=0
    POLL_PASS=false
    while (( ELAPSED < POLL_TIMEOUT )); do
        sleep "$POLL_INTERVAL"
        ELAPSED=$(( ELAPSED + POLL_INTERVAL ))

        ITER_OUT=$(ssh_cmd "bash -c '
            h5ls \"$OUT_DIR/${PDB_ID}_WP/west.h5/iterations\" 2>/dev/null \
                | awk \"/^iter_/ { split(\\\$1,a,\\\"_\\\"); v=a[2] } END { print v+0 }\"
        '" 2>/dev/null) || ITER_OUT="0"
        ITER_OUT=$(echo "$ITER_OUT" | tr -d '[:space:]')
        [[ -z "$ITER_OUT" ]] && ITER_OUT=0

        JOB_STATE=$(ssh_cmd "squeue -j $JOBID --noheader -o '%T'" 2>/dev/null | tr -d '[:space:]') || JOB_STATE=""

        echo "  ${ELAPSED}s: iter=$ITER_OUT job=$JOB_STATE"
        LAST_ITER=$ITER_OUT

        if (( ITER_OUT >= TARGET_ITERS )); then
            POLL_PASS=true
            break
        fi

        if [[ -z "$JOB_STATE" ]] && (( ITER_OUT > 0 && ITER_OUT < TARGET_ITERS )); then
            echo "  Job finished but only reached iter $ITER_OUT"
            break
        fi
        if [[ -z "$JOB_STATE" ]] && (( ITER_OUT == 0 )); then
            echo "  Job gone, 0 iterations - waiting one more cycle..."
            sleep "$POLL_INTERVAL"
            ELAPSED=$(( ELAPSED + POLL_INTERVAL ))
            ITER_OUT=$(ssh_cmd "bash -c '
                h5ls \"$OUT_DIR/${PDB_ID}_WP/west.h5/iterations\" 2>/dev/null \
                    | awk \"/^iter_/ { split(\\\$1,a,\\\"_\\\"); v=a[2] } END { print v+0 }\"
            '" 2>/dev/null) || ITER_OUT="0"
            ITER_OUT=$(echo "$ITER_OUT" | tr -d '[:space:]')
            [[ -z "$ITER_OUT" ]] && ITER_OUT=0
            LAST_ITER=$ITER_OUT
            if (( ITER_OUT >= TARGET_ITERS )); then
                POLL_PASS=true
            fi
            break
        fi
    done

    if $POLL_PASS; then
        stage_pass
    else
        stage_fail "reached $LAST_ITER/$TARGET_ITERS iterations"
    fi
fi

# -------------------------------------------------------------------
# Stage 5: validate output
# -------------------------------------------------------------------
stage_start "Validate output"
VALIDATE=$(ssh_cmd "bash -c '
    h5=\"$OUT_DIR/${PDB_ID}_WP/west.h5\"
    ok=true
    iters=\$(h5ls \"\$h5/iterations\" 2>/dev/null | awk \"/^iter_/ { split(\\\$1,a,\\\"_\\\"); v=a[2] } END { print v+0 }\")
    [[ -z \"\$iters\" ]] && iters=0
    (( iters < $TARGET_ITERS )) && echo \"only \$iters iterations\" && ok=false
    seg_dir=\"$OUT_DIR/${PDB_ID}_WP/traj_segs/000001/000000\"
    [ ! -d \"\$seg_dir\" ] && echo \"missing traj_segs/000001/000000\" && ok=false
    if \$ok; then
        dcd=\$(ls -la \"\$seg_dir/seg.dcd\" 2>/dev/null | awk \"{print \\\$5}\") || dcd=0
        xml_exists=\"no\"
        [ -f \"\$seg_dir/seg.xml\" ] && xml_exists=\"yes\"
        echo \"OK iters=\$iters dcd_bytes=\$dcd xml=\$xml_exists\"
    fi
'" 2>/dev/null) || true
echo "  $VALIDATE"
if [[ "$VALIDATE" == OK* ]]; then
    stage_pass
else
    stage_fail "${VALIDATE:-no output}"
fi

# -------------------------------------------------------------------
# Cleanup and summary
# -------------------------------------------------------------------
cleanup
print_summary

if (( FAIL > 0 )); then
    exit 1
fi
