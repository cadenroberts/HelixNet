#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/run.sh"

usage() {
  cat <<'EOF'
Usage: ./test.sh <command> [args]

Commands:
  mock             Local mock test for run flow
  e2e [PDB_ID]     NERSC SSH E2E pipeline test (default PDB_ID: 1JEY)
EOF
}

mock_cmd() {
  local testdir
  testdir=$(mktemp -d)
  trap "rm -rf '$testdir'" EXIT
  cd "$testdir"

  mkdir -p westpa_template
  for f in run.slurm.template west.cfg.template b.txt.template; do
    echo 'PDBID={{PDB_ID}}' > "westpa_template/$f"
  done
  echo '# propagator' > westpa_template/openmm_explicit_rmsd_p_ca_propagator.py
  echo '#!/bin/bash' > westpa_template/env.sh.template

  echo '["1ABC", "2DEF", "3GHI", "4JKL", "5MNO", "6PQR"]' > pdb_ids.json
  cat > config.json <<'CONFIG'
{"paths":{"project_dir":"__TESTDIR__","out_dir":"out","mamba_exe":"micromamba","mamba_root_prefix":"/tmp/mamba_root"},"westpa":{"target_iterations":12500},"slurm":{"account":"x","constraint":"gpu","qos":"regular","walltime":"01:00:00","nodes":1,"ntasks_per_node":1,"cpus_per_task":1,"gpus_per_task":1},"openmm":{"temperature":300.0,"timestep":4.0,"friction":1.0,"pressure":1.0,"barostat_interval":25,"constraint_tolerance":1e-6,"hydrogen_mass":1.5,"steps":1000,"save_steps":100,"gpu_precision":"mixed","forcefield":["amber14-all.xml","amber14/tip3pfb.xml"]},"preprocessing":{"padding_nm":1.0,"ionic_strength_M":0.15,"ph":7.0}}
CONFIG
  sed -i.bak "s|__TESTDIR__|$testdir|g" config.json && rm -f config.json.bak
  export NDMS_CONFIG_DIR="$testdir"

  mkdir -p out
  for id in 1ABC 2DEF 3GHI 4JKL 5MNO 6PQR; do
    mkdir -p "out/${id}_WP"
    echo "test" > "out/${id}_WP/west.h5"
    echo "#!/bin/bash" > "out/${id}_WP/run.slurm"
    chmod +x "out/${id}_WP/run.slurm"
  done

  h5ls() {
      case "$1" in
          *1ABC_WP*) echo "iter_00100 Group"; echo "iter_12500 Group" ;;
          *2DEF_WP*) echo "iter_00100 Group"; echo "iter_05000 Group" ;;
          *3GHI_WP*) echo "iter_00100 Group"; echo "iter_02500 Group" ;;
          *4JKL_WP*) echo "iter_00100 Group"; echo "iter_08000 Group" ;;
          *5MNO_WP*) echo "iter_00100 Group"; echo "iter_10000 Group" ;;
          *6PQR_WP*) echo "iter_00100 Group"; echo "iter_12500 Group" ;;
          *) return 1 ;;
      esac
  }
  export -f h5ls

  export SQUEUE_COUNTER="$PWD/.squeue_count"
  echo 0 > "$SQUEUE_COUNTER"
  squeue() {
      local count
      count=$(<"$SQUEUE_COUNTER")
      if [[ "$*" == *"--noheader"* ]]; then
          echo "zn_prod  R    4:15:00  1 shared_mi 44911401 nid004125"
          echo "2DEF_WP  R    0:45:30  1 gpu_mi    44911403 nid002847"
          echo "4JKL_WP  R    1:22:15  1 shared_mi 44911404 nid003921"
          ((count > 0)) && echo "3GHI_WP  PD   0:00:00  1 shared_mi 44911500 (Priority)"
          ((count > 1)) && echo "5MNO_WP  PD   0:00:00  1 shared_mi 44911501 (Priority)"
      else
          echo "NAME     ST TIME       NODES PARTITION JOBID    NODELIST(REASON)"
          echo "zn_prod  R  4:15:00    1     shared_mi 44911401 nid004125"
          echo "2DEF_WP  R  0:45:30    1     gpu_mi    44911403 nid002847"
          echo "4JKL_WP  R  1:22:15    1     shared_mi 44911404 nid003921"
          ((count > 0)) && echo "3GHI_WP  PD 0:00:00    1     shared_mi 44911500 (Priority)"
          ((count > 1)) && echo "5MNO_WP  PD 0:00:00    1     shared_mi 44911501 (Priority)"
      fi
      return 0
  }
  export -f squeue

  sbatch() {
      echo $(($(<"$SQUEUE_COUNTER") + 1)) > "$SQUEUE_COUNTER"
  }
  export -f sbatch

  echo ""
  echo "=== Running run.sh run flow ==="
  echo ""
  bash "$RUN_SH" run
}

e2e_cmd() {
  local pdb_id="${1:-1JEY}"
  local ssh_key="${NERSC_SSH_KEY:-$HOME/.ssh/nersc}"
  local nersc_user="${NERSC_USER:-cawrober}"
  local nersc_host=perlmutter.nersc.gov
  local project_dir=/global/cfs/cdirs/m4229/caden/westpa_dna_protein
  local test_out_dir="test_e2e_$$"
  local poll_interval=30
  local poll_timeout=1200
  local target_iters=2
  local pass=0
  local fail=0
  local stages=()

  ssh_cmd() {
      ssh -o BatchMode=yes -o ConnectTimeout=10 -l "$nersc_user" -i "$ssh_key" "$nersc_host" "$@"
  }
  stage_start() { STAGE_NAME="$1"; STAGE_START=$(date +%s); echo ""; echo "=== Stage: $STAGE_NAME ==="; }
  stage_pass() { local elapsed=$(( $(date +%s) - STAGE_START )); echo "  PASS (${elapsed}s)"; stages+=("PASS ${elapsed}s  $STAGE_NAME"); ((pass++)) || true; }
  stage_fail() { local elapsed=$(( $(date +%s) - STAGE_START )); echo "  FAIL: $1 (${elapsed}s)"; stages+=("FAIL ${elapsed}s  $STAGE_NAME - $1"); ((fail++)) || true; }
  cleanup() { echo ""; echo "=== Cleanup ==="; [[ -n "${jobid:-}" ]] && { ssh_cmd "scancel $jobid" 2>/dev/null || true; echo "  Cancelled job $jobid"; }; ssh_cmd "rm -rf $project_dir/$test_out_dir" 2>/dev/null || true; echo "  Removed $project_dir/$test_out_dir on NERSC"; }
  print_summary() { echo ""; echo "========================================"; echo "  SUMMARY: $pass pass, $fail fail"; echo "========================================"; for s in "${stages[@]}"; do echo "  $s"; done; echo ""; }

  echo "Checking SSH to $nersc_host..."
  if ! ssh_cmd "echo ok" >/dev/null 2>&1; then
      echo "ERROR: SSH failed. Run sshproxy or na first."
      exit 1
  fi
  echo "SSH OK"
  trap cleanup EXIT

  local repo_dir="$project_dir/$test_out_dir/repo"
  echo "Syncing repo to $nersc_host:$repo_dir..."
  ssh_cmd "mkdir -p $repo_dir"
  rsync -az --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    -e "ssh -o BatchMode=yes -i $ssh_key -l $nersc_user" \
    "$SCRIPT_DIR/" "${nersc_user}@${nersc_host}:${repo_dir}/"
  echo "Sync OK"

  echo "Creating test config ($test_out_dir)..."
  ssh_cmd "bash -l" <<REMOTE_SETUP
set -e
mkdir -p "$project_dir/$test_out_dir"
cat > "$project_dir/$test_out_dir/config.json" <<'CFGEOF'
{
  "execution": {"nersc_user": "$nersc_user"},
  "paths": {"project_dir": "$project_dir", "out_dir": "$test_out_dir", "micromamba_prefix": "/global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm", "westpa_env_prefix": "/global/cfs/cdirs/m4229/caden/micromamba_root/envs/westpa_env", "mamba_exe": "/global/cfs/cdirs/m4229/caden/micromamba_root/bin/micromamba", "mamba_root_prefix": "/global/cfs/cdirs/m4229/caden/micromamba_root"},
  "rcsb_search": {"keywords": ["DNA"], "keyword_operator": "contains_phrase", "organism": "Homo sapiens", "max_resolution": 2.5, "return_type": "entry"},
  "slurm": {"account": "m4229", "constraint": "gpu", "qos": "debug", "walltime": "00:15:00", "nodes": 1, "ntasks_per_node": 1, "cpus_per_task": 8, "gpus_per_task": 1},
  "westpa": {"target_iterations": $target_iters, "max_run_wallclock": "00:15:00", "pcoord_ndim": 1, "pcoord_len": 11, "nbins": 9, "bin_target_counts": 6},
  "openmm": {"temperature": 300.0, "timestep": 4.0, "friction": 1.0, "pressure": 1.0, "barostat_interval": 25, "constraint_tolerance": 1e-6, "hydrogen_mass": 1.5, "steps": 50, "save_steps": 50, "gpu_precision": "mixed", "forcefield": ["amber14-all.xml", "amber14/tip3pfb.xml"]},
  "preprocessing": {"padding_nm": 1.0, "ionic_strength_M": 0.15, "ph": 7.0}
}
CFGEOF
REMOTE_SETUP
  echo "Test config created"

  local ndms_cfg_dir="$project_dir/$test_out_dir"
  local out_dir="$project_dir/$test_out_dir"

  stage_start "RCSB API search"
  local rcsb_result
  rcsb_result=$(ssh_cmd "bash -l" <<'RCSB_EOF'
python3 -c "
import requests
payload = {'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [
    {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'struct_keywords.pdbx_keywords', 'operator': 'contains_phrase', 'value': 'DNA'}},
    {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'rcsb_entity_source_organism.scientific_name', 'operator': 'exact_match', 'value': 'Homo sapiens'}},
    {'type': 'terminal', 'service': 'text', 'parameters': {'attribute': 'rcsb_entry_info.resolution_combined', 'operator': 'less_or_equal', 'value': 2.5}}
]}, 'return_type': 'entry', 'request_options': {'paginate': {'start': 0, 'rows': 5}}}
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
  echo "  $rcsb_result"
  [[ "$rcsb_result" == OK* ]] && stage_pass || stage_fail "$rcsb_result"

  stage_start "benchmark.py preprocess $pdb_id"
  local preprocess_rc=0
  ssh_cmd "bash -l" <<PREPROCESS_EOF || preprocess_rc=$?
set -e
export NDMS_CONFIG_DIR="$ndms_cfg_dir"
eval "\$(micromamba shell hook --shell bash 2>/dev/null)" || true
micromamba activate /global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm 2>/dev/null || true
cd "$out_dir"
python3 "$repo_dir/benchmark.py" preprocess "$pdb_id"
PREPROCESS_EOF
  if [[ $preprocess_rc -ne 0 ]]; then
      stage_fail "preprocess exit code $preprocess_rc"
  else
      stage_pass
  fi

  stage_start "run.sh setup $pdb_id"
  local setup_rc=0
  ssh_cmd "bash -l" <<SETUP_EOF || setup_rc=$?
set -e
export NDMS_CONFIG_DIR="$ndms_cfg_dir"
cd "$repo_dir"
bash ./run.sh setup "$pdb_id"
SETUP_EOF
  if [[ $setup_rc -ne 0 ]]; then
      stage_fail "setup exit code $setup_rc"
  else
      stage_pass
  fi

  stage_start "sbatch run.slurm"
  local sbatch_out jobid
  sbatch_out=$(ssh_cmd "bash -l" <<SBATCH_EOF
set -e
cd "$out_dir/${pdb_id}_WP"
sbatch run.slurm
SBATCH_EOF
) || true
  echo "  $sbatch_out"
  jobid=$(echo "$sbatch_out" | awk '/[0-9]{5,}/{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]{5,}$/) print $i}' | tail -n1)
  if [[ -z "$jobid" ]]; then
      stage_fail "no job ID from sbatch"
  else
      stage_pass
  fi

  stage_start "Wait for $target_iters iterations (timeout ${poll_timeout}s)"
  if [[ -z "${jobid:-}" ]]; then
      stage_fail "no job to poll"
  else
      local elapsed=0 last_iter=0 poll_pass=false
      while (( elapsed < poll_timeout )); do
          sleep "$poll_interval"
          elapsed=$(( elapsed + poll_interval ))
          local iter_out job_state
          job_state=$(ssh_cmd "squeue -j $jobid --noheader -o '%T' 2>/dev/null" 2>/dev/null | tr -d '[:space:]') || job_state=""
          iter_out=$(ssh_cmd "bash -c 'h5ls \"$out_dir/${pdb_id}_WP/west.h5/iterations\" 2>/dev/null | awk \"/^iter_/ { split(\\\$1,a,\\\"_\\\"); v=a[2] } END { print v+0 }\"'" 2>/dev/null) || iter_out="0"
          iter_out=$(echo "$iter_out" | tr -d '[:space:]')
          [[ -z "$iter_out" ]] && iter_out=0
          last_iter=$iter_out
          echo "  ${elapsed}s: iter=$iter_out job=${job_state:-gone}"
          if (( iter_out >= target_iters )); then
              poll_pass=true
              break
          fi
          if [[ -z "$job_state" && $elapsed -gt 120 ]]; then
              echo "  Job no longer in queue, stopping poll"
              break
          fi
      done
      $poll_pass && stage_pass || stage_fail "reached $last_iter/$target_iters iterations"
  fi

  stage_start "Validate output"
  local validate
  validate=$(ssh_cmd "bash -c '
    h5=\"$out_dir/${pdb_id}_WP/west.h5\"
    iters=\$(h5ls \"\$h5/iterations\" 2>/dev/null | awk \"/^iter_/ { split(\\\$1,a,\\\"_\\\"); v=a[2] } END { print v+0 }\")
    [[ -z \"\$iters\" ]] && iters=0
    if (( iters < $target_iters )); then echo \"only \$iters iterations\"; exit 1; fi
    echo \"OK iters=\$iters\"
  '" 2>/dev/null) || true
  echo "  $validate"
  [[ "$validate" == OK* ]] && stage_pass || stage_fail "${validate:-no output}"

  print_summary
  (( fail > 0 )) && exit 1
}

cmd="${1:-}"
case "$cmd" in
  mock)
    shift
    mock_cmd "$@"
    ;;
  e2e)
    shift
    e2e_cmd "$@"
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
