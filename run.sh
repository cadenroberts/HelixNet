#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="$SCRIPT_DIR/benchmark.py"

cfg() {
  python3 "$BENCHMARK" read-config "$1"
}

usage() {
  cat <<'EOF'
Usage: ./run.sh <command> [args]

Commands:
  setup <PDB_ID>   Per-target WESTPA setup
  run              Monitor/resubmit WESTPA jobs
  batch-setup      Setup all IDs from pdb_ids.json
  batch            Setup all IDs then run
  ui [args...]     Launch Streamlit UI
  demo             Local smoke test
EOF
}

resolve_project_out_dirs() {
  PROJECT_DIR=$(cfg paths.project_dir 2>/dev/null || echo "$SCRIPT_DIR")
  OUT_DIR_RAW=$(cfg paths.out_dir 2>/dev/null || echo "")
  if [[ -z "$OUT_DIR_RAW" ]]; then
    OUT_DIR="$PROJECT_DIR"
  elif [[ "$OUT_DIR_RAW" == /* ]]; then
    OUT_DIR="$OUT_DIR_RAW"
  else
    OUT_DIR="$PROJECT_DIR/$OUT_DIR_RAW"
  fi
}

setup_cmd() {
  local pdb_id="${1:-}"
  if [[ -z "$pdb_id" ]]; then
    [[ -z "${SETUP_STATUS_ONLY:-}" ]] && echo "Usage: ./run.sh setup <PDB_ID>"
    exit 1
  fi
  if [[ ! "$pdb_id" =~ ^[A-Za-z0-9]{4}$ ]]; then
    [[ -z "${SETUP_STATUS_ONLY:-}" ]] && echo "Error: PDB ID must be exactly 4 alphanumeric characters"
    [[ -n "${SETUP_STATUS_ONLY:-}" ]] && echo "FAIL:invalid_pdb_id"
    exit 1
  fi

  local mamba_prefix mamba_exe
  mamba_prefix=$(cfg paths.micromamba_prefix 2>/dev/null || echo "/global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm")
  mamba_exe=$(cfg paths.mamba_exe 2>/dev/null || echo "micromamba")
  resolve_project_out_dirs
  mkdir -p "$OUT_DIR"

  export PATH="$mamba_prefix/bin:$PATH"
  export CONDA_PREFIX="$mamba_prefix"
  python3 -c "import openmm" 2>/dev/null || { echo "ERROR: openmm not importable. Check micromamba activation at $mamba_prefix"; exit 1; }

  cd "$OUT_DIR"
  local preprocess_rc=0
  python3 "$BENCHMARK" preprocess "$pdb_id" || preprocess_rc=$?
  if [[ $preprocess_rc -ne 0 ]]; then
    [[ -n "${SETUP_STATUS_ONLY:-}" ]] && echo "FAIL:preprocess"
    rm -rf "${pdb_id}_WP"
    exit "$preprocess_rc"
  fi

  local account constraint qos walltime nodes ntasks cpus gpus target_iters max_wallclock
  local pcoord_ndim pcoord_len nbins bin_target temperature timestep friction pressure
  local barostat_int const_tol hmass steps save_steps gpu_prec ff_raw ff_list

  account=$(cfg slurm.account)
  constraint=$(cfg slurm.constraint)
  qos=$(cfg slurm.qos)
  walltime=$(cfg slurm.walltime)
  nodes=$(cfg slurm.nodes)
  ntasks=$(cfg slurm.ntasks_per_node)
  cpus=$(cfg slurm.cpus_per_task)
  gpus=$(cfg slurm.gpus_per_task)
  target_iters=$(cfg westpa.target_iterations)
  max_wallclock=$(cfg westpa.max_run_wallclock)
  pcoord_ndim=$(cfg westpa.pcoord_ndim)
  pcoord_len=$(cfg westpa.pcoord_len)
  nbins=$(cfg westpa.nbins)
  bin_target=$(cfg westpa.bin_target_counts)
  temperature=$(cfg openmm.temperature)
  timestep=$(cfg openmm.timestep)
  friction=$(cfg openmm.friction)
  pressure=$(cfg openmm.pressure)
  barostat_int=$(cfg openmm.barostat_interval)
  const_tol=$(cfg openmm.constraint_tolerance)
  hmass=$(cfg openmm.hydrogen_mass)
  steps=$(cfg openmm.steps)
  save_steps=$(cfg openmm.save_steps)
  gpu_prec=$(cfg openmm.gpu_precision)
  ff_raw=$(cfg openmm.forcefield)
  ff_list=$(echo "$ff_raw" | python3 -c "
import sys, json
items = json.loads(sys.stdin.read())
for item in items:
    print('      - ' + item)
")

  sed "s|{{PDB_ID}}|$pdb_id|g; s|{{ACCOUNT}}|$account|g; s|{{CONSTRAINT}}|$constraint|g; s|{{QOS}}|$qos|g; s|{{WALLTIME}}|$walltime|g; s|{{NODES}}|$nodes|g; s|{{NTASKS}}|$ntasks|g; s|{{CPUS}}|$cpus|g; s|{{GPUS}}|$gpus|g" \
    "$SCRIPT_DIR/westpa_template/run.slurm.template" > "${pdb_id}_WP/run.slurm"

  sed "s|{{PDB_ID}}|$pdb_id|g; s|{{PROJECT_DIR}}|$OUT_DIR|g; s|{{TARGET_ITERATIONS}}|$target_iters|g; s|{{MAX_RUN_WALLCLOCK}}|$max_wallclock|g; s|{{PCOORD_NDIM}}|$pcoord_ndim|g; s|{{PCOORD_LEN}}|$pcoord_len|g; s|{{NBINS}}|$nbins|g; s|{{BIN_TARGET_COUNTS}}|$bin_target|g; s|{{NUM_GPUS}}|$gpus|g; s|{{GPU_PRECISION}}|$gpu_prec|g; s|{{TEMPERATURE}}|$temperature|g; s|{{TIMESTEP}}|$timestep|g; s|{{FRICTION}}|$friction|g; s|{{PRESSURE}}|$pressure|g; s|{{BAROSTAT_INTERVAL}}|$barostat_int|g; s|{{CONSTRAINT_TOLERANCE}}|$const_tol|g; s|{{HYDROGEN_MASS}}|$hmass|g; s|{{STEPS}}|$steps|g; s|{{SAVE_STEPS}}|$save_steps|g" \
    "$SCRIPT_DIR/westpa_template/west.cfg.template" | awk -v ff="$ff_list" '{gsub(/{{FF_LIST}}/, ff); print}' > "${pdb_id}_WP/west.cfg"

  sed "s|{{PDB_ID}}|$pdb_id|g" "$SCRIPT_DIR/westpa_template/b.txt.template" > "${pdb_id}_WP/b.txt"
  cp "$SCRIPT_DIR/westpa_template/openmm_explicit_rmsd_p_ca_propagator.py" "${pdb_id}_WP/"

  local mamba_exe mamba_root
  mamba_exe=$(cfg paths.mamba_exe 2>/dev/null || echo "micromamba")
  mamba_root=$(cfg paths.mamba_root_prefix 2>/dev/null || echo "$HOME/micromamba")
  sed "s|{{REPO_DIR}}|$SCRIPT_DIR|g; s|{{MAMBA_EXE}}|$mamba_exe|g; s|{{MAMBA_ROOT_PREFIX}}|$mamba_root|g" \
    "$SCRIPT_DIR/westpa_template/env.sh.template" > "${pdb_id}_WP/env.sh"

  cd "${pdb_id}_WP"
  chmod +x env.sh
  source env.sh

  local winit_rc=0
  w_init --bstate-file b.txt || winit_rc=$?
  if [[ $winit_rc -ne 0 ]]; then
    echo "w_init failed (rc=$winit_rc), retrying..."
    rm -rf traj_segs west.h5
    winit_rc=0
    w_init --bstate-file b.txt || winit_rc=$?
    if [[ $winit_rc -ne 0 ]]; then
      echo "w_init retry failed (rc=$winit_rc)"
      [[ -n "${SETUP_STATUS_ONLY:-}" ]] && echo "FAIL:w_init"
      cd ..
      exit "$winit_rc"
    fi
  fi
  cd ..
  if [[ -n "${SETUP_STATUS_ONLY:-}" ]]; then echo "OK"; fi
}

run_cmd() {
  local target_iterations
  target_iterations=$(cfg westpa.target_iterations)
  resolve_project_out_dirs
  cd "$OUT_DIR"

  local check=0 running=0 submitted=0 errors=0
  local wpdir iterations

  echo "Scanning *_WP directories in $OUT_DIR"
  for wpdir in *_WP; do
    if [[ ! -d "$wpdir" ]]; then
      continue
    fi

    if [[ ! -s "$wpdir/west.h5" ]]; then
      echo "ERROR: $wpdir missing west.h5"
      ((errors++)) || true
      continue
    fi

    if [[ ! -f "$wpdir/run.slurm" ]]; then
      echo "ERROR: $wpdir missing run.slurm"
      ((errors++)) || true
      continue
    fi

    if ! iterations=$(h5ls "$wpdir/west.h5/iterations" | awk '/^iter_/ { split($1, a, "_"); v=a[2] } END { if (v) { printf "%09d", v; exit 0 } else { exit 1 } }'); then
      echo "ERROR: $wpdir unable to read iteration count"
      ((errors++)) || true
      continue
    fi

    if [[ "$iterations" -ge "$target_iterations" ]]; then
      echo "DONE: $wpdir iterations=$iterations target=$target_iterations"
      ((check++)) || true
    elif squeue -u "$USER" | grep -qi "$wpdir" 2>/dev/null; then
      echo "RUNNING: $wpdir iterations=$iterations"
      ((running++)) || true
    else
      (cd "$wpdir" && sbatch run.slurm >/dev/null)
      echo "SUBMITTED: $wpdir iterations=$iterations"
      ((submitted++)) || true
    fi
  done

  echo "Summary: done=$check running=$running submitted=$submitted errors=$errors"
}

batch_setup_cmd() {
  resolve_project_out_dirs

  if [[ ! -s "$SCRIPT_DIR/pdb_ids.json" ]]; then
    echo "ERROR: pdb_ids.json missing or empty"
    exit 1
  fi

  local pdbids=()
  mapfile -t pdbids < <(tr -d '[]"' < "$SCRIPT_DIR/pdb_ids.json" | tr ',' '\n' | sed 's/^ *//;s/ *$//' | grep . | grep -vxFf <(ls -d "$OUT_DIR"/*_WP 2>/dev/null | sed 's|.*/||;s/_WP$//'))

  if [[ ${#pdbids[@]} -eq 0 ]]; then
    echo "No new PDB IDs to set up."
    return
  fi

  local ok=0 failed=0
  local pdbid status
  for pdbid in "${pdbids[@]}"; do
    status=$(SETUP_STATUS_ONLY=1 "$SCRIPT_DIR/run.sh" setup "$pdbid" 2>/dev/null | tail -n1 || true)
    if [[ "$status" == "OK" ]]; then
      echo "SETUP OK: $pdbid"
      ((ok++)) || true
    else
      echo "SETUP FAIL: $pdbid status=${status:-unknown}"
      ((failed++)) || true
    fi
  done

  echo "Batch setup summary: ok=$ok failed=$failed"
}

batch_cmd() {
  batch_setup_cmd
  run_cmd
}

ui_cmd() {
  local venv_dir=".venv"
  local requirements="requirements.txt"
  cd "$SCRIPT_DIR"
  if [[ ! -d "$venv_dir" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$venv_dir"
  fi
  echo "Installing dependencies..."
  "$venv_dir/bin/pip" install -q -r "$requirements"
  echo "Starting NDMS UI..."
  exec "$venv_dir/bin/streamlit" run benchmark.py "$@"
}

demo_cmd() {
  local pdb_id="1L2Y"
  echo "NDMS smoke test"
  cd "$SCRIPT_DIR"

  if [[ -d "${pdb_id}_WP" ]]; then
    rm -rf "${pdb_id}_WP"
  fi

  echo "Step 1: preprocess $pdb_id"
  local prep_rc=0
  python3 "$BENCHMARK" preprocess "$pdb_id" > /tmp/ndms_preprocess.log 2>&1 || prep_rc=$?
  if [[ $prep_rc -ne 0 ]]; then
    echo "FAIL: preprocess exited $prep_rc"
    sed -n '1,200p' /tmp/ndms_preprocess.log
    exit 1
  fi

  echo "Step 2: verify directories and files"
  [[ -d "${pdb_id}_WP/raw" ]] || { echo "FAIL: missing raw dir"; exit 1; }
  [[ -d "${pdb_id}_WP/processed" ]] || { echo "FAIL: missing processed dir"; exit 1; }
  [[ -f "${pdb_id}_WP/raw/${pdb_id}.pdb" ]] || { echo "FAIL: missing raw pdb"; exit 1; }
  [[ -f "${pdb_id}_WP/processed/${pdb_id}_processed.pdb" ]] || { echo "FAIL: missing processed pdb"; exit 1; }
  [[ -f "${pdb_id}_WP/processed/forcefield.json" ]] || { echo "FAIL: missing forcefield.json"; exit 1; }

  echo "Step 3: template expansion check"
  local test_cfg="/tmp/ndms_test_west.cfg"
  sed "s/{{PDB_ID}}/${pdb_id}/g" westpa_template/west.cfg.template > "$test_cfg"
  grep -q "topology_path.*${pdb_id}_WP" "$test_cfg" || { echo "FAIL: template substitution incorrect"; exit 1; }
  local remaining
  remaining=$(grep -o '{{[^}]*}}' "$test_cfg" | grep -v '{{PDB_ID}}' | sort -u || true)
  if [[ -n "$remaining" ]]; then
    echo "INFO: unsubstituted placeholders (expected in demo-only expansion): $remaining"
  fi

  rm -rf "${pdb_id}_WP"
  rm -f "$test_cfg" /tmp/ndms_preprocess.log
  echo "SMOKE_OK"
}

cmd="${1:-}"
case "$cmd" in
  setup)
    shift
    setup_cmd "$@"
    ;;
  run)
    shift
    run_cmd "$@"
    ;;
  batch-setup)
    shift
    batch_setup_cmd "$@"
    ;;
  batch)
    shift
    batch_cmd "$@"
    ;;
  ui)
    shift
    ui_cmd "$@"
    ;;
  demo)
    shift
    demo_cmd "$@"
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
