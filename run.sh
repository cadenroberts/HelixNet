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
  batch            Setup all IDs from pdb_ids.json, then run
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

  local mamba_prefix
  mamba_prefix=$(cfg paths.micromamba_prefix 2>/dev/null || echo "/global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm")
  resolve_project_out_dirs
  mkdir -p "$OUT_DIR"

  eval "$(micromamba shell hook --shell bash 2>/dev/null)" || true
  micromamba activate "$mamba_prefix" 2>/dev/null || true

  cd "$OUT_DIR"
  python3 "$BENCHMARK" preprocess "$pdb_id"
  local preprocess_rc=$?
  if [[ $preprocess_rc -ne 0 ]]; then
    [[ -n "${SETUP_STATUS_ONLY:-}" ]] && echo "FAIL:preprocess"
    rm -rf "${pdb_id}_WP"
    exit "$preprocess_rc"
  fi

  local account constraint qos walltime nodes ntasks cpus gpus target_iters max_wallclock
  local pcoord_ndim pcoord_len nbins bin_target temperature timestep friction pressure
  local barostat_int const_tol hmass steps save_steps gpu_prec ff_raw ff_0 ff_1

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
  ff_0=$(echo "$ff_raw" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())[0])")
  ff_1=$(echo "$ff_raw" | python3 -c "import sys,json; print(json.loads(sys.stdin.read())[1])")

  sed "s|{{PDB_ID}}|$pdb_id|g; s|{{ACCOUNT}}|$account|g; s|{{CONSTRAINT}}|$constraint|g; s|{{QOS}}|$qos|g; s|{{WALLTIME}}|$walltime|g; s|{{NODES}}|$nodes|g; s|{{NTASKS}}|$ntasks|g; s|{{CPUS}}|$cpus|g; s|{{GPUS}}|$gpus|g" \
    "$SCRIPT_DIR/westpa_template/run.slurm.template" > "${pdb_id}_WP/run.slurm"

  sed "s|{{PDB_ID}}|$pdb_id|g; s|{{PROJECT_DIR}}|$OUT_DIR|g; s|{{TARGET_ITERATIONS}}|$target_iters|g; s|{{MAX_RUN_WALLCLOCK}}|$max_wallclock|g; s|{{PCOORD_NDIM}}|$pcoord_ndim|g; s|{{PCOORD_LEN}}|$pcoord_len|g; s|{{NBINS}}|$nbins|g; s|{{BIN_TARGET_COUNTS}}|$bin_target|g; s|{{NUM_GPUS}}|$gpus|g; s|{{GPU_PRECISION}}|$gpu_prec|g; s|{{FF_0}}|$ff_0|g; s|{{FF_1}}|$ff_1|g; s|{{TEMPERATURE}}|$temperature|g; s|{{TIMESTEP}}|$timestep|g; s|{{FRICTION}}|$friction|g; s|{{PRESSURE}}|$pressure|g; s|{{BAROSTAT_INTERVAL}}|$barostat_int|g; s|{{CONSTRAINT_TOLERANCE}}|$const_tol|g; s|{{HYDROGEN_MASS}}|$hmass|g; s|{{STEPS}}|$steps|g; s|{{SAVE_STEPS}}|$save_steps|g" \
    "$SCRIPT_DIR/westpa_template/west.cfg.template" > "${pdb_id}_WP/west.cfg"

  sed "s|{{PDB_ID}}|$pdb_id|g" "$SCRIPT_DIR/westpa_template/b.txt.template" > "${pdb_id}_WP/b.txt"
  cp "$SCRIPT_DIR/westpa_template/openmm_explicit_rmsd_p_ca_propagator.py" "${pdb_id}_WP/"
  sed "s|{{REPO_DIR}}|$SCRIPT_DIR|g" "$SCRIPT_DIR/westpa_template/env.sh" > "${pdb_id}_WP/env.sh"

  cd "${pdb_id}_WP"
  chmod +x env.sh
  source env.sh

  w_init --bstate-file b.txt >/dev/null 2>&1
  local winit_rc=$?
  if [[ $winit_rc -ne 0 ]]; then
    rm -rf traj_segs west.h5
    w_init --bstate-file b.txt >/dev/null 2>&1
    winit_rc=$?
    if [[ $winit_rc -ne 0 ]]; then
      [[ -n "${SETUP_STATUS_ONLY:-}" ]] && echo "FAIL:w_init"
      cd ..
      rm -rf "${pdb_id}_WP"
      exit "$winit_rc"
    fi
  fi
  cd ..
  [[ -n "${SETUP_STATUS_ONLY:-}" ]] && echo "OK"
}

run_cmd() {
  local target_iterations
  target_iterations=$(cfg westpa.target_iterations)
  resolve_project_out_dirs
  cd "$OUT_DIR"

  local check=0 running=0 submitted=0 errors=0
  local pdbid iterations

  echo "Scanning *_WP directories in $OUT_DIR"
  for pdbid in *_WP; do
    if [[ ! -d "$pdbid" ]]; then
      continue
    fi

    if [[ ! -s "$pdbid/west.h5" ]]; then
      echo "ERROR: $pdbid missing west.h5"
      ((errors++))
      continue
    fi

    if [[ ! -f "$pdbid/run.slurm" ]]; then
      echo "ERROR: $pdbid missing run.slurm"
      ((errors++))
      continue
    fi

    if ! iterations=$(h5ls "$pdbid/west.h5/iterations" | awk '/^iter_/ { split($1, a, "_"); v=a[2] } END { if (v) { printf "%09d", v; exit 0 } else { exit 1 } }'); then
      echo "ERROR: $pdbid unable to read iteration count"
      ((errors++))
      continue
    fi

    if [[ "$iterations" -ge "$target_iterations" ]]; then
      echo "DONE: $pdbid iterations=$iterations target=$target_iterations"
      ((check++))
    elif squeue -u "$USER" | rg -i "$pdbid" >/dev/null 2>&1; then
      echo "RUNNING: $pdbid iterations=$iterations"
      ((running++))
    else
      (cd "$pdbid" && sbatch run.slurm >/dev/null)
      echo "SUBMITTED: $pdbid iterations=$iterations"
      ((submitted++))
    fi
  done

  echo "Summary: done=$check running=$running submitted=$submitted errors=$errors"
}

batch_cmd() {
  resolve_project_out_dirs

  if [[ ! -s pdb_ids.json ]]; then
    echo "ERROR: pdb_ids.json missing or empty"
    exit 1
  fi

  local pdbids=()
  mapfile -t pdbids < <(tr -d '[]"' < pdb_ids.json | tr ',' '\n' | sed 's/^ *//;s/ *$//' | grep -vxFf <(ls -d "$OUT_DIR"/*_WP 2>/dev/null | sed 's|.*/||;s/_WP$//'))

  if [[ ${#pdbids[@]} -eq 0 ]]; then
    echo "No new PDB IDs to set up; running monitor step."
    run_cmd
    return
  fi

  local ok=0 failed=0
  local pdbid status
  for pdbid in "${pdbids[@]}"; do
    status=$(SETUP_STATUS_ONLY=1 "$SCRIPT_DIR/run.sh" setup "$pdbid" 2>/dev/null | tail -n1 || true)
    if [[ "$status" == "OK" ]]; then
      echo "SETUP OK: $pdbid"
      ((ok++))
    else
      echo "SETUP FAIL: $pdbid status=${status:-unknown}"
      ((failed++))
    fi
  done

  echo "Batch setup summary: ok=$ok failed=$failed"
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
  echo "Starting HelixNet UI..."
  exec "$venv_dir/bin/streamlit" run benchmark.py "$@"
}

demo_cmd() {
  local pdb_id="1L2Y"
  echo "HelixNet Smoke Test"
  cd "$SCRIPT_DIR"

  if [[ -d "${pdb_id}_WP" ]]; then
    rm -rf "${pdb_id}_WP"
  fi

  echo "Step 1: preprocess $pdb_id"
  python3 "$BENCHMARK" preprocess "$pdb_id" > /tmp/helixnet_preprocess.log 2>&1
  local prep_rc=$?
  if [[ $prep_rc -ne 0 ]]; then
    echo "FAIL: preprocess exited $prep_rc"
    sed -n '1,200p' /tmp/helixnet_preprocess.log
    exit 1
  fi

  echo "Step 2: verify directories and files"
  [[ -d "${pdb_id}_WP/raw" ]] || { echo "FAIL: missing raw dir"; exit 1; }
  [[ -d "${pdb_id}_WP/processed" ]] || { echo "FAIL: missing processed dir"; exit 1; }
  [[ -f "${pdb_id}_WP/raw/${pdb_id}.pdb" ]] || { echo "FAIL: missing raw pdb"; exit 1; }
  [[ -f "${pdb_id}_WP/processed/${pdb_id}_processed.pdb" ]] || { echo "FAIL: missing processed pdb"; exit 1; }
  [[ -f "${pdb_id}_WP/processed/forcefield.json" ]] || { echo "FAIL: missing forcefield.json"; exit 1; }

  echo "Step 3: template expansion check"
  local test_cfg="/tmp/helixnet_test_west.cfg"
  sed "s/{{PDB_ID}}/${pdb_id}/g" westpa_template/west.cfg.template > "$test_cfg"
  rg "topology_path.*${pdb_id}_WP" "$test_cfg" >/dev/null || { echo "FAIL: template substitution incorrect"; exit 1; }

  rm -rf "${pdb_id}_WP"
  rm -f "$test_cfg" /tmp/helixnet_preprocess.log
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
