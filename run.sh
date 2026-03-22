#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK="$SCRIPT_DIR/benchmark.py"
export TERM="${TERM:-xterm}"
tput() {
  command tput "$@" 2>/dev/null || true
}

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

strip_ansi() {
    printf '%s' "$1" | sed -E 's/\x1B\[[0-9;]*[mK]//g'
}

visible_len() {
    local s
    s=$(strip_ansi "$1")
    echo "${#s}"
}

color_char() {
    case "$1" in
    𐄂) tput setaf 1 ;;
    +) tput setaf 2 ;;
    ★) tput setaf 3 ;;
    ✔) tput setaf 4 ;;
    *) tput sgr0 ;;
    esac
    printf "%s" "$1"
    tput sgr0
}

pad() {
  local width="$1" left="$2" sep="$3" right="$4"
  local left_len right_len pad_len
  left_len=$(visible_len "$left")
  right_len=$(visible_len "$right")
  pad_len=$((width - left_len - right_len))
  ((pad_len < 1)) && pad_len=1

  for ((i=0; i<${#left}; i++)); do
    color_char "${left:i:1}"
  done
  printf '%*s' "$pad_len" '' | tr ' ' "$sep"
  for ((i=0; i<${#right}; i++)); do
    color_char "${right:i:1}"
  done
  printf "\n"
}

summary() {
    local name="$1" check="$2" add="$3" star="$4" x="$5"
    local final=${6:-true} chain=${7:-false}
    local right_vis=" $check ✔ $add + $star ★ $x 𐄂"
    local right=" $check $(color_char ✔) $add $(color_char +) $star $(color_char ★) $x $(color_char 𐄂) │"
    local prefix="╞═════════════════════════════╪▶"
    local left_text="$prefix $name SUMMARY "
    while :; do
        local left_vis
        left_vis=$(strip_ansi "$left_text")
        local filler_len=$((80 - ${#left_vis} - ${#right_vis}))
        if (( filler_len >= 1 )); then
            break
        fi
        if [[ "$left_text" == *" $name SUMMARY "* ]]; then
            left_text="$prefix SUMMARY "
        elif [[ "$left_text" == *" SUMMARY "* ]]; then
            left_text="$prefix"
        else
            right_vis="…${right_vis:1}"
            right="${right_vis} │"
        fi
    done
    local left_vis
    left_vis=$(strip_ansi "$left_text")
    local filler_len=$((80 - ${#left_vis} - ${#right_vis}))
    (( filler_len < 0 )) && filler_len=0
    local filler
    filler=$(printf '%*s' "$filler_len" '' | tr ' ' '─')
    if [[ "$chain" == true ]]; then
        pad 80 "│ └────────────────────────────┬" "─" "┤"
    fi
    printf '%s%s%s\n' "$left_text" "$filler" "$right"
    if [[ "$chain" == true ]]; then
        pad 80 "│ ┌────────────────────────────┴───────────────────────────────────────────────┤"
    elif [[ "$final" == true ]]; then
        pad 80 "└" "─" "┴────────────────────────────────────────────────┘"
    else
        pad 80 "├─┬" "─" "┴───────────────────────────────────────────────┤"
    fi
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
      exit $preprocess_rc
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
          exit $winit_rc
      fi
  fi
  cd ..
  [[ -n "${SETUP_STATUS_ONLY:-}" ]] && echo "OK"
}

format_squeue() {
  awk '{
      r = substr($0, 46)
      gsub(/^ +/, "", r)
      if (r ~ /^nid/) r=""
      sub(/^\(/, "", r); sub(/\)$/, "", r)
      slot = (r != "" ? substr(r,1,10) : $3)
      printf "%-8s %-2s %-10s %-2s %-9s %-8s\n", $1,$2,slot,$4,$5,$6
  }'
}

print_diff() {
  while IFS= read -r line; do
      [[ "$2" == "old" && $line =~ ^'< ' ]] && printf '%s%-46s│\n' "│ │                         │ │  " "${line#< }"
      [[ "$2" == "new" && $line =~ ^'> ' ]] && printf '%s%-46s│\n' "│ │                         │ │  " "${line#> }"
  done <<< "$1"
}

run_cmd() {
  local target_iterations
  target_iterations=$(cfg westpa.target_iterations)
  resolve_project_out_dirs
  cd "$OUT_DIR"

  display_error() {
      pad 80  "│𐄂├▶ $1 000000000"  " "  "𐄂 $(printf '%09d' "$target_iterations") │"
  }

  local old_queue
  old_queue=$(squeue -u "$USER" --noheader --format="%.8j %.2t %.10M %.2D %.9P %.8i %R" | format_squeue)
  pad 80 "│ ┌" "─" "┴────────────────────────────────────────────────┤"
  pad 80 "╞═╪▶ $(tput setaf 4)WEST.H5 ITERATION$(tput sgr0) ─── $(tput setaf 4)? REMAINING$(tput sgr0) " "─" "┤"
  pad 80 "│ ├" "─" "┤"
  local check=0 star=0 x=0 plus=0
  local done_list=() error_list=()

  local pdbid
  for pdbid in *_WP; do
      if [ ! -s "$pdbid/west.h5" ]; then
          display_error "$pdbid"
          error_list+=("$pdbid:west.h5")
          ((x++))
      elif [ ! -f "$pdbid/run.slurm" ]; then
          display_error "$pdbid"
          error_list+=("$pdbid:run.slurm")
          ((x++))
      elif ! iterations=$(h5ls "$pdbid/west.h5/iterations" | awk '/^iter_/ { split($1, a, "_"); v=a[2] } END { if (v) { printf "%09d", v; exit 0 } else { exit 1 } }'); then
          display_error "$pdbid"
          error_list+=("$pdbid:iterations")
          ((x++))
      elif [ "$iterations" -ge "$target_iterations" ]; then
          pad 80  "│✔├▶ $pdbid $iterations"  " " "✔ 000000000 │"
          done_list+=("$pdbid:$iterations")
          ((check++))
      else
          if squeue -u "$USER" | grep -qi "$pdbid"; then
              pad 80  "│★├▶ $pdbid $iterations"  " " "★ $(printf '%09d' $((target_iterations - 10#$iterations))) │"
              ((star++))
          else
              (cd "$pdbid" && sbatch run.slurm && cd ..)
              pad 80  "│+├▶ $pdbid $iterations"    " "    "+ $(printf '%09d' $((target_iterations - 10#$iterations))) │"
              ((plus++))
          fi
      fi
  done

  pad 80 "│ ├" "─" "┬─┬────────────────────────────────────────────────┤"
  local new_queue
  new_queue=$(squeue -u "$USER" --noheader --format="%.8j %.2t %.10M %.2D %.9P %.8i %R" | format_squeue)
  local diff_out
  diff_out=$(diff <(echo "$old_queue") <(echo "$new_queue") || true)
  pad 80 "│$(tput setaf 1)𐄂$(tput sgr0)├▶ $(tput setaf 1)SLURM │ WESTPA ERROR$(tput sgr0) ──┤$(tput setaf 3)★$(tput sgr0)├▶ $(tput setaf 3)OLD SUBMISSIONS$(tput sgr0) " "─" "┤"
  pad 80 "│ ├" "─" "┤ ├────────────────────────────────────────────────┤"
  local old_diff=()
  while IFS= read -r line; do old_diff+=("${line#< }"); done < <(echo "$diff_out" | grep '^< ')
  local max=${#error_list[@]}
  ((${#old_diff[@]} > max)) && max=${#old_diff[@]}
  for ((i=0; i<max; i++)); do
      if ((i < ${#error_list[@]})); then
          IFS=: read -r p f <<< "${error_list[i]}"
          left=$(printf '%-12s%9s' "$p" "$f")
      else
          left="                     "
      fi
      right="${old_diff[i]:-}"
      printf '│ │  %s  │ │  %-46s│\n' "$left" "$right"
  done
  pad 80 "│ ├" "─" "┤ ├────────────────────────────────────────────────┤"
  pad 80 "│$(tput setaf 4)✔$(tput sgr0)├▶ $(tput setaf 4)$(printf '%09d' "$target_iterations") ITERS DONE$(tput sgr0) ──┤$(tput setaf 2)+$(tput sgr0)├▶ $(tput setaf 2)NEW SUBMISSIONS$(tput sgr0) " "─" "┤"
  pad 80 "│ ├" "─" "┤ ├────────────────────────────────────────────────┤"
  local new_diff=()
  while IFS= read -r line; do new_diff+=("${line#> }"); done < <(echo "$diff_out" | grep '^> ')
  max=${#done_list[@]}
  ((${#new_diff[@]} > max)) && max=${#new_diff[@]}
  for ((i=0; i<max; i++)); do
      if ((i < ${#done_list[@]})); then
          IFS=: read -r p it <<< "${done_list[i]}"
          left=$(printf '%-12s%9s' "$p" "$it")
      else
          left="                     "
      fi
      right="${new_diff[i]:-}"
      printf '│ │  %s  │ │  %-46s│\n' "$left" "$right"
  done
  pad 80 "│ └" "─" "┘ ├────────────────────────────────────────────────┤"
  summary "RUN" "$check" "$plus" "$star" "$x"
  pad 80 "└" "─" "┴────────────────────────────────────────────────┘"
}

batch_cmd() {
  local target_iterations
  target_iterations=$(cfg westpa.target_iterations)
  resolve_project_out_dirs

  local check=0 x=0
  pad 80 "┌─┬" "─" "┐"
  pad 80 "╞═╪▶ $(tput setaf 4)RUN.SH BATCH$(tput sgr0) " "─" "┤"
  pad 80 "│ ├" "─" "┤"
  if [ ! -s pdb_ids.json ]; then
      pad 80 "│𐄂├▶ pdb_ids.json" " " "𐄂 missing │"
      ((x++))
  else
      pad 80 "│✔├▶ pdb_ids.json" " " "✔ ready │"
      ((check++))
  fi
  if ((x > 0)); then
      pad 80 "└" "─" "┘"
      exit 1
  fi

  local pdbids=()
  mapfile -t pdbids < <(tr -d '[]"' < pdb_ids.json | tr ',' '\n' | sed 's/^ *//;s/ *$//' | grep -vxFf <(ls -d "$OUT_DIR"/*_WP 2>/dev/null | sed 's|.*/||;s/_WP$//'))
  if [ ${#pdbids[@]} -eq 0 ]; then
      pad 80 "│★├▶ No new PDB IDs found" " " "★ all exist │"
      pad 80 "└─┴" "─" "┘"
      run_cmd
      return
  fi

  local setup_keys=() setup_vals=()
  setup_status_set() {
    local k="$1" v="$2"
    local i
    for i in "${!setup_keys[@]}"; do
      if [ "${setup_keys[i]}" = "$k" ]; then setup_vals[i]="$v"; return; fi
    done
    setup_keys+=("$k")
    setup_vals+=("$v")
  }
  setup_status_get() {
    local k="$1"
    local i
    for i in "${!setup_keys[@]}"; do
      if [ "${setup_keys[i]}" = "$k" ]; then printf '%s' "${setup_vals[i]}"; return; fi
    done
    printf ''
  }

  local count_check=0 count_plus=0 count_star=0 count_fail=0
  local pdbid status failstep sts
  for pdbid in "${pdbids[@]}"; do
      status=$(SETUP_STATUS_ONLY=1 "$SCRIPT_DIR/run.sh" setup "$pdbid" 2>/dev/null | tail -n1)
      if [[ "$status" == OK ]]; then
          setup_status_set "$pdbid" "OK"
          ((count_check++))
      elif [[ "$status" == FAIL:* ]]; then
          failstep="${status#FAIL:}"
          setup_status_set "$pdbid" "$failstep"
          ((count_fail++))
      elif [[ "$status" == PARTIAL:* ]]; then
          setup_status_set "$pdbid" "partial"
          ((count_plus++))
      elif [[ "$status" == WARN:* ]]; then
          setup_status_set "$pdbid" "warn"
          ((count_star++))
      else
          setup_status_set "$pdbid" "unknown"
          ((count_fail++))
      fi
  done

  pad 80 "│ └────────────────────────────┬" "─" "┤"
  summary "BATCH" "$count_check" "$count_plus" "$count_star" "$count_fail"
  pad 80 "│ ┌" "─" "┴───────────────────────────────────────────────┤"
  pad 80 "╞═╪▶ $(tput setaf 4)SETUP$(tput sgr0) " "─" "┤"
  pad 80 "│ ├" "─" "┤"
  for pdbid in "${pdbids[@]}"; do
      sts=$(setup_status_get "$pdbid")
      case "$sts" in
          OK) pad 80 "│✔├▶ ${pdbid}_WP" " " "✔ done │" ;;
          partial) pad 80 "│+├▶ $pdbid" " " "+ partial │" ;;
          warn) pad 80 "│★├▶ $pdbid" " " "★ warning │" ;;
          preprocess) pad 80 "│𐄂├▶ $pdbid" " " "𐄂 preprocess │" ;;
          w_init) pad 80 "│𐄂├▶ $pdbid" " " "𐄂 w_init │" ;;
          *) pad 80 "│𐄂├▶ $pdbid" " " "𐄂 unknown │" ;;
      esac
  done
  pad 80 "│ └────────────────────────────┬" "─" "┤"
  summary "SETUP" "$count_check" "$count_plus" "$count_star" "$count_fail"
  pad 80 "├─┬" "─" "┴───────────────────────────────────────────────┤"
  echo ""
  run_cmd
}

ui_cmd() {
  local venv_dir=".venv"
  local requirements="requirements.txt"
  cd "$SCRIPT_DIR"
  if [ ! -d "$venv_dir" ]; then
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
  echo "=== HelixNet Smoke Test ==="
  echo ""
  cd "$SCRIPT_DIR"
  if [ -d "${pdb_id}_WP" ]; then
      echo "Cleaning existing test directory ${pdb_id}_WP"
      rm -rf "${pdb_id}_WP"
  fi
  echo "Step 1: Preprocessing PDB ${pdb_id}..."
  python3 "$BENCHMARK" preprocess "$pdb_id" > /tmp/helixnet_preprocess.log 2>&1
  local prep_rc=$?
  if [ $prep_rc -ne 0 ]; then
      echo "FAIL: Preprocessing failed with exit code $prep_rc"
      echo "--- Preprocessing log ---"
      sed -n '1,200p' /tmp/helixnet_preprocess.log
      echo "-------------------------"
      exit 1
  fi
  echo "  ✓ Preprocessing completed"

  echo "Step 2: Verifying directory structure..."
  [ -d "${pdb_id}_WP/raw" ] || { echo "FAIL: Missing ${pdb_id}_WP/raw directory"; exit 1; }
  [ -d "${pdb_id}_WP/processed" ] || { echo "FAIL: Missing ${pdb_id}_WP/processed directory"; exit 1; }
  echo "  ✓ Directory structure valid"

  echo "Step 3: Verifying raw PDB..."
  [ -f "${pdb_id}_WP/raw/${pdb_id}.pdb" ] || { echo "FAIL: Missing raw PDB file"; exit 1; }
  local raw_atoms
  raw_atoms=$(grep -c "^ATOM" "${pdb_id}_WP/raw/${pdb_id}.pdb" || echo "0")
  [ "$raw_atoms" -ge 100 ] || { echo "FAIL: Raw PDB has too few atoms ($raw_atoms)"; exit 1; }
  echo "  ✓ Raw PDB downloaded ($raw_atoms atoms)"

  echo "Step 4: Verifying processed PDB..."
  [ -f "${pdb_id}_WP/processed/${pdb_id}_processed.pdb" ] || { echo "FAIL: Missing processed PDB file"; exit 1; }
  local proc_atoms
  proc_atoms=$(grep -c "^ATOM" "${pdb_id}_WP/processed/${pdb_id}_processed.pdb" || echo "0")
  [ "$proc_atoms" -ge 100 ] || { echo "FAIL: Processed PDB has too few atoms ($proc_atoms)"; exit 1; }
  if [ "$proc_atoms" -lt 1000 ]; then
      echo "  ! Processed PDB valid but not fully solvated ($proc_atoms atoms)"
  else
      echo "  ✓ Processed PDB valid ($proc_atoms atoms, solvated)"
  fi

  echo "Step 5: Verifying forcefield configuration..."
  [ -f "${pdb_id}_WP/processed/forcefield.json" ] || { echo "FAIL: Missing forcefield.json"; exit 1; }
  local ff_content
  ff_content=$(python3 -c "import json;print(json.load(open('${pdb_id}_WP/processed/forcefield.json')))")
  [[ "$ff_content" == *"amber14-all.xml"* ]] || { echo "FAIL: forcefield missing expected file"; exit 1; }
  echo "  ✓ Forcefield configuration valid"

  echo "Step 6: Testing template expansion..."
  local test_cfg="/tmp/helixnet_test_west.cfg"
  sed "s/{{PDB_ID}}/${pdb_id}/g" westpa_template/west.cfg.template > "$test_cfg"
  grep -q "topology_path.*${pdb_id}_WP" "$test_cfg" || { echo "FAIL: Template substitution incorrect"; exit 1; }
  echo "  ✓ Template expansion works"

  echo "Step 7: Cleaning up test artifacts..."
  rm -rf "${pdb_id}_WP"
  rm -f "$test_cfg" /tmp/helixnet_preprocess.log
  echo "  ✓ Cleanup complete"
  echo ""
  echo "=== Smoke Test Summary ==="
  echo "All checks passed. Core preprocessing and template logic functional."
  echo ""
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
