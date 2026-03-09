#!/bin/bash
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
    𐄂) tput setaf 1 ;;  # red
    +) tput setaf 2 ;;    # green
    ★) tput setaf 3 ;;      # yellow
    ✔) tput setaf 4 ;;      # blue
        *) tput sgr0 ;;
    esac
    printf "%s" "$1"
    tput sgr0
}
pad() {
  local width="$1" left="$2" sep="$3" right="$4"
  local left_len right_len pad
  left_len=$(visible_len "$left")
  right_len=$(visible_len "$right")
  pad=$((width - left_len - right_len))
  ((pad < 1)) && pad=1

  for ((i=0; i<${#left}; i++)); do
    color_char "${left:i:1}"
  done

  printf '%*s' "$pad" '' | tr ' ' "$sep"

  for ((i=0; i<${#right}; i++)); do
    color_char "${right:i:1}"
  done

  printf "\n"
}

# summary(NAME, CHECK, ADD, STAR, X, final=true)
summary() {
    local NAME="$1" CHECK_VAL="$2" ADD_VAL="$3" STAR_VAL="$4" X_VAL="$5" FINAL="${6:-true}"
    local CHAIN=false
    if [[ "$FINAL" == "--chain" || "$FINAL" == "false" || "$FINAL" == "0" ]]; then CHAIN=true; fi

    local sym_plain="${CHECK_VAL} ✔ ${ADD_VAL} + ${STAR_VAL} ★ ${X_VAL} 𐄂"
    local sym_col="${CHECK_VAL} $(tput setaf 4)✔$(tput sgr0) ${ADD_VAL} $(tput setaf 2)+$(tput sgr0) ${STAR_VAL} $(tput setaf 3)★$(tput sgr0) ${X_VAL} $(tput setaf 1)𐄂$(tput sgr0)"
    strip_ansi() { printf '%s' "$1" | sed -E 's/\x1B\[[0-9;]*[mK]//g'; }
    local base_left="╞══════════════════════════════╪▶ "
    local name_full="${NAME} SUMMARY"
    local base_vis; base_vis=$(strip_ansi "$base_left")
    local name_vis; name_vis=$(strip_ansi "$name_full")
    local sym_vis; sym_vis=$(strip_ansi "$sym_col")
    local base_len=${#base_vis}
    local name_len=${#name_vis}
    local sym_len=${#sym_vis}
    local extra=0
    if [[ "$NAME" == "RUN" ]]; then extra=2; fi
    local dash_len=$((80 - base_len - name_len - sym_len - 4 - extra))
    if (( dash_len < 1 )); then
        name_full="SUMMARY"
        name_vis=$(strip_ansi "$name_full")
        name_len=${#name_vis}
        dash_len=$((80 - base_len - name_len - sym_len - 4 - extra))
    fi
    if (( dash_len < 1 )); then
        name_full=""
        name_len=0
        dash_len=$((80 - base_len - name_len - sym_len - 4 - extra))
    fi
    if (( dash_len < 1 )); then
        local dash_min=1
        dash_len=$dash_min
        local avail_sym=$((80 - base_len - name_len - dash_len - 4 - extra))
        ((avail_sym<1)) && avail_sym=1
        local sp="${sym_plain}"
        local sl=${#sp}
        if (( sl > avail_sym )); then
            local cut=$((sl - avail_sym + 1))
            local tail="${sp:cut}"
            sp="…${tail}"
        fi
        sym_col=$(printf '%s' "$sp" | sed "s/✔/$(tput setaf 4)✔$(tput sgr0)/g; s/+/$(tput setaf 2)+$(tput sgr0)/g; s/★/$(tput setaf 3)★$(tput sgr0)/g; s/𐄂/$(tput setaf 1)𐄂$(tput sgr0)/g")
        sym_vis=$(strip_ansi "$sym_col")
        sym_len=${#sym_vis}
    fi
    local dashes=""
    if (( dash_len > 0 )); then dashes=$(printf '%*s' "$dash_len" '' | tr ' ' '─'); fi
    if $CHAIN; then
        pad 80 "│ ┌" "─" "┴───────────────────────────────────────────────┤"
    fi
    printf '%s%s %s %s │\n' "$base_left" "$name_full" "$dashes" "$sym_col"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$SCRIPT_DIR/read_config.py"
TARGET_ITERATIONS=$(python3 "$CFG" westpa.target_iterations)

PROJECT_DIR=$(python3 "$CFG" paths.project_dir 2>/dev/null || echo "$SCRIPT_DIR")
OUT_DIR_RAW=$(python3 "$CFG" paths.out_dir 2>/dev/null || echo "")
if [[ -z "$OUT_DIR_RAW" ]]; then
    OUT_DIR="$PROJECT_DIR"
elif [[ "$OUT_DIR_RAW" == /* ]]; then
    OUT_DIR="$OUT_DIR_RAW"
else
    OUT_DIR="$PROJECT_DIR/$OUT_DIR_RAW"
fi

CHECK=0 X=0
pad 80 "┌─┬" "─" "┐"
pad 80 "╞═╪▶ $(tput setaf 4)BATCH_WP.SH$(tput sgr0) " "─" "┤"
pad 80 "│ ├" "─" "┤"
if [ ! -s pdb_ids.json ]; then
    pad 80 "│𐄂├▶ pdb_ids.json" " " "𐄂 missing │"
    ((X++))
else
    pad 80 "│✔├▶ pdb_ids.json" " " "✔ ready │"
    ((CHECK++))
fi


if ((X > 0)); then
    pad 80 "└" "─" "┘"
    exit 1
fi


pdbids=($(tr -d '[]"' < pdb_ids.json | tr ',' '\n' | grep -vxFf <(ls -d "$OUT_DIR"/*_WP 2>/dev/null | sed 's|.*/||;s/_WP$//')))

if [ ${#pdbids[@]} -eq 0 ]; then
    pad 80 "│★├▶ No new PDB IDs found" " " "★ all exist │"
    pad 80 "└─┴" "─" "┘"
    (cd "$OUT_DIR" && "$SCRIPT_DIR/run_wp.sh")
    exit 0
fi

# Collect setup results for all PDBs (portable: emulate associative array)
SETUP_KEYS=()
SETUP_VALS=()
setup_status_set() {
    local k="$1" v="$2"
    for i in "${!SETUP_KEYS[@]}"; do
        if [ "${SETUP_KEYS[i]}" = "$k" ]; then SETUP_VALS[i]="$v"; return; fi
    done
    SETUP_KEYS+=("$k")
    SETUP_VALS+=("$v")
}
setup_status_get() {
    local k="$1"
    for i in "${!SETUP_KEYS[@]}"; do
        if [ "${SETUP_KEYS[i]}" = "$k" ]; then printf '%s' "${SETUP_VALS[i]}"; return; fi
    done
    printf ''
}
COUNT_CHECK=0
COUNT_PLUS=0
COUNT_STAR=0
COUNT_FAIL=0
for pdbid in "${pdbids[@]}"; do
    status=$(SETUP_STATUS_ONLY=1 "$SCRIPT_DIR/setup_wp.sh" "$pdbid" 2>/dev/null | tail -n1)
    if [[ "$status" == OK ]]; then
        setup_status_set "$pdbid" "OK"
        ((COUNT_CHECK++))
    elif [[ "$status" == FAIL:* ]]; then
        failstep="${status#FAIL:}"
        setup_status_set "$pdbid" "$failstep"
        ((COUNT_FAIL++))
    elif [[ "$status" == PARTIAL:* ]]; then
        setup_status_set "$pdbid" "partial"
        ((COUNT_PLUS++))
    elif [[ "$status" == WARN:* ]]; then
        setup_status_set "$pdbid" "warn"
        ((COUNT_STAR++))
    else
        setup_status_set "$pdbid" "unknown"
        ((COUNT_FAIL++))
    fi
done

# Print BATCH SUMMARY (before SETUP_WP.SH)
pad 80 "│ └────────────────────────────┬" "─" "┤"
summary "BATCH" "$COUNT_CHECK" "$COUNT_PLUS" "$COUNT_STAR" "$COUNT_FAIL"

# Print SETUP_WP.SH dashboard
pad 80 "│ ┌" "─" "┴───────────────────────────────────────────────┤"
pad 80 "╞═╪▶ $(tput setaf 4)SETUP_WP.SH$(tput sgr0) " "─" "┤"
pad 80 "│ ├" "─" "┤"
for pdbid in "${pdbids[@]}"; do
    sts=$(setup_status_get "$pdbid")
    case "$sts" in
        OK)
            pad 80 "│✔├▶ ${pdbid}_WP" " " "✔ done │"
            ;;
        partial)
            pad 80 "│+├▶ $pdbid" " " "+ partial │"
            ;;
        warn)
            pad 80 "│★├▶ $pdbid" " " "★ warning │"
            ;;
        preprocess_pdb.py)
            pad 80 "│𐄂├▶ $pdbid" " " "𐄂 preprocess_pdb.py │"
            ;;
        westpa_template)
            pad 80 "│𐄂├▶ $pdbid" " " "𐄂 westpa_template │"
            ;;
        w_init)
            pad 80 "│𐄂├▶ $pdbid" " " "𐄂 w_init │"
            ;;
        *)
            pad 80 "│𐄂├▶ $pdbid" " " "𐄂 unknown │"
            ;;
    esac
done
pad 80 "│ └────────────────────────────┬" "─" "┤"
summary "SETUP" "$COUNT_CHECK" "$COUNT_PLUS" "$COUNT_STAR" "$COUNT_FAIL"
pad 80 "├─┬" "─" "┴───────────────────────────────────────────────┤"

echo ""
(cd "$OUT_DIR" && "$SCRIPT_DIR/run_wp.sh")
