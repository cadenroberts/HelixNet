#!/bin/bash
    ✔) tput setaf 4 ;;      # blue
        *) tput sgr0 ;;
    esac
pad() {

# strip ANSI for length calculations (used by summary)
strip_ansi() {
    printf '%s' "$1" | sed -E 's/\x1B\[[0-9;]*[mK]//g'
}

# summary helper as in other scripts
summary() {
    local name="$1" check="$2" add="$3" star="$4" x="$5"
    local final=${6:-true} chain=${7:-false}
    local right_vis=" $check ✔ $add + $star ★ $x 𐄂"
    local right=" $check $(color_char ✔) $add $(color_char +) $star $(color_char ★) $x $(color_char 𐄂) │"
    local prefix="╞══════════════════════════════╪▶"
    local left_text="$prefix $name SUMMARY "
    while :; do
        local left_vis=$(strip_ansi "$left_text")
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
    local left_vis=$(strip_ansi "$left_text")
    local filler_len=$((80 - ${#left_vis} - ${#right_vis}))
    (( filler_len < 0 )) && filler_len=0
    local filler=$(printf '%*s' "$filler_len" '' | tr ' ' '─')
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

if [ -z "$pdb_id" ]; then
    [ -z "$SETUP_STATUS_ONLY" ] && echo "Usage: ./setup_wp.sh <PDB_ID>"
    exit 1
fi

CHECK=0 X=0

pdb_id="$1"
micromamba activate /global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm

./preprocess_pdb.py "$pdb_id"
preprocess_rc=$?
if [ $preprocess_rc -ne 0 ]; then
    if [ -n "$SETUP_STATUS_ONLY" ]; then
        echo "FAIL:preprocess_pdb.py"
    else
        pad 80 "│𐄂├▶ preprocess_pdb.py" " " "𐄂 failed │"
    fi
    ((X++))
    rm -rf "${pdb_id}_WP"
    if [ -z "$SETUP_STATUS_ONLY" ]; then
        pad 80 "│ └────────────────────────────┬" "─" "┤"
        summary "SETUP" "$CHECK" 0 0 "$X"
        pad 80 "└" "─" "┴────────────────────────────────────────────────┘"
    fi
    exit $preprocess_rc
fi
[ -z "$SETUP_STATUS_ONLY" ] && pad 80 "│✔├▶ preprocess_pdb.py" " " "✔ done │"
((CHECK++))
if [ -z "$pdb_id" ]; then
    echo "Usage: ./setup_wp.sh <PDB_ID>"
    exit 1
fi

pad 80 "┌─┬" "─" "┐"
echo "╞═╪▶ $(tput setaf 4)SETUP_WP.SH$(tput sgr0) ──────────────────────────────────────────────────────────────┤"
pad 80 "│ ├" "─" "┤"

CHECK=0 X=0

eval "$(micromamba shell hook --shell bash)"

micromamba activate /global/cfs/cdirs/m4229/caden/micromamba_root/envs/openmm

./preprocess_pdb.py "$pdb_id"
preprocess_rc=$?

if [ $preprocess_rc -ne 0 ]; then
    pad 80 "│𐄂├▶ preprocess_pdb.py" " " "𐄂 failed │"
    ((X++))
    rm -rf "${pdb_id}_WP"
    pad 80 "│ └────────────────────────────┬" "─" "┤"
    echo "╞══════════════════════════════╪▶ $(tput setaf 4)SETUP SUMMARY$(tput sgr0) ────────────────────── $CHECK $(tput setaf 4)✔$(tput sgr0) $X $(tput setaf 1)𐄂$(tput sgr0) │"
    pad 80 "└──────────────────────────────┴" "─" "┘"
    exit $preprocess_rc
fi
pad 80 "│✔├▶ preprocess_pdb.py" " " "✔ done │"
((CHECK++))

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/run.slurm.template > "${pdb_id}_WP/run.slurm"

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/west.cfg.template > "${pdb_id}_WP/west.cfg"

sed "s/{{PDB_ID}}/$pdb_id/g" \
    westpa_template/b.txt.template > "${pdb_id}_WP/b.txt"

pad 80 "│✔├▶ westpa_template" " " "✔ copied │"

cp westpa_template/openmm_explicit_rmsd_p_ca_propagator.py "${pdb_id}_WP/openmm_explicit_rmsd_p_ca_propagator.py"
cp westpa_template/env.sh "${pdb_id}_WP/env.sh"
[ -z "$SETUP_STATUS_ONLY" ] && pad 80 "│✔├▶ westpa_template" " " "✔ copied │"
((CHECK++))

   pad 80 "│ └────────────────────────────┬" "─" "┤"
   summary "SETUP" "$CHECK" 0 0 "$X"
   pad 80 "└" "─" "┴────────────────────────────────────────────────┘"
cd "${pdb_id}_WP"
chmod +x env.sh
source env.sh
w_init --bstate-file b.txt >/dev/null 2>&1
winit_rc=$?
if [ $winit_rc -ne 0 ]; then
    rm -rf "traj_segs"
    rm -f  "west.h5"
    w_init --bstate-file b.txt >/dev/null 2>&1
    winit_rc=$?
    if [ $winit_rc -ne 0 ]; then
        if [ -n "$SETUP_STATUS_ONLY" ]; then
            echo "FAIL:w_init"
        else
            pad 80 "│𐄂├▶ w_init" " " "𐄂 failed │"
        fi
        ((X++))
        cd ..
        rm -rf "${pdb_id}_WP"
        [ -z "$SETUP_STATUS_ONLY" ] && pad 80 "│ └────────────────────────────┬" "─" "┤"
        [ -z "$SETUP_STATUS_ONLY" ] && echo "╞══════════════════════════════╪▶ $(tput setaf 4)SETUP SUMMARY$(tput sgr0) ────────────────────── $CHECK $(tput setaf 4)✔$(tput sgr0) $X $(tput setaf 1)𐄂$(tput sgr0) │"
        [ -z "$SETUP_STATUS_ONLY" ] && pad 80 "└──────────────────────────────┴" "─" "┘"
        exit $winit_rc
    fi
    [ -z "$SETUP_STATUS_ONLY" ] && pad 80 "│★├▶ w_init" " " "★ retry ok │"
else
    [ -z "$SETUP_STATUS_ONLY" ] && pad 80 "│✔├▶ w_init" " " "✔ done │"
fi
((CHECK++))
cd ..
if [ -n "$SETUP_STATUS_ONLY" ]; then
    echo "OK"
else
    pad 80 "│ └────────────────────────────┬" "─" "┤"
    echo "╞══════════════════════════════╪▶ $(tput setaf 4)SETUP SUMMARY$(tput sgr0) ────────────────────── $CHECK $(tput setaf 4)✔$(tput sgr0) $X $(tput setaf 1)𐄂$(tput sgr0) │"
    pad 80 "└──────────────────────────────┴" "─" "┘"
fi
