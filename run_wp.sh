#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ITERATIONS=$(python3 "$SCRIPT_DIR/read_config.py" westpa.target_iterations)
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

# strip ANSI escape sequences
strip_ansi() {
    printf '%s' "$1" | sed -E 's/\x1B\[[0-9;]*[mK]//g'
}

summary() {
    local name="$1" check="$2" add="$3" star="$4" x="$5"
    local final=${6:-true} chain=${7:-false}
    local right_vis=" $check ✔ $add + $star ★ $x 𐄂"
    local right=" $check $(color_char ✔) $add $(color_char +) $star $(color_char ★) $x $(color_char 𐄂) │"
    local prefix="╞═════════════════════════════╪▶"
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
print_diff() {
    while IFS= read -r line; do
        [[ "$2" == "old" && $line =~ ^'< ' ]] && printf '%s%-46s│\n' "│ │                         │ │  " "${line#< }"
        [[ "$2" == "new" && $line =~ ^'> ' ]] && printf '%s%-46s│\n' "│ │                         │ │  " "${line#> }"
    done <<< "$1"
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
display_error() {
    pad 80  "│𐄂├▶ $1 000000000"  " "  "𐄂 $(printf '%09d' $TARGET_ITERATIONS) │"
}
OLD_QUEUE=$(squeue -u "$USER" --noheader --format="%.8j %.2t %.10M %.2D %.9P %.8i %R" | format_squeue)

pad 80 "│ ┌" "─" "┴────────────────────────────────────────────────┤"
pad 80 "╞═╪▶ $(tput setaf 4)WEST.H5 ITERATION$(tput sgr0) ─── $(tput setaf 4)? REMAINING$(tput sgr0) " "─" "┤"
pad 80 "│ ├" "─" "┤"
CHECK=0 STAR=0 X=0 PLUS=0
DONE_LIST=() ERROR_LIST=()
for pdbid in *_WP; do
    if [ ! -s "$pdbid/west.h5" ]; then
        display_error "$pdbid"
        ERROR_LIST+=("$pdbid:west.h5")
        ((X++))
    elif [ ! -f "$pdbid/run.slurm" ]; then
        display_error "$pdbid"
        ERROR_LIST+=("$pdbid:run.slurm")
        ((X++))
    elif ! ITERATIONS=$(h5ls "$pdbid/west.h5/iterations" | awk '/^iter_/ { split($1, a, "_"); v=a[2] } END { if (v) { printf "%09d", v; exit 0 } else { exit 1 } }'); then
        display_error "$pdbid"
        ERROR_LIST+=("$pdbid:iterations")
        ((X++))
    elif [ "$ITERATIONS" -ge "$TARGET_ITERATIONS" ]; then
        pad 80  "│✔├▶ $pdbid $ITERATIONS"  " " "✔ 000000000 │"
        DONE_LIST+=("$pdbid:$ITERATIONS")
        ((CHECK++))
    else
        if squeue -u "$USER" | grep -qi "$pdbid"; then
            pad 80  "│★├▶ $pdbid $ITERATIONS"  " " "★ $(printf '%09d' $((TARGET_ITERATIONS - 10#$ITERATIONS))) │"
            ((STAR++))
        else
            (cd "$pdbid" && sbatch run.slurm && cd ..)
            pad 80  "│+├▶ $pdbid $ITERATIONS"    " "    "+ $(printf '%09d' $((TARGET_ITERATIONS - 10#$ITERATIONS))) │"
            ((PLUS++))
        fi
    fi
done
pad 80 "│ ├" "─" "┬─┬────────────────────────────────────────────────┤"
NEW_QUEUE=$(squeue -u "$USER" --noheader --format="%.8j %.2t %.10M %.2D %.9P %.8i %R" | format_squeue)
DIFF=$(diff <(echo "$OLD_QUEUE") <(echo "$NEW_QUEUE"))
pad 80 "│$(tput setaf 1)𐄂$(tput sgr0)├▶ $(tput setaf 1)SLURM │ WESTPA ERROR$(tput sgr0) ──┤$(tput setaf 3)★$(tput sgr0)├▶ $(tput setaf 3)OLD SUBMISSIONS$(tput sgr0) " "─" "┤"
pad 80 "│ ├" "─" "┤ ├────────────────────────────────────────────────┤"
OLD_DIFF=(); while IFS= read -r line; do OLD_DIFF+=("${line#< }"); done < <(echo "$DIFF" | grep '^< ')
max=${#ERROR_LIST[@]}; ((${#OLD_DIFF[@]} > max)) && max=${#OLD_DIFF[@]}
for ((i=0; i<max; i++)); do
    if ((i < ${#ERROR_LIST[@]})); then IFS=: read p f <<< "${ERROR_LIST[i]}"; left=$(printf '%-12s%9s' "$p" "$f"); else left="                     "; fi
    right="${OLD_DIFF[i]:-}"
    printf '│ │  %s  │ │  %-46s│\n' "$left" "$right"
done
pad 80 "│ ├" "─" "┤ ├────────────────────────────────────────────────┤"
pad 80 "│$(tput setaf 4)✔$(tput sgr0)├▶ $(tput setaf 4)$(printf '%09d' $TARGET_ITERATIONS) ITERS DONE$(tput sgr0) ──┤$(tput setaf 2)+$(tput sgr0)├▶ $(tput setaf 2)NEW SUBMISSIONS$(tput sgr0) " "─" "┤"
pad 80 "│ ├" "─" "┤ ├────────────────────────────────────────────────┤"
NEW_DIFF=(); while IFS= read -r line; do NEW_DIFF+=("${line#> }"); done < <(echo "$DIFF" | grep '^> ')
max=${#DONE_LIST[@]}; ((${#NEW_DIFF[@]} > max)) && max=${#NEW_DIFF[@]}
for ((i=0; i<max; i++)); do
    if ((i < ${#DONE_LIST[@]})); then IFS=: read p it <<< "${DONE_LIST[i]}"; left=$(printf '%-12s%9s' "$p" "$it"); else left="                     "; fi
    right="${NEW_DIFF[i]:-}"
    printf '│ │  %s  │ │  %-46s│\n' "$left" "$right"
done
pad 80 "│ └" "─" "┘ ├────────────────────────────────────────────────┤"
summary "RUN" "$CHECK" "$PLUS" "$STAR" "$X"
# bottom border after run summary
pad 80 "└" "─" "┴────────────────────────────────────────────────┘"
