#!/bin/bash

TARGET_ITERS=12500

for d in *_WP; do
    [ -d "$d" ] || continue

    H5="$d/west.h5"
    SLURM="$d/run.slurm"

    echo "Checking $d ..."

    if [ ! -f "$H5" ]; then
        echo "  ❌ No west.h5 found — skipping"
        continue
    fi

    if [ ! -f "$SLURM" ]; then
        echo "  ❌ No run.slurm found — skipping"
        continue
    fi

    # Use h5ls to list iterations
    LAST_ITER=$(h5ls "$d/west.h5/iterations" | awk '{print $1}' | sed 's/iter_//' | sort -n | tail -n 1)
    echo $LAST_ITER
    if [ -z "$LAST_ITER" ]; then
        echo "  ❌ Could not parse iterations — skipping"
        continue
    fi

    echo "  → Found last iteration = $LAST_ITER"

    if [ "$LAST_ITER" -lt "$TARGET_ITERS" ]; then
        echo "  ✅ Below $TARGET_ITERS — submitting"
        #(cd "$d" && sbatch run.slurm)
    else
        echo "  ⏭️ Already ≥ $TARGET_ITERS — skipping"
    fi

    echo
done

