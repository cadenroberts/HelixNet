#!/bin/bash

JSON_FILE=$1

# Read file, remove characters [, ], quotes, and spaces
pdb_list=$(cat "$JSON_FILE" | tr -d '[]" ' | tr ',' '\n')

# Loop through PDB IDs
for pdb in $pdb_list; do
    echo "Running setup on $pdb..."
    ./setup_wp.sh "$pdb"
done

