#!/bin/bash

JSON_FILE=$1
pdb_list=$(cat "$JSON_FILE" | tr -d '[]" ' | tr ',' '\n')
for pdb in $pdb_list; do
    echo "Running setup on $pdb..."
    ./setup_wp.sh "$pdb"
done

