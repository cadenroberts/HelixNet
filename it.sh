#!/bin/bash
for i in *_WP; do echo "Directory: $i"; h5ls "$i/west.h5/iterations" | awk '{print $1}' | sed 's/iter_//' | sort | tail -n 1; done

