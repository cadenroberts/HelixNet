#!/bin/bash
# HelixNet UI launcher - creates venv, installs deps, runs Streamlit

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install -q -r "$REQUIREMENTS"

echo "Starting HelixNet UI..."
exec "$VENV_DIR/bin/streamlit" run app.py "$@"
