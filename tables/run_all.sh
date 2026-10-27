#!/usr/bin/env bash
# Write the paper's tables and the data of Figure 2 from the results ../run_metacog_grid.sh wrote.
# No GPU and no images.
#
#   METACOG_RUN_DIR=/path/to/run-root ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

# Unquoted: PYTHON may be a command with arguments.
${PYTHON:-uv run --project .. python} tables_seeds.py
${PYTHON:-uv run --project .. python} figure2.py

echo "outputs in $(pwd)/out"
