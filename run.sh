#!/bin/bash
# Run one experiment.
#   SEED=<int>           sets train.py's seed (default 42 if unset)
#   TIME_BUDGET_S=<sec>  sets the fixed wall-clock training budget (default 90)
# Example:
#   SEED=0 bash run.sh > run.log 2>&1
set -euo pipefail
export SEED="${SEED:-42}"
export TIME_BUDGET_S="${TIME_BUDGET_S:-90}"
torchrun --standalone --nproc_per_node=8 train.py
