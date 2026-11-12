#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
GPU=${1:-0}
cd "$P"; mkdir -p logs
if tmux has-session -t bp05-setfit 2>/dev/null; then echo "session bp05-setfit exists"; exit 1; fi
tmux new-session -d -s bp05-setfit tcsh
tmux send-keys -t bp05-setfit "cd $P" Enter
tmux send-keys -t bp05-setfit "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES $GPU && python -m bp05.runners.setfit --run-name setfit |& tee -a logs/setfit_run.log && python -m bp05.analyze_setfit --run-name setfit |& tee -a logs/setfit_analyze.log && echo SETFIT-DONE" Enter
echo "started bp05-setfit on GPU $GPU"
