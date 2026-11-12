#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=${1:?usage: launch_tmux.sh prepare|pilot|full|analyze}
cd "$P"; mkdir -p logs

case "$STAGE" in
  pilot) ARGS="--limit 50 --run-name pilot" ;;
  full)  ARGS="--run-name main" ;;
  prepare|analyze) ARGS="" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac

start() {
  local name=$1 cmd=$2
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "!! session $name already exists (attach or kill it: tmux kill-session -t $name)"; return
  fi
  tmux new-session -d -s "$name" tcsh
  tmux send-keys -t "$name" "cd $P" Enter
  tmux send-keys -t "$name" "$cmd" Enter
  echo "started $name"
}

log() { echo "logs/${STAGE}_$1.log"; }

if [[ "$STAGE" == "prepare" ]]; then
  start bp05-prep "source activate_project.csh && python -m bp05.data |& tee -a $(log data) && setenv CUDA_VISIBLE_DEVICES 2 && python -m bp05.embed |& tee -a $(log embed) && python -m bp05.records |& tee -a $(log records) && echo PREPARE-DONE"
  exit 0
fi
if [[ "$STAGE" == "analyze" ]]; then
  start bp05-analyze "source activate_project.csh && python -m bp05.analyze --run-name main |& tee -a $(log analyze) && echo ANALYZE-DONE"
  exit 0
fi

QWEN="setenv VLLM_USE_DEEP_GEMM 0 && python -m bp05.runners.vllm_llm --llm qwen --mode readout"

start bp05-gpu0 "source activate_clef.csh && setenv CUDA_VISIBLE_DEVICES 0 && python -m bp05.runners.clef --variant clef $ARGS |& tee -a $(log gpu0_clef); source activate_project.csh && $QWEN --shard 0,1,2,3,4,5,6,7/32 $ARGS |& tee -a $(log gpu0_qwen_readout); echo GPU0-DONE"

start bp05-gpu1 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 1 && $QWEN --shard 8,9,10,11,12,13,14,15,16,17/32 $ARGS |& tee -a $(log gpu1_qwen_readout); echo GPU1-DONE"

start bp05-gpu2 "source activate_project.csh && setenv CUDA_VISIBLE_DEVICES 2 && python -m bp05.runners.vllm_llm --llm gemma --mode readout $ARGS |& tee -a $(log gpu2_gemma_readout); python -m bp05.runners.baselines --which modernbert $ARGS |& tee -a $(log gpu2_modernbert); $QWEN --shard 18,19,20,21,22,23/32 $ARGS |& tee -a $(log gpu2_qwen_readout); echo GPU2-DONE"

start bp05-gpu3 "source activate_pplx.csh && setenv CUDA_VISIBLE_DEVICES 3 && python -m bp05.runners.pplx $ARGS |& tee -a $(log gpu3_pplx); source activate_project.csh && $QWEN --shard 24,25,26,27,28,29,30,31/32 $ARGS |& tee -a $(log gpu3_qwen_readout); echo GPU3-DONE"

start bp05-cpu "source activate_project.csh && python -m bp05.runners.api --provider jev $ARGS |& tee -a $(log cpu_jev); python -m bp05.runners.baselines --which embedlr $ARGS |& tee -a $(log cpu_embedlr); echo CPU-DONE"
start bp05-pplxapi "source activate_project.csh && python -m bp05.runners.api --provider pplx_api $ARGS |& tee -a $(log cpu_pplx_api); echo PPLXAPI-DONE"

echo "All sessions started."
