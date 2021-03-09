#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: evaluate_spp_gpu.sh [-h|--help] neural_model_checkpoint

Evaluate SoPa++ model(s) on an evaluation data set on a GPU

Optional arguments:
  -h, --help                           Show this help message and exit

Required arguments:
  neural_model_checkpoint <glob_path>  Glob path to neural model checkpoint(s)
                                       with '.pt' extension
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 0
    fi
  done
}

# define function
evaluate_spp_gpu() {
  local neural_model_checkpoint
  neural_model_checkpoint="$1"

  python3 -m src.evaluate_spp \
    --eval-data "./data/fmtod/clean/test.uncased.data" \
    --eval-labels "./data/fmtod/clean/test.labels" \
    --model-checkpoint "$neural_model_checkpoint" --gpu
}

# execute function
check_help "$@"
evaluate_spp_gpu "$@"
