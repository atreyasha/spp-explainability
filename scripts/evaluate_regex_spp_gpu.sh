#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: evaluate_regex_spp_gpu.sh [-h|--help] model_checkpoint

Evaluate SoPa++ regex model on an evaluation data set on a GPU

Optional arguments:
  -h, --help                    Show this help message and exit

Required arguments:
  model_checkpoint <glob_path>  Glob path to model checkpoint with '.pt'
                                extension
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
evaluate_regex_spp_gpu() {
  local model_checkpoint
  model_checkpoint="$1"

  python3 -m src.evaluate_regex_spp \
    --eval-data "./data/facebook_multiclass_nlu/clean/test.uncased.data" \
    --eval-labels "./data/facebook_multiclass_nlu/clean/test.labels" \
    --model-checkpoint "$model_checkpoint" --gpu
}

# execute function
check_help "$@"
evaluate_regex_spp_gpu "$@"
