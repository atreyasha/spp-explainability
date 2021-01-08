#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: evaluate_spp_cpu.sh [-h|--help] model_checkpoint

Evaluate SoPa++ model on an evaluation data set on the CPU

Optional arguments:
  -h, --help                    Show this help message and exit

Required arguments:
  model_checkpoint <file_path>  Path to model checkpoint with '.pt'
                                extension. Note that 'model_config.json'
                                must be in the same directory level as
                                the model checkpoint file
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
evaluate_spp_cpu() {
  local model_checkpoint
  model_checkpoint="$1"

  python3 -m src.evaluate_spp \
    --eval-data "./data/facebook_multiclass_nlu/clean/test.uncased.data" \
    --eval-labels "./data/facebook_multiclass_nlu/clean/test.labels" \
    --model-checkpoint "$model_checkpoint"
}

# execute function
check_help "$@"
evaluate_spp_cpu "$@"
