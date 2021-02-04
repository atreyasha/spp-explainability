#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: explain_simplify_spp_cpu.sh [-h|--help] model_checkpoint

Explain and simplify a given SoPa++ model to retrieve highest
activating regular expressions on the CPU

Optional arguments:
  -h, --help                    Show this help message and exit

Required arguments:
  model_checkpoint <file_path>  Path to model checkpoint with '.pt' extension
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
explain_simplify_spp_cpu() {
  local model_checkpoint
  model_checkpoint="$1"

  python3 -m src.explain_simplify_spp \
    --train-data "./data/facebook_multiclass_nlu/clean/train.uncased.data" \
    --train-labels "./data/facebook_multiclass_nlu/clean/train.labels" \
    --valid-data "./data/facebook_multiclass_nlu/clean/valid.uncased.data" \
    --valid-labels "./data/facebook_multiclass_nlu/clean/valid.labels" \
    --model-checkpoint "$model_checkpoint"
}

# execute function
check_help "$@"
explain_simplify_spp_cpu "$@"
