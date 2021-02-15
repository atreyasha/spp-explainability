#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: compare_dual_models_spp_cpu.sh [-h|--help] neural_model_checkpoint regex_model_checkpoint

Compare neural and regex SoPa++ models an evaluation data set on the CPU

Optional arguments:
  -h, --help                           Show this help message and exit

Required arguments:
  neural_model_checkpoint <file_path>  Path to neural model checkpoint with '.pt'
                                       extension
  regex_model_checkpoint  <file_path>  Path to regex model checkpoint with '.pt'
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
compare_dual_models_spp_cpu() {
  local neural_model_checkpoint regex_model_checkpoint
  neural_model_checkpoint="$1"
  regex_model_checkpoint="$2"

  python3 -m src.compare_dual_models_spp \
    --eval-data "./data/facebook_multiclass_nlu/clean/test.uncased.data" \
    --eval-labels "./data/facebook_multiclass_nlu/clean/test.labels" \
    --neural-model-checkpoint "$neural_model_checkpoint" \
    --regex-model-checkpoint "$regex_model_checkpoint"
}

# execute function
check_help "$@"
compare_dual_models_spp_cpu "$@"
