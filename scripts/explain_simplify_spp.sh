#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: explain_simplify_spp.sh [-h|--help] neural_model_checkpoint

Explain and simplify SoPa++ model(s) into regex model(s)

Optional arguments:
  -h, --help                           Show this help message and exit

Required arguments:
  neural_model_checkpoint <glob_path>  Path to neural model checkpoint(s)
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
explain_simplify_spp() {
  local neural_model_checkpoint
  neural_model_checkpoint="$1"

  python3 -m src.explain_simplify_spp \
    --train-data "./data/fmtod/clean/train.uncased.data" \
    --train-labels "./data/fmtod/clean/train.labels" \
    --valid-data "./data/fmtod/clean/valid.uncased.data" \
    --valid-labels "./data/fmtod/clean/valid.labels" \
    --neural-model-checkpoint "$neural_model_checkpoint"
}

# execute function
check_help "$@"
explain_simplify_spp "$@"
