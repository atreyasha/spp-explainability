#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: compare_dual_models_spp_gpu.sh [-h|--help] model_log_directory
                                                  [model_log_directory...]

Compare neural and regex SoPa++ models an evaluation data set on a GPU

Optional arguments:
  -h, --help                      Show this help message and exit

Required arguments:
  model_log_directory <dir_path>  Model log directory which contains
                                  both the best neural and compressed
                                  regex models
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
compare_dual_models_spp_gpu() {
  python3 -m src.compare_dual_models_spp \
    --eval-data "./data/facebook_multiclass_nlu/clean/test.uncased.data" \
    --eval-labels "./data/facebook_multiclass_nlu/clean/test.labels" \
    --model-log-directory "$@" --gpu
}

# execute function
check_help "$@"
compare_dual_models_spp_gpu "$@"
