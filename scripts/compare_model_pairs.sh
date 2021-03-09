#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: compare_model_pairs.sh [-h|--help] model_log_directory

Compare neural SoPa++ and regex model pairs on an evaluation data set

Optional arguments:
  -h, --help                       Show this help message and exit

Required arguments:
  model_log_directory <glob_path>  Model log directory/directories which
                                   contain both the best neural and
                                   compressed regex models
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
compare_model_pairs() {
  local model_log_directory
  model_log_directory="$1"

  python3 -m src.compare_model_pairs \
    --eval-data "./data/fmtod/clean/test.uncased.data" \
    --eval-labels "./data/fmtod/clean/test.labels" \
    --model-log-directory "$model_log_directory"
}

# execute function
check_help "$@"
compare_model_pairs "$@"
