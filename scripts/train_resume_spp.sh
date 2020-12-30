#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: train_resume_spp.sh [-h|--help] model_log_directory
Resume single SoPa++ model training run with previously-used defaults

Required arguments:
  --model-log-directory    Path to model log directory where previously saved
                           model checkpoints are located

Optional arguments:
  -h, --help               Show this help message and exit
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 1
    fi
  done
}

# define function
train_resume_spp() {
  local model_log_directory="$1"
  python3 -m src.train_resume_spp --model-log-directory "$model_log_directory"
}

# execute function
check_help "$@"
train_resume_spp "$@"
