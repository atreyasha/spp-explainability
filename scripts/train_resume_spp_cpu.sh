#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: train_resume_spp_cpu.sh [-h|--help] model_log_directory

Resume single SoPa++ model training run with previously-used defaults
on the CPU

Optional arguments:
  -h, --help                      Show this help message and exit

Required arguments:
  model_log_directory <dir_path>  Path to model log directory where previously saved
                                  model checkpoints are located
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
train_resume_spp_cpu() {
  local model_log_directory
  model_log_directory="$1"
  python3 -m src.train_resume_spp --model-log-directory "$model_log_directory"
}

# execute function
check_help "$@"
train_resume_spp_cpu "$@"
