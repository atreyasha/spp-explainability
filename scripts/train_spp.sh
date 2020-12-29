#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: train_spp.sh [-h|--help]
Execute single SoPa++ training run using repository defaults

Optional arguments:
  -h, --help    Show this help message and exit
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
train_spp() {
  local model_log_directory session_logfile
  model_log_directory="./models/spp_single_train_$(date +%s)"
  session_logfile="$model_log_directory/session.log"
  mkdir -p "$model_log_directory"

  python3 -m src.train_spp \
    --train-data "./data/facebook_multiclass_nlu/clean/train.uncased.data" \
    --train-labels "./data/facebook_multiclass_nlu/clean/train.labels" \
    --valid-data "./data/facebook_multiclass_nlu/clean/valid.uncased.data" \
    --valid-labels "./data/facebook_multiclass_nlu/clean/valid.labels" \
    --model-log-directory "$model_log_directory" | tee -a "$session_logfile"
}

# execute function
check_help "$@"
train_spp
