#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: prepare_models.sh [-h|--help]

Untar and unzip pre-trained SoPa++ and regex model pairs

Optional arguments:
  -h, --help    Show this help message and exit
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

# untar and unzip downloaded pre-trained models
prepare_models() {
  local directory="./models"
  tar -zxvf "$directory/models.tar.gz" -C "$directory"
}

# execute all functions
check_help "$@"
prepare_models
