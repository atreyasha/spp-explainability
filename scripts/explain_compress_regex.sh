#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: explain_compress_regex.sh [-h|--help] regex_model_checkpoint

Compress regex model(s) using a simplistic compression algorithm

Optional arguments:
  -h, --help                          Show this help message and exit

Required arguments:
  regex_model_checkpoint <glob_path>  Path to regex model checkpoint(s)
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
explain_compress_regex() {
  local regex_model_checkpoint
  regex_model_checkpoint="$1"

  python3 -m src.explain_compress_regex \
    --regex-model-checkpoint "$regex_model_checkpoint"
}

# execute function
check_help "$@"
explain_compress_regex "$@"
