#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: test_typecheck.sh [-h|--help]

Test source code typing consistency with mypy

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

# define function
test_typecheck() {
  find src -type f -name "*.py" | sed 's|/|.|g; s|\.py||g' | xargs -t -I {} mypy -m {}
}

# execute function
check_help "$@"
test_typecheck
