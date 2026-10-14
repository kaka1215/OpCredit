#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON_BIN:-python3}" -m opcredit.training.rl "$@"
