#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
    grpo) flags=(--no-use-boc --no-use-ftc) ;;
    boc) flags=(--use-boc --no-use-ftc) ;;
    ftc) flags=(--no-use-boc --use-ftc) ;;
    opcredit) flags=(--use-boc --use-ftc) ;;
    -h|--help)
        echo 'Usage: run_ablation.sh {grpo|boc|ftc|opcredit} [training arguments]'
        exit 0 ;;
    *) echo 'Specify grpo, boc, ftc, or opcredit.' >&2; exit 2 ;;
esac
shift
exec "${PYTHON_BIN:-python3}" -m opcredit.training.rl "$@" "${flags[@]}"
