#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    exec "${PYTHON_BIN:-python3}" -m torch.distributed.run \
        --standalone --nproc_per_node "$NPROC_PER_NODE" \
        --module opcredit.training.sft "$@"
fi
exec "${PYTHON_BIN:-python3}" -m opcredit.training.sft "$@"
