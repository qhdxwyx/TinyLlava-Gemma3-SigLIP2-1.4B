#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_local_setup.py"

if [[ "${RUN_PRETRAIN:-1}" == "1" ]]; then
    bash "${SCRIPT_DIR}/pretrain_gemma3.sh"
fi

if [[ "${RUN_FINETUNE:-1}" == "1" ]]; then
    bash "${SCRIPT_DIR}/finetune_gemma3.sh"
fi
