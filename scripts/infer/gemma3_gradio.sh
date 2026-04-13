#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FACTORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_MODEL_PATH="${FACTORY_ROOT}/checkpoints/llava_factory/tiny-llava-gemma-3-1b-pt-siglip2-so400m-patch14-384-gemma3-1b-pt_base-finetune"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
CONV_MODE="${CONV_MODE:-gemma3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

EXTRA_ARGS=()
if [[ "${LOAD_4BIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--load-4bit)
fi
if [[ "${LOAD_8BIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--load-8bit)
fi

cd "${FACTORY_ROOT}"
"${PYTHON_BIN}" tinyllava/serve/app.py \
    --model-path "${MODEL_PATH}" \
    --conv-mode "${CONV_MODE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "${EXTRA_ARGS[@]}"
