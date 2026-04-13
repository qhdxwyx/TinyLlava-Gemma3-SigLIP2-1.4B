#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FACTORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${FACTORY_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_MODEL_PATH="${FACTORY_ROOT}/checkpoints/llava_factory/tiny-llava-gemma-3-1b-pt-siglip2-so400m-patch14-384-gemma3-1b-pt_base-finetune"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
CONV_MODE="${CONV_MODE:-gemma3}"
IMAGE_FILE="${IMAGE_FILE:-}"
QUERY="${QUERY:-Describe the image in detail.}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

if [[ -z "${IMAGE_FILE}" ]]; then
    echo "Set IMAGE_FILE to the image you want to test."
    exit 1
fi

EXTRA_ARGS=()
if [[ "${LOAD_4BIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--load-4bit)
fi
if [[ "${LOAD_8BIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--load-8bit)
fi

cd "${FACTORY_ROOT}"
"${PYTHON_BIN}" tinyllava/serve/cli.py \
    --model-path "${MODEL_PATH}" \
    --image-file "${IMAGE_FILE}" \
    --conv-mode "${CONV_MODE}" \
    --temperature "${TEMPERATURE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    "${EXTRA_ARGS[@]}"
