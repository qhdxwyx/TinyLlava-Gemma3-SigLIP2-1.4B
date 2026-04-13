#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FACTORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="$(cd -- "${FACTORY_ROOT}/.." && pwd)"
LOCAL_DATA_ROOT="${PROJECT_ROOT}/data"
LOCAL_THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party"
SERVER_DATA_ROOT="/root/autodl-tmp/TinyLLaVA_Factory/data"
SERVER_THIRD_PARTY_ROOT="/root/autodl-tmp/TinyLLaVA_Factory/third_party"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
NODE_RANK="${NODE_RANK:-0}"
if [[ -z "${DATA_ROOT:-}" ]]; then
    if [[ -e "${SERVER_DATA_ROOT}" ]]; then
        DATA_ROOT="${SERVER_DATA_ROOT}"
    else
        DATA_ROOT="${LOCAL_DATA_ROOT}"
    fi
fi
if [[ -z "${THIRD_PARTY_ROOT:-}" ]]; then
    if [[ -e "${SERVER_THIRD_PARTY_ROOT}" ]]; then
        THIRD_PARTY_ROOT="${SERVER_THIRD_PARTY_ROOT}"
    else
        THIRD_PARTY_ROOT="${LOCAL_THIRD_PARTY_ROOT}"
    fi
fi

LLM_VERSION="${LLM_VERSION:-${THIRD_PARTY_ROOT}/LLM-Research/gemma-3-1b-pt}"
VT_VERSION="${VT_VERSION:-${THIRD_PARTY_ROOT}/google/siglip2-so400m-patch14-384}"
VT_VERSION2="${VT_VERSION2:-}"
CN_VERSION="${CN_VERSION:-mlp2x_gelu}"
CONV_VERSION="${CONV_VERSION:-gemma3}"
TRAIN_RECIPE="${TRAIN_RECIPE:-common}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
REPORT_TO="${REPORT_TO:-none}"

PRETRAIN_DATA_PATH="${PRETRAIN_DATA_PATH:-${DATA_ROOT}/text_files/blip_laion_cc_sbu_558k.json}"
PRETRAIN_IMAGE_PATH="${PRETRAIN_IMAGE_PATH:-${DATA_ROOT}/llava/llava_pretrain/images}"
FINETUNE_DATA_PATH="${FINETUNE_DATA_PATH:-${DATA_ROOT}/text_files/llava_v1_5_mix665k_cleaned_data_w_ocr_vqa.json}"
FINETUNE_IMAGE_PATH="${FINETUNE_IMAGE_PATH:-${DATA_ROOT}}"

VERSION="${VERSION:-gemma3-1b-pt_base}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${FACTORY_ROOT}/checkpoints/llava_factory}"
RUN_PREFIX="${RUN_PREFIX:-tiny-llava-$(basename "${LLM_VERSION}")-$(basename "${VT_VERSION}")-${VERSION}}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-${CHECKPOINT_ROOT}/${RUN_PREFIX}-pretrain}"
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${CHECKPOINT_ROOT}/${RUN_PREFIX}-finetune}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${PRETRAIN_OUTPUT_DIR}}"

build_launch_cmd() {
    if [[ "${NPROC_PER_NODE}" == "1" ]]; then
        LAUNCH_CMD=("${PYTHON_BIN}")
    else
        LAUNCH_CMD=(
            "${TORCHRUN_BIN}"
            --nproc_per_node "${NPROC_PER_NODE}"
            --master_port "${MASTER_PORT}"
            --node_rank "${NODE_RANK}"
        )
    fi
}

preflight_check() {
    local missing=0
    for path in \
        "${LLM_VERSION}" \
        "${VT_VERSION}" \
        "${PRETRAIN_DATA_PATH}" \
        "${PRETRAIN_IMAGE_PATH}" \
        "${FINETUNE_DATA_PATH}" \
        "${FINETUNE_IMAGE_PATH}"; do
        if [[ ! -e "${path}" ]]; then
            echo "[missing] ${path}"
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        echo "One or more required paths are missing. Review the variables above or run verify_local_setup.py."
        exit 1
    fi
}

print_resolved_paths() {
    cat <<EOF
[gemma3 setup]
PROJECT_ROOT=${PROJECT_ROOT}
GEMMALLAVA_ROOT=${FACTORY_ROOT}
DATA_ROOT=${DATA_ROOT}
THIRD_PARTY_ROOT=${THIRD_PARTY_ROOT}
LLM_VERSION=${LLM_VERSION}
VT_VERSION=${VT_VERSION}
PRETRAIN_DATA_PATH=${PRETRAIN_DATA_PATH}
PRETRAIN_IMAGE_PATH=${PRETRAIN_IMAGE_PATH}
FINETUNE_DATA_PATH=${FINETUNE_DATA_PATH}
FINETUNE_IMAGE_PATH=${FINETUNE_IMAGE_PATH}
PRETRAIN_OUTPUT_DIR=${PRETRAIN_OUTPUT_DIR}
FINETUNE_OUTPUT_DIR=${FINETUNE_OUTPUT_DIR}
EOF
}
