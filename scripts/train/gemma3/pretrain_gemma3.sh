#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${CHECKPOINT_ROOT}"
print_resolved_paths
preflight_check
cd "${FACTORY_ROOT}"
build_launch_cmd

PRETRAIN_TUNE_VISION_TOWER_MODE="${PRETRAIN_TUNE_VISION_TOWER:-partially-tune}"
PRETRAIN_VISION_FROM_LAYER="${PRETRAIN_TUNE_VISION_FROM_LAYER:-20}"
PRETRAIN_MM_VISION_SELECT_LAYER="${PRETRAIN_MM_VISION_SELECT_LAYER:--2}"
PRETRAIN_CONNECTOR_LR="${PRETRAIN_CONNECTOR_LR:-${PRETRAIN_LR:-3e-4}}"
PRETRAIN_VISION_TOWER_LR="${PRETRAIN_VISION_TOWER_LR:-4.5e-5}"
if [[ -z "${PRETRAIN_DDP_FIND_UNUSED_PARAMETERS:-}" ]]; then
    if [[ "${NPROC_PER_NODE}" != "1" && "${PRETRAIN_TUNE_VISION_TOWER_MODE}" == "partially-tune" ]]; then
        PRETRAIN_DDP_FIND_UNUSED_PARAMETERS="True"
    else
        PRETRAIN_DDP_FIND_UNUSED_PARAMETERS="False"
    fi
fi

"${LAUNCH_CMD[@]}" tinyllava/train/train.py \
    --data_path "${PRETRAIN_DATA_PATH}" \
    --image_folder "${PRETRAIN_IMAGE_PATH}" \
    --is_multimodal True \
    --conv_version pretrain \
    --model_name_or_path "${LLM_VERSION}" \
    --vision_tower "${VT_VERSION}" \
    --vision_tower2 "${VT_VERSION2}" \
    --connector_type "${CN_VERSION}" \
    --mm_vision_select_layer "${PRETRAIN_MM_VISION_SELECT_LAYER}" \
    --image_aspect_ratio square \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --bf16 True \
    --training_recipe "${TRAIN_RECIPE}" \
    --tune_type_llm frozen \
    --tune_type_vision_tower "${PRETRAIN_TUNE_VISION_TOWER_MODE}" \
    --tune_vision_tower_from_layer "${PRETRAIN_VISION_FROM_LAYER}" \
    --tune_type_connector full \
    --output_dir "${PRETRAIN_OUTPUT_DIR}" \
    --num_train_epochs "${PRETRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PRETRAIN_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size "${PRETRAIN_EVAL_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${PRETRAIN_GRAD_ACCUM:-16}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${PRETRAIN_SAVE_STEPS:-5000}" \
    --save_total_limit "${PRETRAIN_SAVE_TOTAL_LIMIT:-2}" \
    --learning_rate "${PRETRAIN_LR:-3e-4}" \
    --mm_projector_lr "${PRETRAIN_CONNECTOR_LR}" \
    --vision_tower_lr "${PRETRAIN_VISION_TOWER_LR}" \
    --weight_decay 0. \
    --warmup_ratio "${PRETRAIN_WARMUP_RATIO:-0.06}" \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps "${PRETRAIN_LOGGING_STEPS:-10}" \
    --tf32 False \
    --ddp_find_unused_parameters "${PRETRAIN_DDP_FIND_UNUSED_PARAMETERS}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_WORKERS:-8}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    --tokenizer_use_fast False \
    --run_name "${RUN_PREFIX}-pretrain"
