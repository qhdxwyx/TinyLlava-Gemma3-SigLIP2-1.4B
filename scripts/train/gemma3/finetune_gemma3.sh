#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${CHECKPOINT_ROOT}"
print_resolved_paths
preflight_check
cd "${FACTORY_ROOT}"
build_launch_cmd

"${LAUNCH_CMD[@]}" tinyllava/train/train.py \
    --data_path "${FINETUNE_DATA_PATH}" \
    --image_folder "${FINETUNE_IMAGE_PATH}" \
    --is_multimodal True \
    --conv_version "${CONV_VERSION}" \
    --model_name_or_path "${LLM_VERSION}" \
    --vision_tower "${VT_VERSION}" \
    --vision_tower2 "${VT_VERSION2}" \
    --connector_type "${CN_VERSION}" \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation "${ATTN_IMPLEMENTATION}" \
    --bf16 True \
    --training_recipe "${TRAIN_RECIPE}" \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length True \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --output_dir "${FINETUNE_OUTPUT_DIR}" \
    --num_train_epochs "${FINETUNE_EPOCHS:-1}" \
    --per_device_train_batch_size "${FINETUNE_BATCH_SIZE:-4}" \
    --per_device_eval_batch_size "${FINETUNE_EVAL_BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${FINETUNE_GRAD_ACCUM:-8}" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${FINETUNE_SAVE_STEPS:-10000}" \
    --save_total_limit "${FINETUNE_SAVE_TOTAL_LIMIT:-5}" \
    --learning_rate "${FINETUNE_LR:-1e-4}" \
    --weight_decay 0. \
    --warmup_ratio "${FINETUNE_WARMUP_RATIO:-0.03}" \
    --max_grad_norm 1.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps "${FINETUNE_LOGGING_STEPS:-10}" \
    --tf32 False \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_WORKERS:-8}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    --tokenizer_use_fast False \
    --run_name "${RUN_PREFIX}-finetune"
