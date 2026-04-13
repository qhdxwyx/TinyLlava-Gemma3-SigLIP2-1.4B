#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/.." && pwd)"

# Default to a single chunk on Windows/local machines. Override with CHUNKS_OVERRIDE if needed.
CHUNKS="${CHUNKS_OVERRIDE:-1}"
SPLIT="${SPLIT_OVERRIDE:-llava_vqav2_mscoco_test-dev2015}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/llava_factory/tiny-llava-gemma-3-1b-pt-siglip2-so400m-patch14-384-gemma3_pretrain_dual5090_v1-finetune}"
MODEL_NAME="${MODEL_NAME:-GemmaLlava_Gemma3_1B_SigLIP2}"
CONV_MODE="${CONV_MODE:-gemma3}"
EVAL_DIR="${EVAL_DIR:-${WORKSPACE_ROOT}/Dataset/eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QUESTION_FILE_OVERRIDE="${QUESTION_FILE_OVERRIDE:-}"
IMAGE_FOLDER_OVERRIDE="${IMAGE_FOLDER_OVERRIDE:-}"
SKIP_SUBMISSION_CONVERT="${SKIP_SUBMISSION_CONVERT:-0}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra AVAILABLE_GPUS <<< "$gpu_list"
if [[ "${CHUNKS}" -gt "${#AVAILABLE_GPUS[@]}" ]]; then
    CHUNKS="${#AVAILABLE_GPUS[@]}"
fi
if [[ "${CHUNKS}" -lt 1 ]]; then
    CHUNKS=1
fi

GPULIST=()
for ((i=0; i<CHUNKS; i++)); do
    GPULIST+=("${AVAILABLE_GPUS[$i]}")
done

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "CONV_MODE=${CONV_MODE}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "SPLIT=${SPLIT}"
echo "CHUNKS=${CHUNKS}"
echo "GPU list: ${GPULIST[*]}"

question_file="${QUESTION_FILE_OVERRIDE:-${EVAL_DIR}/vqav2/${SPLIT}.jsonl}"
image_folder="${IMAGE_FOLDER_OVERRIDE:-${EVAL_DIR}/vqav2/test2015}"
answer_dir="${EVAL_DIR}/vqav2/answers/${SPLIT}/${MODEL_NAME}"
mkdir -p "${answer_dir}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" "${PYTHON_BIN}" -m tinyllava.eval.model_vqa_loader \
        --model-path "${MODEL_PATH}" \
        --question-file "${question_file}" \
        --image-folder "${image_folder}" \
        --answers-file "${answer_dir}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "${CHUNKS}" \
        --chunk-idx "${IDX}" \
        --conv-mode "${CONV_MODE}" \
        --temperature 0 \
        --max_new_tokens 128 &
done

wait

output_file="${answer_dir}/merge.jsonl"
> "${output_file}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${answer_dir}/${CHUNKS}_${IDX}.jsonl" >> "${output_file}"
done

echo "Merged output: ${output_file}"

if [[ "${SKIP_SUBMISSION_CONVERT}" == "1" ]]; then
    echo "Skipping submission conversion because SKIP_SUBMISSION_CONVERT=1"
else
    "${PYTHON_BIN}" scripts/convert_vqav2_for_submission.py \
        --split "${SPLIT}" \
        --ckpt "${MODEL_NAME}" \
        --dir "${EVAL_DIR}/vqav2"
    echo "Submission file: ${EVAL_DIR}/vqav2/answers_upload/${SPLIT}/${MODEL_NAME}.json"
fi
