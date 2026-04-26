#!/bin/bash
# run_training_unsloth.sh — LSF launcher for train_grpo_unsloth.py
#
# Usage:
#   ./run_training_unsloth.sh <model-name>
#   ./run_training_unsloth.sh unsloth/Qwen3-1.7B
#   ./run_training_unsloth.sh unsloth/Qwen3-4B --lora-rank 32 --save-merged
#
# All arguments after the model name are forwarded verbatim to the Python
# script, so any train_grpo_unsloth.py flag can be passed here:
#   ./run_training_unsloth.sh unsloth/Qwen3-4B --dataset-size 500 --save-merged
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model-name> [extra python args...]"
    echo "  e.g.: $0 unsloth/Qwen3-1.7B"
    echo "  e.g.: $0 unsloth/Qwen3-4B --lora-rank 32 --save-merged"
    exit 1
fi

MODEL_NAME="$1"
shift                             # remaining args forwarded to the python script
EXTRA_ARGS="$*"

MODEL_SAFE=$(echo "$MODEL_NAME" | tr '/:' '--' | tr -cd '[:alnum:]_-')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIMESTAMP}_${MODEL_SAFE}"

OUTPUT_DIR="./outputs/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
UV_BIN=$(which uv)

echo "Model     : ${MODEL_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Job name  : ${RUN_ID}"
echo "Port      : ${MASTER_PORT}"
echo "uv        : ${UV_BIN}"
echo "Extra args: ${EXTRA_ARGS}"

bsub -q normal -M 128 -n 1 \
  -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB" \
  -o "${OUTPUT_DIR}/output.txt" \
  -e "${OUTPUT_DIR}/error.txt" \
  -J "${RUN_ID}" \
  -env "MASTER_PORT=${MASTER_PORT}" \
  "${UV_BIN}" run python train_grpo_unsloth.py \
    --model-id     "${MODEL_NAME}" \
    --dataset-size 200 \
    --lora-rank    32 \
    --output-dir   "${OUTPUT_DIR}" \
    --report-to    "tensorboard" \
    --seed         42 \
    ${EXTRA_ARGS}
