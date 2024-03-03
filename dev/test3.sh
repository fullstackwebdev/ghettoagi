#!/bin/bash

# Current date and time in YYYY-MM-DD_HH-MM-SS format
CURRENT_DATETIME=$(date "+%Y-%m-%d_%H-%M-%S")

# Output directory prefix, including the date and time
OUTPUT_PATH_PREFIX="./results/${CURRENT_DATETIME}"

# Base model path
# Uncomment the model you want to use and comment out the others
# MODEL_PATH="${HOME}/finetune/axolotl/output/merged"
# MODEL_PATH="${HOME}/models/opus-v0"
MODEL_PATH="${HOME}/models/Mistral-7B-v0.1"

# Model name extracted from MODEL_PATH for use in output naming
MODEL_NAME=$(basename "${MODEL_PATH}")

# Model arguments
MODEL_ARGS="pretrained=${MODEL_PATH},dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096"

# Ensure the base output directory exists
mkdir -p "${OUTPUT_PATH_PREFIX}"

# Define function to run lm_eval with parameters
run_evaluation() {
    local task_name=$1
    local tasks=$2
    local num_fewshot=$3
    local output_suffix=$4

    # Full output path for this evaluation
    local output_path="${OUTPUT_PATH_PREFIX}/${MODEL_NAME}_${output_suffix}"

    echo "Running evaluation for ${task_name}..."
    lm_eval --model vllm \
            --model_args "${MODEL_ARGS}" \
            --tasks "${tasks}" \
            --num_fewshot="${num_fewshot}" \
            --batch_size=1 \
            --output_path="${output_path}"
}

# Run evaluations
#lambada_openai
run_evaluation "LambadaOpenAI" "lambada_openai" 0 "lambada_openai"
#run_evaluation "ARCeasy" "arc_easy" 25 "arc_easy"
#uun_evaluation "ARC" "arc_challenge" 25 "arc_challenge"
#run_evaluation "HellaSwag" "hellaswag" 10 "hellaswag"
#run_evaluation "TruthfulQA" "truthfulqa-mc" 0 "truthfulqa"
#run_evaluation "MMLU" "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,...,hendrycksTest-world_religions" 5 "mmlu"
#run_evaluation "Winogrande" "winogrande" 5 "winogrande"
#run_evaluation "GSM8k" "gsm8k" 5 "gsm8k"

echo "All evaluations completed."

