#!/bin/bash

# Output directory prefix
OUTPUT_PATH_PREFIX="./results"

# Function to calculate next run number
calculate_next_run_number() {
    local prefix="$1/run"
    local max_num=-1

    # Check for existing run directories and find the highest number
    for dir in ${prefix}*; do
        if [[ -d "$dir" ]]; then
            # Extract number and compare
            num="${dir#${prefix}}"
            if [[ "$num" =~ ^[0-9]+$ ]] && (( num > max_num )); then
                max_num="$num"
            fi
        fi
    done

    # Calculate next run number (add 1)
    next_num=$((max_num + 1))

    # Format with leading zeros
    printf -v next_run_num "%03d" "$next_num"
    echo "$next_run_num"
}

# Calculate next run number based on existing directories
NEXT_RUN_NUM=$(calculate_next_run_number "$OUTPUT_PATH_PREFIX")

# New run directory with incremented run number
NEW_RUN_DIR="${OUTPUT_PATH_PREFIX}/run${NEXT_RUN_NUM}"

# Base model path
# Uncomment the model you want to use and comment out the others
# MODEL_PATH="${HOME}/finetune/axolotl/output/merged"
# MODEL_PATH="${HOME}/models/opus-v0"
MODEL_PATH="${HOME}/models/Mistral-7B-v0.1"

# Model name extracted from MODEL_PATH for use in output naming
MODEL_NAME=$(basename "${MODEL_PATH}")

# Model arguments
MODEL_ARGS="pretrained=${MODEL_PATH},dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096"

# Ensure the new run directory exists
mkdir -p "${NEW_RUN_DIR}"

CACHE="`pwd`/cache.db"

# Define function to run lm_eval with parameters
run_evaluation() {
    local task_name=$1
    local tasks=$2
    local num_fewshot=$3
    local output_suffix=$4

    # Full output path for this evaluation
    local output_path="${NEW_RUN_DIR}/${MODEL_NAME}_${output_suffix}"

    echo "Running evaluation for ${task_name}..."
    lm_eval --model vllm \
            --model_args "${MODEL_ARGS}" \
            --tasks "${tasks}" \
            --num_fewshot="${num_fewshot}" \
            --batch_size="auto" \
            --output_path="${output_path}" \
	    --limit=0.01  -c ${CACHE} \
	    --write_out \
	    --log_samples
       
}


# Example evaluation call
#run_evaluation "LambadaOpenAI" "lambada_openai" 0 "lambada_openai"
#run_evaluation "ARCeasy" "arc_easy" 25 "arc_easy"
#uun_evaluation "ARC" "arc_challenge" 25 "arc_challenge"
#run_evaluation "HellaSwag" "hellaswag" 10 "hellaswag"
run_evaluation "TruthfulQA" "truthfulqa-mc" 0 "truthfulqa"


MMLU_TASKS="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions "

run_evaluation "MMLU" "${MMLU_TASKS}" 5 "mmlu"
run_evaluation "Winogrande" "winogrande" 5 "winogrande"
run_evaluation "GSM8k" "gsm8k" 5 "gsm8k"

echo "All evaluations completed."
