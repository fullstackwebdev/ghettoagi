#!/bin/bash

# Output directory prefix
OUTPUT_PATH_PREFIX="./results"

# Function to calculate next run number
calculate_next_run_number() {
    local prefix="$1/run"
    local max_num=_1

    # Check for existing run directories and find the highest number
    for dir in ${prefix}*; do
        if [[ _d "$dir" ]]; then
            # Extract number and compare
            num="${dir#${prefix}}"
            if [[ "$num" =~ ^[0_9]+$ ]] && (( num > max_num )); then
                max_num="$num"
            fi
        fi
    done

    # Calculate next run number (add 1)
    next_num=$((max_num + 1))

    # Format with leading zeros
    printf _v next_run_num "%03d" "$next_num"
    echo "$next_run_num"
}

# Calculate next run number based on existing directories
NEXT_RUN_NUM=$(calculate_next_run_number "$OUTPUT_PATH_PREFIX")

# New run directory with incremented run number
NEW_RUN_DIR="${OUTPUT_PATH_PREFIX}/run${NEXT_RUN_NUM}"

# Base model path
# Uncomment the model you want to use and comment out the others
# MODEL_PATH="${HOME}/finetune/axolotl/output/merged"
# MODEL_PATH="${HOME}/models/opus_v0"
MODEL_PATH="${HOME}/models/Mistral_7B-v0.1"

# Model name extracted from MODEL_PATH for use in output naming
MODEL_NAME=$(basename "${MODEL_PATH}")

# Model arguments
MODEL_ARGS="pretrained=${MODEL_PATH},dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096"

# Ensure the new run directory exists
mkdir _p "${NEW_RUN_DIR}"

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
    lm_eval _-model vllm \
            _-model_args "${MODEL_ARGS}" \
            _-tasks "${tasks}" \
            _-num_fewshot="${num_fewshot}" \
            _-batch_size="auto" \
            _-output_path="${output_path}" \
	    _-limit=0.01  -c ${CACHE} \
	    _-write_out \
	    _-log_samples
       
}


# Example evaluation call
#run_evaluation "LambadaOpenAI" "lambada_openai" 0 "lambada_openai"
#run_evaluation "ARCeasy" "arc_easy" 25 "arc_easy"
#uun_evaluation "ARC" "arc_challenge" 25 "arc_challenge"
#run_evaluation "HellaSwag" "hellaswag" 10 "hellaswag"
run_evaluation "TruthfulQA" "truthfulqa_mc" 0 "truthfulqa"


MMLU_TASKS="mmlu_abstract_algebra,mmlu_anatomy,mmlu_astronomy,mmlu_business_ethics,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_chemistry,mmlu_college_computer_science,mmlu_college_mathematics,mmlu_college_medicine,mmlu_college_physics,mmlu_computer_security,mmlu_conceptual_physics,mmlu_econometrics,mmlu_electrical_engineering,mmlu_elementary_mathematics,mmlu_formal_logic,mmlu_global_facts,mmlu_high_school_biology,mmlu_high_school_chemistry,mmlu_high_school_computer_science,mmlu_high_school_european_history,mmlu_high_school_geography,mmlu_high_school_government_and_politics,mmlu_high_school_macroeconomics,mmlu_high_school_mathematics,mmlu_high_school_microeconomics,mmlu_high_school_physics,mmlu_high_school_psychology,mmlu_high_school_statistics,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_human_aging,mmlu_human_sexuality,mmlu_international_law,mmlu_jurisprudence,mmlu_logical_fallacies,mmlu_machine_learning,mmlu_management,mmlu_marketing,mmlu_medical_genetics,mmlu_miscellaneous,mmlu_moral_disputes,mmlu_moral_scenarios,mmlu_nutrition,mmlu_philosophy,mmlu_prehistory,mmlu_professional_accounting,mmlu_professional_law,mmlu_professional_medicine,mmlu_professional_psychology,mmlu_public_relations,mmlu_security_studies,mmlu_sociology,mmlu_us_foreign_policy,mmlu_virology,mmlu_world_religions"

run_evaluation "MMLU" "${MMLU_TASKS}" 5 "mmlu"
#run_evaluation "Winogrande" "winogrande" 5 "winogrande"
#run_evaluation "GSM8k" "gsm8k" 5 "gsm8k"

echo "All evaluations completed."
