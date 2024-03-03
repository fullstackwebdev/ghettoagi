#!/bin/bash

# Define common parameters
MODEL_ARGS=/home/shazam/finetune/axolotl/output/merged/,dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096
#/home/shazam/models/$1,dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096
BATCH_SIZE=1

# Replace <output_path> with the desired output directory for each task
OUTPUT_PATH_PREFIX="./results"

#MODEL_PATH=${HOME}/models/
MODEL_PATH=${HOME}/finetune/axolotl/output/merged

#(axo2) (.venv) ➜  finetune ls -lad ~/models/opus-v0/           
#(axo2) (.venv) ➜  finetune ls -lad ~/models/Mistral-7B-v0.1 

MODEL_PATH=${HOME}/models/opus-v0
MODEL_PATH=${HOME}/models/Mistral-7B-v0.1


MODEL_ARGS="${MODEL_PATH}",dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096


#lm_eval --model vllm --model_args pretrained="${MODEL_PATH}",dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096 --tasks arc_challenge  --num_fewshot=25

lm_eval --model vllm --model_args pretrained=${MODEL_ARGS} --tasks arc_challenge  --num_fewshot=25


# ARC Task
#lm_eval --model=${MODEL} \
#               --model_args="${MODEL_ARGS}" \
#               --tasks="arc_challenge" \
#               --num_fewshot=25 \
#               --batch_size=${BATCH_SIZE} \
 #              --output_path="${OUTPUT_PATH_PREFIX}/arc_output"

exit


# HellaSwag Task
lm_eval --model=${MODEL} \
               --model_args="${MODEL_ARGS}" \
               --tasks="hellaswag" \
               --num_fewshot=10 \
               --batch_size=${BATCH_SIZE} \
               --output_path="${OUTPUT_PATH_PREFIX}/hellaswag_output"

# TruthfulQA Task
lm_eval --model=${MODEL} \
               --model_args="${MODEL_ARGS}" \
               --tasks="truthfulqa-mc" \
               --num_fewshot=0 \
               --batch_size=${BATCH_SIZE} \
               --output_path="${OUTPUT_PATH_PREFIX}/truthfulqa_output"

# MMLU Task
MMLU_TASKS="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
lm_eval --model=${MODEL} \
               --model_args="${MODEL_ARGS}" \
               --tasks="${MMLU_TASKS}" \
               --num_fewshot=5 \
               --batch_size=${BATCH_SIZE} \
               --output_path="${OUTPUT_PATH_PREFIX}/mmlu_output"

# Winogrande Task
lm_eval --model=${MODEL} \
               --model_args="${MODEL_ARGS}" \
               --tasks="winogrande" \
               --num_fewshot=5 \
               --batch_size=${BATCH_SIZE} \
               --output_path="${OUTPUT_PATH_PREFIX}/winogrande_output"

# GSM8k Task
lm_eval --model=${MODEL} \
               --model_args="${MODEL_ARGS}" \
               --tasks="
