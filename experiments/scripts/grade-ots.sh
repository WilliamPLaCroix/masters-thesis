#!/usr/bin/env bash

# ---------------- User knobs ----------------
PROJECT_VERSION="v3"                 # used in WANDB_PROJECT
ENTITY=""                              # optional W&B entity

# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct-greedy"
CACHE="/scratch/wlacroix/.cache/llama_factory"

LOG_DIR="${REPO}/experiments/logs/original"
CFG_DIR="${REPO}/experiments/configs"

mkdir -p "${LOG_DIR}" "${LOG_DIR}/logs" "${LOG_DIR}/generated_predictions"

# --------------- System info ---------------
source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
cd "$REPO"

# if ! python -c "import bert_score" >/dev/null 2>&1; then
#   python -m pip install -U bert-score
# fi

echo "=== ENV ==="
echo "Conda: $CONDA_DEFAULT_ENV"; which python
nvidia-smi || true; nvcc --version || true

set -euo pipefail

echo "Starting sequential grade processing at $(date)"
total_start_time=$(date +%s)

GRADES=(02 03 04 05 06 07 08 09 10 11 12)
ITERATION_NUM="6"
ITERATION="-${ITERATION_NUM}"
RUN_KEY="graded-from-off-the-shelf${ITERATION_NUM}"

echo "Processing iteration: ${ITERATION_NUM}"
for GRADE in "${GRADES[@]}"; do
    echo "----- Starting grade ${GRADE} -----"
    echo "staring run at $(date)"
    run_start_time=$(date +%s)

    OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_ots_grade${GRADE}-adapter"
    mkdir -p "${OUT_ADAPTER}"

    # ---------------- Separate W&B run IDs for train vs infer ----------------
    ID_DIR="${HOME}/.llf_wandb_ids"
    mkdir -p "${ID_DIR}"
    
    # Shared inference run ID (consistent across all grades)
    INFER_WBRUN_FILE="${ID_DIR}/${RUN_KEY}-infer.id"
    if [[ -f "${INFER_WBRUN_FILE}" ]]; then
      INFER_WANDB_RUN_ID="$(cat "${INFER_WBRUN_FILE}")"
    else
      INFER_WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
      echo "${INFER_WANDB_RUN_ID}" > "${INFER_WBRUN_FILE}"
    fi
    
    # Unique training run ID for each grade
    TRAIN_WBRUN_FILE="${ID_DIR}/${RUN_KEY}-train-grade${GRADE}.id"
    if [[ -f "${TRAIN_WBRUN_FILE}" ]]; then
      TRAIN_WANDB_RUN_ID="$(cat "${TRAIN_WBRUN_FILE}")"
    else
      TRAIN_WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
      echo "${TRAIN_WANDB_RUN_ID}" > "${TRAIN_WBRUN_FILE}"
    fi

    # Persist for other scripts and future resumes
    echo "${INFER_WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_infer_id.txt"
    echo "${TRAIN_WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_train_id.txt"
    echo "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

    # ---------------- Core W&B env ----------------
    export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
    [[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
    export WANDB_DIR="${LOG_DIR}"
    export WANDB_RESUME=allow
    export WANDB_ENABLE_SERVICE=true
    export WANDB_HTTP_TIMEOUT=300

    # --------------- TRAIN ---------------
    # Set training-specific W&B config
    export WANDB_RUN_ID="${TRAIN_WANDB_RUN_ID}"
    export WANDB_RUN_GROUP="ots-graded"
    export WANDB_NAME="grade${GRADE}-from-off-the-shelf"
    export WANDB_TAGS="${GRADE},train"
    export WANDB_JOB_TYPE="train"

    echo "[train] will now run llamafactory-cli train"
    llamafactory-cli train \
      --model_name_or_path "${BASE_MODEL}" \
      --trust_remote_code True \
      --seed 42 \
      --use_fast_tokenizer True \
      --stage sft \
      --do_train True \
      --finetuning_type lora \
      --lora_rank 8 \
      --lora_alpha 16 \
      --lora_target all \
      --lora_dropout 0.05 \
      --cutoff_len 1024 \
      --template llama3 \
      --preprocessing_num_workers 16 \
      --train_on_prompt False \
      --overwrite_cache True \
      --dataset "cleaned_grade${GRADE}_train" \
      --output_dir "${OUT_ADAPTER}" \
      --logging_strategy steps \
      --logging_steps 10 \
      --save_steps 100 \
      --plot_loss True \
      --overwrite_output_dir True \
      --report_to wandb \
      --run_name "grade${GRADE}-from-off-the-shelf" \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 32 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1.0e-5 \
      --num_train_epochs 10 \
      --bf16 True \
      --lr_scheduler_type cosine \
      --do_eval True \
      --eval_dataset "cleaned_grade${GRADE}_validation" \
      --eval_strategy epoch \
      --predict_with_generate False \
      --do_sample False \
      --metric_for_best_model "eval_cleaned_grade${GRADE}_validation_pred-tgt-dFKGL" \
      --save_strategy epoch \
      --save_total_limit 20 \
      --load_best_model_at_end True \
      --greater_is_better False \
      > "${LOG_DIR}/train-ots_grade${GRADE}.log" 2>&1

    # --------------- INFER (same run; tag infer dataset + grade) ---------------
    # Switch to shared inference W&B config
    grade_start_time=$(date +%s)
    DATASET_VARIATION="cleaned" # original augmented)

    export WANDB_RUN_ID="${INFER_WANDB_RUN_ID}"
    export WANDB_RUN_GROUP="ots-graded"
    export WANDB_NAME="graded-from-off-the-shelf"
    export WANDB_TAGS="ds:${DATASET_VARIATION},grade:${GRADE}"
    export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${GRADE};"
    export WANDB_JOB_TYPE="infer"

    export TRAIN_VARIANT="cleaned"
    export INFER_VARIANT="${DATASET_VARIATION}"
    export INFER_GRADE="${GRADE}"

    # echo the specific inference arguments
    echo "[infer]   grade: ${GRADE}"
    
    echo "[infer] dataset variation: ${DATASET_VARIATION}"
    echo "[wandb] using project=${WANDB_PROJECT} id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"

    # -------------- INFERENCE CALL --------------
    llamafactory-cli train \
      --model_name_or_path "${BASE_MODEL}" \
      --adapter_name_or_path "${OUT_ADAPTER}" \
      --trust_remote_code True \
      --template llama3 \
      --do_train False \
      --do_eval True \
      --do_predict False \
      --finetuning_type lora \
      --eval_dataset "cleaned_grade${GRADE}_validation" \
      --output_dir "${LOG_DIR}" \
      --overwrite_output_dir True \
      --cutoff_len 1024 \
      --seed 42 \
      --per_device_eval_batch_size 32 \
      --bf16 True \
      --predict_with_generate False \
      --do_sample False \
      --report_to wandb \
      --run_name "${WANDB_NAME}" \
      > "${LOG_DIR}/graded-ots_grade${GRADE}_eval.log" 2>&1
    # -------------- INFERENCE END --------------

    echo "[infer] completed grade ${GRADE} into run ${WANDB_RUN_ID}"
    grade_end_time=$(date +%s)
    echo "[infer]  ${DATASET_VARIATION} grade ${GRADE} took $((grade_end_time - grade_start_time)) seconds"
    end_time=$(date +%s)
    echo "Run time: $((end_time - run_start_time)) seconds"
done

total_end_time=$(date +%s)
echo "=== ALL GRADES COMPLETED ==="
echo "Total processing time: $((total_end_time - total_start_time)) seconds"