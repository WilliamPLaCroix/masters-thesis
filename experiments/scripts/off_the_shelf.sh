#!/usr/bin/env bash
# export VLLM_ENABLE_V1_MULTIPROCESSING=0

# ---------------- User knobs ----------------
# MODEL_VARIATION="${1:?model variation required: original|cleaned|augmented}"
PROJECT_VERSION="v3"                 # used in WANDB_PROJECT
ENTITY=""                              # optional W&B entity

# ---------------- Paths & env ----------------
source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/wlacroix/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct-greedy"
CACHE="/scratch/wlacroix/.cache/llama_factory"

LOG_DIR="${REPO}/experiments/logs/cleaned"
MERGED_MODEL="${CACHE}/${PROJECT_VERSION}_cleaned_baseline_merged"
OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_baseline-adapter"
mkdir -p "${OUT_ADAPTER}" "${LOG_DIR}" "${LOG_DIR}/logs" "${LOG_DIR}/generated_predictions"

# An experiment group id to compare the trio {original,cleaned,augmented} together
EXPERIMENT_GROUP="exp-$(date +%Y%m%d-%H%M%S)"

# ---------------- Core W&B env ----------------
export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}"
[[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
export WANDB_DIR="${LOG_DIR}"
export WANDB_RESUME=allow
export WANDB_RUN_GROUP="${EXPERIMENT_GROUP}" # shared across the 3 variants for this run of experiments
export WANDB_TAGS="baseline,cleaned"

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

# for ITERATION_NUM in {97..98}; do
ITERATION_NUM="6"
ITERATION="-${ITERATION_NUM}"
echo "Starting experiment for iteration: ${ITERATION_NUM}"
# RUN_KEY="baseline-${PROJECT_VERSION}${ITERATION}"
RUN_KEY="off_the_shelf-${PROJECT_VERSION}${ITERATION}"
# export WANDB_NAME="baseline${ITERATION}"           # stable name per train variant
export WANDB_NAME="off_the_shelf${ITERATION}"

# ---------------- Stable W&B run id per train variant ----------------
ID_DIR="${HOME}/.llf_wandb_ids"
mkdir -p "${ID_DIR}"
WBRUN_FILE="${ID_DIR}/${RUN_KEY}.id"

if [[ -f "${WBRUN_FILE}" ]]; then
export WANDB_RUN_ID="$(cat "${WBRUN_FILE}")"
else
# short stable id
export WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' 
' | cut -c1-12)"
echo "${WANDB_RUN_ID}" > "${WBRUN_FILE}"
fi

# -------------------- Persist for other scripts and future resumes
printf '%s
' "${WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_parent_id.txt"
printf '%s
' "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

# --------------- TRAIN ---------------
# echo "[train] will now run llamafactory-cli train"
# llamafactory-cli train \
#   --model_name_or_path "${BASE_MODEL}" \
#   --trust_remote_code True \
#   --seed 42 \
#   --use_fast_tokenizer True \
#   --stage sft \
#   --do_train True \
#   --finetuning_type lora \
#   --lora_rank 8 \
#   --lora_alpha 16 \
#   --lora_target all \
#   --lora_dropout 0.05 \
#   --cutoff_len 1024 \
#   --template llama3 \
#   --preprocessing_num_workers 16 \
#   --train_on_prompt False \
#   --overwrite_cache True \
#   --dataset cleaned_baseline_train \
#   --output_dir "${OUT_ADAPTER}" \
#   --logging_strategy steps \
#   --logging_steps 10 \
#   --plot_loss True \
#   --overwrite_output_dir True \
#   --report_to wandb \
#   --run_name baseline \
#   --per_device_train_batch_size 8 \
#   --per_device_eval_batch_size 32 \
#   --gradient_accumulation_steps 4 \
#   --max_grad_norm 0.5 \
#   --learning_rate 1.0e-5 \
#   --num_train_epochs 5 \
#   --bf16 True \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.2 \
#   --do_eval True \
#   --eval_dataset cleaned_baseline_validation \
#   --eval_strategy steps \
#   --eval_steps 1768 \
#   --predict_with_generate False \
#   --do_sample False \
#   --metric_for_best_model eval_cleaned_baseline_validation_pred-tgt-dFKGL \
#   --save_strategy steps \
#   --save_steps 1768 \
#   --save_total_limit 20 \
#   --load_best_model_at_end True \
#   --greater_is_better False \
#   > "${LOG_DIR}/baseline_train.log" 2>&1


# --------------- MERGE ---------------
# echo "Begin Merge"
# llamafactory-cli export \
#   --model_name_or_path /scratch/common_models/Llama-3.2-3B-Instruct-greedy \
#   --adapter_name_or_path /scratch/wlacroix/.cache/llama_factory/${PROJECT_VERSION}_cleaned_baseline-adapter \
#   --trust_remote_code true \
#   --template llama3 \
#   --export_dir ${MERGED_MODEL} \
#   --export_size 5 \
#   --export_device cpu \
#   > "${LOG_DIR}/merge_cleaned_baseline.log" 2>&1

# --------------- single manual eval ---------------
# echo "[train] will now run llamafactory-cli train eval only"
# echo "starting manual eval"
# export WANDB_JOB_TYPE="eval"
# export LF_DUMP_JSONL="${LOG_DIR}/generated_predictions_eval${ITERATION}.jsonl"
# # # --model_name_or_path "${MERGED_MODEL}" \
# llamafactory-cli train \
#   --model_name_or_path /scratch/common_models/Llama-3.2-3B-Instruct-greedy \
#   --adapter_name_or_path "${OUT_ADAPTER}" \
#   --trust_remote_code True \
#   --template llama3 \
#   --do_train False \
#   --do_eval True \
#   --do_predict False \
#   --finetuning_type lora \
#   --eval_dataset cleaned_baseline_validation \
#   --output_dir "${LOG_DIR}" \
#   --overwrite_output_dir True \
#   --cutoff_len 1024 \
#   --seed 42 \
#   --per_device_eval_batch_size 32 \
#   --bf16 True \
#   --predict_with_generate False \
#   --do_sample False \
#   --report_to wandb \
#   --run_name "${WANDB_NAME}" \
#   > "${LOG_DIR}/cleaned_baseline_validation${ITERATION}_eval.log" 2>&1
# echo "[eval] completed eval for iteration ${ITERATION} into run ${WANDB_RUN_ID}"
# --------------- END manual eval ---------------


# --------------- INFER (same run; tag infer dataset + grade) ---------------
export WANDB_JOB_TYPE="infer"

echo "staring run at $(date)"
run_start_time=$(date +%s)
DATASET_VARIATION="cleaned"

for grade in {02..12}; do
    grade_start_time=$(date +%s)
    echo "[infer]   grade: ${grade}"

    # Keep SAME run id as training; do NOT create per-grade runs
    export WANDB_RUN_ID
    export WANDB_RESUME=allow
    # export WANDB_NAME="baseline-graded-eval${ITERATION}"   # keep stable name for color-by-run
    export WANDB_NAME="off_the_shelf-graded-eval${ITERATION}"

    # Rich tags & notes for grouping/filtering in the UI
    export WANDB_TAGS="baseline,cleaned,ds:${DATASET_VARIATION},grade:${grade}"
    export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${grade}; train_variant=cleaned"

    # If your inference script forwards env to W&B config, also export custom hints
    export TRAIN_VARIANT="cleaned"
    export INFER_VARIANT="${DATASET_VARIATION}"
    export INFER_GRADE="${grade}"

    # echo the specific inference arguments


    # Call your inference (must use wandb.init(resume='allow') or respect env id)
    # --adapter_name_or_path "${OUT_ADAPTER}" \
    llamafactory-cli train \
      --model_name_or_path "${BASE_MODEL}" \
      --trust_remote_code True \
      --template llama3 \
      --do_train False \
      --do_eval True \
      --do_predict False \
      --finetuning_type lora \
      --eval_dataset cleaned_grade${grade}_validation \
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
      > "${LOG_DIR}/off_the_shelf_grade${grade}_eval.log" 2>&1
      # > "${LOG_DIR}/baseline_grade${grade}_eval.log" 2>&1
    echo "[eval] completed grade${grade} eval for into run ${WANDB_RUN_ID}"

    echo "[infer] completed grade ${grade} into run ${WANDB_RUN_ID}"
    grade_end_time=$(date +%s)
    echo "[infer]   grade ${grade} took $((grade_end_time - grade_start_time)) seconds"
done

end_time=$(date +%s)
echo "Total infer time: $((end_time - run_start_time)) seconds"
echo "[infer] completed all 3×3×11 calls into run ${WANDB_RUN_ID}"

# echo "Done. Tips in W&B UI:
# • Group by group: ${EXPERIMENT_GROUP} to compare the three runs.
# • Color by run to keep train variants consistent.
# • Filter by tag ds:<dataset> or grade:<n> to slice inference results."
