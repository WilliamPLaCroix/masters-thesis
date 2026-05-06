#!/usr/bin/env bash
# ---------------- User knobs ----------------
#MERGE_METHOD="${1:?merge method required: debug|svd|linear|ties|ties_svd|dare_ties|dare_linear|dare_ties_svd|dare_linear_svd|magnitude_prune|magnitude_prune_svd}"


DENSITY=None
MAJ_SIGN="total"
ITERATION="15"
# TARGET_GRADE="all"

WINDOW_SIZE="${1:?window size required: all|integer >=1}"
WEIGHT_METHOD="${2:?weight method required: uniform|proximity}"
WEIGHT_BALANCE="${3:?weight balance required: sum|average}"
MERGE_METHOD="${4:?merge method required: dare_ties|linear}"

MODEL_VARIATION="cleaned"              # fixed for baseline runs
PROJECT_VERSION="v3"                 # used in WANDB_PROJECT  
ENTITY=""                              # optional W&B entity

# ---------------- Paths & env ----------------
source /nethome/${USER}/LLaMA-Factory/experiments/scripts/rename_gpus.sh
REPO="/nethome/${USER}/LLaMA-Factory"
BASE_MODEL="/scratch/common_models/Llama-3.2-3B-Instruct-greedy"
CACHE="/scratch/${USER}/.cache/llama_factory"
RUN_KEY="${PROJECT_VERSION}-${MERGE_METHOD}_ws@${WINDOW_SIZE}_w@${WEIGHT_METHOD}-${WEIGHT_BALANCE}-${ITERATION}"
LOG_DIR="${REPO}/experiments/logs/merged"
CFG_DIR="${REPO}/experiments/configs"

mkdir -p "${LOG_DIR}" "${LOG_DIR}/generated_predictions"

# --------------- System info ---------------
source /nethome/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate /nethome/${USER}/miniconda3/envs/llama_factory_v2
cd "$REPO"

# if ! python -c "import bert_score" >/dev/null 2>&1; then
#   python -m pip install -U bert-score
# fi

echo "=== ENV ==="
echo "Conda: $CONDA_DEFAULT_ENV"; which python
nvidia-smi || true; nvcc --version || true

set -euo pipefail

# -------- temp + wandb routing (HPC/HTCondor safe) --------
# Prefer Condor per-job scratch if present; otherwise use your quota-backed /scratch
if [ -n "${_CONDOR_SCRATCH_DIR:-}" ] && [ -d "${_CONDOR_SCRATCH_DIR:-}" ]; then
  export JOB_SCRATCH="${_CONDOR_SCRATCH_DIR}"
else
  # fall back: unique-ish directory on shared scratch
  export JOB_SCRATCH="/scratch/${USER}/condor_tmp/${CLUSTER:-noCluster}.${PROCESS:-noProcess}.$(date +%s)"
  mkdir -p "${JOB_SCRATCH}"
fi

# Route all tempfile users away from /tmp
export TMPDIR="${JOB_SCRATCH}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

# Route caches (optional but often reduces /tmp/home churn)
export HF_HOME="/scratch/${USER}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="/scratch/${USER}/.cache/huggingface/datasets"
export XDG_CACHE_HOME="/scratch/${USER}/.cache"

# Route W&B off /tmp and off /nethome
export WANDB_DIR="${JOB_SCRATCH}/wandb"
export WANDB_CACHE_DIR="/scratch/${USER}/.cache/wandb"
export WANDB_CONFIG_DIR="/scratch/${USER}/.config/wandb"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

echo "[routing] host=$(hostname) JOB_SCRATCH=${JOB_SCRATCH} TMPDIR=${TMPDIR} WANDB_DIR=${WANDB_DIR}"
# ---------------------------------------------------------


echo "Starting adapter merging at $(date)"
total_start_time=$(date +%s)

# # --------------- MERGE ADAPTERS --------------- 
# # Uncomment below to re-run  merging
# echo "Begin Merger"
# python3 experiments/scripts/adapter_merging.py \
#   --model "${BASE_MODEL}" \
#   --density "${DENSITY}" \
#   --majority_sign_method "${MAJ_SIGN}" \
#   --output "${CACHE}" \
#   --window_size "${WINDOW_SIZE}" \
#   --project_version "${PROJECT_VERSION}" \
#   --merge_method "${MERGE_METHOD}" \
#   --target_grade "${TARGET_GRADE}" \
#   --weight_method "${WEIGHT_METHOD}" \
#   --project_version "${PROJECT_VERSION}" \
#   > "${LOG_DIR}/merge@${MERGE_METHOD}_ws@${WINDOW_SIZE}_weight@${WEIGHT_METHOD}_merge.log" 2>&1
# # --------------- MERGE END ---------------

GRADES=(02 03 04 05 06 07 08 09 10 11 12)

for GRADE in "${GRADES[@]}"; do
    TARGET_GRADE="${GRADE}"
    echo "----- Starting grade ${GRADE} -----"
    echo "Begin Merger"
    # ----------- begin per-grade merging -----------
    # --project_version "${PROJECT_VERSION}" \
    python3 experiments/scripts/adapter_merging.py \
      --model "${BASE_MODEL}" \
      --density "${DENSITY}" \
      --majority_sign_method "${MAJ_SIGN}" \
      --output "${CACHE}" \
      --window_size "${WINDOW_SIZE}" \
      --merge_method "${MERGE_METHOD}" \
      --target_grade "${TARGET_GRADE}" \
      --weight_method "${WEIGHT_METHOD}" \
      --weight_balance "${WEIGHT_BALANCE}" \
      --project_version "${PROJECT_VERSION}" \
      > "${LOG_DIR}/merge@${MERGE_METHOD}_grade${TARGET_GRADE}_ws@${WINDOW_SIZE}_weight@${WEIGHT_METHOD}-${WEIGHT_BALANCE}_merge.log" 2>&1
    # ----------- end per-grade merging -----------
    echo "staring run at $(date)"
    run_start_time=$(date +%s)
    
    
    OUT_ADAPTER="${CACHE}/${PROJECT_VERSION}_merge@${MERGE_METHOD}_grade@${TARGET_GRADE}_window@${WINDOW_SIZE}_weight@${WEIGHT_METHOD}-${WEIGHT_BALANCE}"
    # mkdir -p "${OUT_ADAPTER}"

    ID_DIR="/scratch/${USER}/.llf_wandb_ids"
    mkdir -p "${ID_DIR}"
    
    # Shared inference run ID (consistent across all grades)
    INFER_WBRUN_FILE="${ID_DIR}/${RUN_KEY}-infer.id"
    if [[ -f "${INFER_WBRUN_FILE}" ]]; then
      INFER_WANDB_RUN_ID="$(cat "${INFER_WBRUN_FILE}")"
    else
      INFER_WANDB_RUN_ID="$(head -c16 /dev/urandom | od -An -tx1 | tr -d ' \n' | cut -c1-12)"
      echo "${INFER_WANDB_RUN_ID}" > "${INFER_WBRUN_FILE}"
    fi

    # Persist for other scripts and future resumes
    echo "${INFER_WANDB_RUN_ID}" > "${OUT_ADAPTER}/wandb_infer_id.txt"
    echo "Thesis_Phase_${PROJECT_VERSION}" > "${OUT_ADAPTER}/wandb_project.txt"

    # ---------------- Core W&B env ----------------
    export WANDB_PROJECT="Thesis_Phase_${PROJECT_VERSION}.3" ### TODO CHANGE BACK "Thesis_Phase_${PROJECT_VERSION}", we need to hack in the .2 for now
    [[ -n "${ENTITY}" ]] && export WANDB_ENTITY="${ENTITY}"
    export WANDB_RESUME=allow
    export WANDB_ENABLE_SERVICE=true
    export WANDB_HTTP_TIMEOUT=300

    # --------------- INFER (same run; tag infer dataset + grade) ---------------
    # Switch to shared inference W&B config
    export WANDB_RUN_ID="${INFER_WANDB_RUN_ID}"
    export WANDB_RUN_GROUP="merged"
    export WANDB_NAME="${MERGE_METHOD}_ws@${WINDOW_SIZE}_w@${WEIGHT_METHOD}-${WEIGHT_BALANCE}"
    # echo the specific inference arguments
    echo "[infer]   grade: ${GRADE}"
    grade_start_time=$(date +%s)
    DATASET_VARIATION="${MODEL_VARIATION}" # original augmented)
    export TRAIN_VARIANT="${MODEL_VARIATION}"
    export INFER_VARIANT="${DATASET_VARIATION}"
    export WANDB_TAGS="${MODEL_VARIATION},ds:${DATASET_VARIATION},grade:${GRADE}"
    export WANDB_NOTES="infer_ds=${DATASET_VARIATION}; grade=${GRADE}; train_variant=${MODEL_VARIATION}"
    export WANDB_JOB_TYPE="infer"
    export INFER_GRADE="${GRADE}"

    echo "[infer] dataset variation: ${DATASET_VARIATION}"
    echo "[wandb] using project=${WANDB_PROJECT} id=${WANDB_RUN_ID} resume=${WANDB_RESUME}"

    # # -------------- INFERENCE CALL --------------
    # python3 scripts/vllm_infer_metrics.py \
    #     --model_name_or_path "${BASE_MODEL}" \
    #     --adapter_name_or_path "${OUT_ADAPTER}" \
    #     --save_path "${LOG_DIR}/generated_predictions" \
    #     --save_name "${MERGE_METHOD}_a@${TARGET_GRADE}_w@${WEIGHT_METHOD}_grade${GRADE}-infer" \
    #     --template llama3 \
    #     --dataset "${DATASET_VARIATION}_grade${GRADE}_validation" \
    #     --temperature 0 \
    #     --grade "${GRADE}" \
    #     > "${LOG_DIR}/${MERGE_METHOD}_a@${TARGET_GRADE}_w@${WEIGHT_METHOD}_infer_grade${GRADE}.log" 2>&1

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
      --output_dir "${OUT_ADAPTER}" \
      --overwrite_output_dir True \
      --cutoff_len 1024 \
      --seed 42 \
      --per_device_eval_batch_size 32 \
      --bf16 True \
      --predict_with_generate False \
      --do_sample False \
      --report_to wandb \
      --run_name "${WANDB_NAME}" \
      > "${LOG_DIR}/merge@${MERGE_METHOD}_grade${TARGET_GRADE}_ws@${WINDOW_SIZE}_weight@${WEIGHT_METHOD}-${WEIGHT_BALANCE}_eval.log" 2>&1
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