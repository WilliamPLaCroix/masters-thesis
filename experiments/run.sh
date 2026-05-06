#!/bin/sh
. /nethome/wlacroix/miniconda3/bin/activate thesis
cd /nethome/wlacroix/Thesis-Project/
export WANDB_CACHE_DIR=/scratch/wlacroix/.cache/wandb/
export HF_HOME=/scratch/wlacroix/.cache/huggingface/

python3 helper.py --helper b