#!/usr/bin/env bash
#SBATCH --job-name=gpt2-c4
#SBATCH --account=ACD115198
#SBATCH --partition=dev
#SBATCH --gres=gpu:1                 
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00             
#SBATCH --output=/work/%u/ML-study-group-lab4/slurm/%x-%j.out
#BATCH --exclude=25a-hgpn001,25ahpgpn-003
set -euo pipefail

JOB_LIMIT_S=${JOB_LIMIT_S:-1800}
STARTUP_S=${STARTUP_S:-120}
TAIL_S=${TAIL_S:-90}

JOB_START=$(date +%s)
log() { echo "[$(date +%H:%M:%S) +$(( $(date +%s) - JOB_START ))s] $*"; }

REPO=/work/$USER/ML-study-group-lab4
DATA=${DATA:-/work/$USER/c4}                 # directory with train.bin / val.bin from data.py
RUN=${RUN:-baseline-$(date +%m%d-%H%M)}
OUT=$REPO/result/$RUN
NGPU=${SLURM_GPUS_ON_NODE:-1}

module purge
module load cuda/13.0

cd "$REPO"
export HF_HOME=/work/$USER/hf-cache
export HF_HUB_OFFLINE=1               # use the cached tokenizer; never try the network
export WANDB_MODE=offline             # log locally; `wandb sync` on the login node afterwards
export WANDB_PROJECT=gpt2-training
export WANDB_DIR=$REPO                # offline runs land in $REPO/wandb/

# training budget = whatever is left of the job after startup and tail
ELAPSED=$(( $(date +%s) - JOB_START ))
TRAIN_MIN=$(awk -v t=$(( JOB_LIMIT_S - ELAPSED - STARTUP_S - TAIL_S )) 'BEGIN { printf "%.2f", t / 60 }')

log "run=$RUN gpus=$NGPU data=$DATA out=$OUT train_budget=${TRAIN_MIN}min"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-sync accelerate launch --num_processes "$NGPU" \
    train.py \
    --train "$DATA" \
    --out "$OUT" \
    --time-limit-min "$TRAIN_MIN" \
    --report-to wandb \
    ${EXTRA_ARGS:-}

log "done: $OUT/final"

