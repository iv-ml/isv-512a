#!/bin/bash
#SBATCH --job-name=dinet_train
#SBATCH --output=results/slurm_%j_%N_%t.out
#SBATCH --error=results/slurm_%j_%N_%t.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --mem=256G

set -euo pipefail

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID

# Print debug information
echo "=========================================="
echo "Node Information:"
echo "Hostname: $(hostname)"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "=========================================="

# Run training with enroot container
srun --container-image=$ENROOT_IMAGE_PATH/isv-512a.sqsh \
     --container-mounts=$ISV_512A_ROOT:/src \
     --container-writable \
    uv run python scripts/dinet/train.py \
    --config_path ${CONFIG_PATH:-scripts/config/exp1_64_clip.py}