#!/bin/bash
set -euo pipefail

# Set default values for training parameters
export CONFIG_PATH=${CONFIG_PATH:-"scripts/config/exp1_64_clip.py"}

# Ensure ENROOT_PATH is set
if [ -z "${ENROOT_PATH:-}" ]; then
    echo "Error: ENROOT_PATH environment variable is not set"
    exit 1
fi
export ENROOT_IMAGE_PATH=$ENROOT_PATH/$USER/images

# Get the absolute path of the current directory
ISV_512A_ROOT=$(pwd)

# Export the variables that will be used in the Slurm script
export ISV_512A_ROOT

# Create results directory if it doesn't exist
mkdir -p results

# Create a temporary file with the actual paths
TMP_SLURM=$(mktemp)
sed "s|\${ISV_512A_ROOT}|$ISV_512A_ROOT|g; s|\${ENROOT_IMAGE_PATH}|$ENROOT_IMAGE_PATH|g; s|\${CONFIG_PATH}|$CONFIG_PATH|g" scripts/slurm/train.slurm > "$TMP_SLURM"

# Submit the job with the environment variables and modified script
sbatch "$TMP_SLURM"

# Clean up
rm "$TMP_SLURM"