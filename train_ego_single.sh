#!/bin/bash
# Simple Training Script for Ego-Hand Classification
# Usage: ./train_ego_single.sh [OPTIONS]

set -e  # Exit on any error

# Default parameters
CONDA_ENV="vit_pose"
CONFIG="configs/ego_hand/ego_classifier_cfg.py"
WORK_DIR="work_dirs/ego_hand_single"
DATA_ROOT="data/ego_hand/"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e, --env ENV_NAME        Conda environment name (default: vit_pose)"
    echo "  -c, --config CONFIG       Config file path (default: configs/ego_hand/ego_classifier_cfg.py)"
    echo "  -w, --work-dir DIR        Work directory (default: work_dirs/ego_hand_single)"
    echo "  -d, --data-root DIR       Data root directory (default: data/ego_hand/)"
    echo "  --amp                     Enable automatic mixed precision"
    echo "  --resume                  Resume training from latest checkpoint"
    echo "  --seed SEED               Random seed (default: 42)"
    echo "  -h, --help                Show this help message"
    exit 1
}

# Parse command line arguments
AMP_FLAG=""
RESUME_FLAG=""
SEED="42"

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -w|--work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        -d|--data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --amp)
            AMP_FLAG="--amp"
            shift
            ;;
        --resume)
            RESUME_FLAG="--resume"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not available. Please install conda or make sure it's in your PATH."
    exit 1
fi

# Check if the specified conda environment exists
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Error: Conda environment '${CONDA_ENV}' not found."
    echo "Available environments:"
    conda env list
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file '$CONFIG' not found."
    exit 1
fi

# Check if data directory exists
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Error: Data directory '$DATA_ROOT' not found."
    exit 1
fi

# Check if required data files exist
if [[ ! -f "${DATA_ROOT}/train.json" ]]; then
    echo "Error: Missing train.json in '$DATA_ROOT'"
    exit 1
fi

if [[ ! -f "${DATA_ROOT}/valid.json" ]]; then
    echo "Error: Missing valid.json in '$DATA_ROOT'"
    exit 1
fi

echo "=== Ego-Hand Classification Training ==="
echo "Conda Environment: $CONDA_ENV"
echo "Config File: $CONFIG"
echo "Work Directory: $WORK_DIR"
echo "Data Root: $DATA_ROOT"
echo "Random Seed: $SEED"
echo "========================================"

# Activate conda environment and run training
echo "Activating conda environment: $CONDA_ENV"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Verify Python environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Check if required packages are installed
echo "Checking required packages..."
python -c "import torch; import mmengine; import mmpretrain; print('✓ All required packages found')" || {
    echo "Error: Missing required packages. Please install MMPretrain and dependencies in the conda environment."
    exit 1
}

# Run the training script
echo "Starting training..."
python tools/train.py \
    "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --cfg-options randomness.seed="$SEED" \
    --cfg-options data_root="$DATA_ROOT" \
    $AMP_FLAG \
    $RESUME_FLAG

echo "=== Training Complete ==="
echo "Results saved to: $WORK_DIR"
echo "Best checkpoint: $(ls -t ${WORK_DIR}/*.pth 2>/dev/null | head -1 || echo 'No checkpoints found')"
