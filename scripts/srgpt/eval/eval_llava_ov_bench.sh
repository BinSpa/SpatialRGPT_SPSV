#!/bin/bash
# Evaluation script for LLaVA-OV-1.5 series models on SpatialRGPT-Bench
# This script runs evaluation on all 4 LLaVA-OV-1.5 models sequentially
# Usage: bash scripts/srgpt/eval/eval_llava_ov_bench.sh [BATCH_SIZE]

# Batch size (default: 1)
BATCH_SIZE=${1:-1}

# Model paths - update these paths after downloading the models
MODEL_DIR="/root/autodl-fs/models"
CACHE_DIR="/root/autodl-fs/models"

# Define the 4 LLaVA-OV-1.5 model paths (using HuggingFace model names)
# Note: Only 8B model is currently downloaded, others need to be downloaded
declare -a MODEL_NAMES=(
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Base"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Base"
)

# Output folder names
declare -a OUTPUT_NAMES=(
    "llava-ov-1.5-4b-instruct"
    "llava-ov-1.5-8b-instruct"
    "llava-ov-1.5-4b-base"
    "llava-ov-1.5-8b-base"
)

# Update these paths to your actual data locations
ANNOTATION_FILE=/root/autodl-tmp/spsv/SpatialRGPT-Bench/SpatialRGPT-Bench_v1.json
IMAGE_FOLDER=/root/autodl-tmp/spsv/SpatialRGPT-Bench/som_images

# Function to check if model exists
check_model_exists() {
    local model_name=$1
    local model_dir_name=${model_name//\//--}
    local model_path="${CACHE_DIR}/models--${model_dir_name}"
    
    if [ -d "$model_path" ]; then
        return 0
    else
        return 1
    fi
}

# Function to run evaluation for a single model
run_evaluation() {
    local model_name=$1
    local ckpt=$2

    echo "=========================================="
    echo "Evaluating: $ckpt"
    echo "Model: $model_name"
    echo "=========================================="

    # Check if model directory exists (will download if not)
    if check_model_exists "$model_name"; then
        echo "Model found in cache: $model_name"
    else
        echo "Model not found. Will download from HuggingFace: $model_name"
    fi

    # Create output directory
    output_dir="./eval_output/$ckpt/spatial/v1"
    mkdir -p "$output_dir"

    # Run evaluation (single GPU, full benchmark)
    python -m llava.eval.eval_spatial_llava_ov \
        --model-path "$model_name" \
        --cache-dir "$CACHE_DIR" \
        --annotation-file "$ANNOTATION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$output_dir/results.json" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens 128 \
        --temperature 0.0

    echo "Evaluation completed for $ckpt"
    echo "Output file: $output_dir/results.json"
    echo ""
}

# Main execution
echo "Starting LLaVA-OV-1.5 Series Evaluation"
echo "Cache Directory: $CACHE_DIR"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Run evaluation for each model
for i in "${!MODEL_NAMES[@]}"; do
    run_evaluation "${MODEL_NAMES[$i]}" "${OUTPUT_NAMES[$i]}"
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in ./eval_output/<model_name>/spatial/v1/results.json"
echo "=========================================="
