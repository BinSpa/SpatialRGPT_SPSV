#!/bin/bash
# Evaluation script for a single LLaVA-OV-1.5 model on SpatialRGPT-Bench
# Usage: bash scripts/srgpt/eval/eval_llava_ov_single.sh <MODEL_PATH> <CKPT> [BATCH_SIZE]

MODEL_PATH=$1
CKPT=$2
BATCH_SIZE=${3:-1}

# Set CUDA_VISIBLE_DEVICES to use only the first GPU for single-model evaluation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Update these paths to your actual data locations
ANNOTATION_FILE=/root/autodl-tmp/spsv/SpatialRGPT-Bench/SpatialRGPT-Bench_v1.json
IMAGE_FOLDER=/root/autodl-tmp/spsv/SpatialRGPT-Bench/som_images
CACHE_DIR=/root/autodl-fs/models

# Create output directory
output_dir="./eval_output/$CKPT"
mkdir -p "$output_dir"

# Run evaluation (single GPU, full benchmark)
python -m llava.eval.eval_spatial_llava_ov \
    --model-path "$MODEL_PATH" \
    --cache-dir "$CACHE_DIR" \
    --annotation-file "$ANNOTATION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$output_dir/results.json" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens 128 \
    --temperature 0.0

echo "Evaluation completed for $CKPT"
echo "Output file: $output_dir/results.json"
