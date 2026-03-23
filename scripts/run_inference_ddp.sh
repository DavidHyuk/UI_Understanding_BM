#!/bin/bash

# Default values
DATASET_NAME="screenspot"
NUM_SAMPLES=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_id)
      MODEL_ID="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --dataset_name NAME  Dataset name (default: screenspot)"
      echo "  --num_samples NUM    Number of samples to evaluate"
      echo "  --output_dir PATH    Output directory"
      exit 0
      ;;
    *)
      # Heuristic for backward compatibility:
      # If it's a number, assume it's num_samples
      if [[ $1 =~ ^[0-9]+$ ]]; then
        NUM_SAMPLES="$1"
        echo "Note: Interpreting '$1' as --num_samples"
      # If it's a known dataset name
      elif [[ "$1" == "screenspot" || "$1" == "sroie" || "$1" == "screenqa" || "$1" == "widget_captioning" ]]; then
        DATASET_NAME="$1"
        echo "Note: Interpreting '$1' as --dataset_name"
      # Otherwise assume it's model_id
      else
        MODEL_ID="$1"
        echo "Note: Interpreting '$1' as --model_id"
      fi
      shift
      ;;
  esac
done

# Extract model name for output directory if not set
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_NAME=$(basename "$MODEL_ID")
    OUTPUT_DIR="results/$MODEL_NAME"
fi

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    echo "nvidia-smi not found. Defaulting to 1 GPU."
    NUM_GPUS=1
fi

echo "Detected $NUM_GPUS GPUs. Starting DDP Inference..."
echo "Configuration:"
echo "  Model:    $MODEL_ID"
echo "  Dataset:  $DATASET_NAME"
echo "  Samples:  ${NUM_SAMPLES:-all}"
echo "  Output:   $OUTPUT_DIR"
echo ""

# Construct arguments
ARGS=(
    --model_id "$MODEL_ID"
    --dataset_name "$DATASET_NAME"
    --output_dir "$OUTPUT_DIR"
    --ddp
)

# Add num_samples only if provided
if [ -n "$NUM_SAMPLES" ]; then
    ARGS+=(--num_samples "$NUM_SAMPLES")
fi

# Run with torchrun
torchrun --nproc_per_node=$NUM_GPUS scripts/inference.py "${ARGS[@]}"
