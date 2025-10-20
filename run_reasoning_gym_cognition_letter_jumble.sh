#!/bin/bash

# Configuration
#This is in dpo-experiments
PROJECT_DIR="/home/sa_115331388710787999833/grpo_difficulty"
LOG_DIR="$PROJECT_DIR/experiment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

# Navigate to project
cd "$PROJECT_DIR"

# Activate conda
conda activate grpo

# Log file
MAIN_LOG="$LOG_DIR/all_experiments_${TIMESTAMP}.log"

echo "Starting all experiments at $(date)" | tee -a "$MAIN_LOG"
echo "Logs will be saved to: $LOG_DIR" | tee -a "$MAIN_LOG"

# Define experiments
declare -a DATASETS=("cognition_letter_jumble")
declare -a MODELS=("unsloth/Qwen3-4B-unsloth-bnb-4bit" "unsloth/Qwen3-8B-unsloth-bnb-4bit" "unsloth/phi-4-bnb-4bit")

# Training parameters
PROJECT="GRPO_REASONING_GYM"
NUM_GENERATIONS=8
MAX_STEPS=1000
SAVE_STEPS=100
BETA=0.001
TEST_BATCH_SIZE=64
TEST_NUM_REPEAT=3

# Counter
count=0
total=$((${#DATASETS[@]} * ${#MODELS[@]}))

# Run all combinations
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    count=$((count + 1))
    
    # Create friendly model name for logs
    model_short=$(echo "$model" | sed 's/unsloth\///' | sed 's/-bnb-4bit//')
    log_file="$LOG_DIR/${dataset}_${model_short}_${TIMESTAMP}.log"
    
    echo "" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    echo "Experiment $count/$total" | tee -a "$MAIN_LOG"
    echo "Dataset: $dataset" | tee -a "$MAIN_LOG"
    echo "Model: $model" | tee -a "$MAIN_LOG"
    echo "Started: $(date)" | tee -a "$MAIN_LOG"
    echo "Log file: $log_file" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"
    
    # Run training with all parameters
    python src/train.py \
      --dataset_name "$dataset" \
      --model-name "$model" \
      --project "$PROJECT" \
      --num_generations "$NUM_GENERATIONS" \
      --max_steps "$MAX_STEPS" \
      --save_steps "$SAVE_STEPS" \
      --beta "$BETA" \
      --test_batch_size "$TEST_BATCH_SIZE" \
      --test_num_repeat "$TEST_NUM_REPEAT" \
      --load_4bit \
      2>&1 | tee "$log_file"
    
    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
      echo "✅ SUCCESS: $dataset + $model_short" | tee -a "$MAIN_LOG"
    else
      echo "❌ FAILED: $dataset + $model_short" | tee -a "$MAIN_LOG"
      echo "Check log: $log_file" | tee -a "$MAIN_LOG"
    fi
    
    echo "Completed: $(date)" | tee -a "$MAIN_LOG"
    
    # Clear GPU memory between runs
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    sleep 10
  done
done

echo "" | tee -a "$MAIN_LOG"
echo "🎉 All experiments completed at $(date)" | tee -a "$MAIN_LOG"
echo "Summary log: $MAIN_LOG" | tee -a "$MAIN_LOG"