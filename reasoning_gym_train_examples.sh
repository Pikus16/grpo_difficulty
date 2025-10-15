#!/bin/bash
# Example commands for training with Reasoning Gym datasets using quantized models

# ============================================================================
# QUICK TEST (5 minutes, verify it works)
# ============================================================================
python src/train_reasoning_gym.py \
    --dataset_config simple_algorithmic \
    --dataset_size 1000 \
    --max_steps 50 \
    --num_generations 4 \
    --batch_size 2

# ============================================================================
# SHORT TRAINING RUN (1-2 hours)
# ============================================================================
python src/train_reasoning_gym.py \
    --dataset_config simple_algorithmic \
    --dataset_size 5000 \
    --max_steps 500 \
    --num_generations 8 \
    --model-name unsloth/Qwen2.5-3B-Instruct-bnb-4bit

# ============================================================================
# FULL TRAINING RUN (4-6 hours)
# ============================================================================
python src/train_reasoning_gym.py \
    --dataset_config algorithmic \
    --dataset_size 20000 \
    --max_steps 1000 \
    --num_generations 8 \
    --batch_size 4 \
    --save_steps 100

# ============================================================================
# CUSTOM CONFIG FILE
# ============================================================================
python src/train_reasoning_gym.py \
    --dataset_config src/reasoning_gym_configs/algorithmic_simple.json \
    --dataset_size 15000 \
    --max_steps 800

# ============================================================================
# DIFFERENT MODEL (Llama 3.1 8B)
# ============================================================================
python src/train_reasoning_gym.py \
    --dataset_config algorithmic \
    --dataset_size 20000 \
    --model-name unsloth/Llama-3.1-8B-Instruct-bnb-4bit \
    --max_steps 1000 \
    --batch_size 2  # Smaller batch for larger model

# ============================================================================
# WITH CUSTOM WANDB PROJECT
# ============================================================================
python src/train_reasoning_gym.py \
    --dataset_config algorithmic \
    --dataset_size 20000 \
    --project my_reasoning_gym_experiments \
    --max_steps 1000

# ============================================================================
# LONGER COMPLETIONS (for complex reasoning)
# ============================================================================
python src/train_reasoning_gym.py \
    --dataset_config algorithmic \
    --dataset_size 20000 \
    --max_completion_length 1024 \
    --max_steps 1000

# ============================================================================
# CREATE YOUR OWN CONFIG
# ============================================================================
# 1. Create a JSON file with your task mix:
cat > my_tasks.json << 'EOF'
{
  "ab": {"weight": 1},
  "caesar_cipher": {
    "weight": 2,
    "config": {"max_words": 10}
  },
  "jugs": {
    "weight": 1,
    "config": {"difficulty": 6}
  }
}
EOF

# 2. Train with it:
python src/train_reasoning_gym.py \
    --dataset_config my_tasks.json \
    --dataset_size 10000

