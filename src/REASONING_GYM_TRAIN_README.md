# Training with Reasoning Gym Datasets (Quantized Models + LoRA)

This script (`train_reasoning_gym.py`) allows you to train on Reasoning Gym's procedural datasets while using your existing infrastructure:
- ✅ Unsloth for quantized models (4-bit)
- ✅ LoRA for memory-efficient training
- ✅ GRPO algorithm (same as original train.py)
- ✅ Procedurally generated Reasoning Gym tasks

## Setup

1. **Install Reasoning Gym** (if not already done):
```bash
cd reasoning-gym
pip install -e .
cd ..
```

2. **Verify installation**:
```bash
python -c "import reasoning_gym; print('✓ Reasoning Gym installed')"
```

## Quick Start

### Use Pre-defined Configs

```bash
# Train on algorithmic tasks (10 different task types)
python src/train_reasoning_gym.py \
    --dataset_config algorithmic \
    --dataset_size 20000 \
    --max_steps 1000 \
    --num_generations 8

# Train on simple algorithmic tasks (3 task types)
python src/train_reasoning_gym.py \
    --dataset_config simple_algorithmic \
    --dataset_size 10000 \
    --max_steps 500
```

### Use Custom Config File

```bash
# Train with your own task mix
python src/train_reasoning_gym.py \
    --dataset_config src/reasoning_gym_configs/algorithmic_simple.json \
    --dataset_size 15000
```

## Available Pre-defined Configs

### `algorithmic` (Full Suite)
10 different algorithmic reasoning tasks:
- ab, base_conversion, binary_alternation, binary_matrix
- caesar_cipher, cryptarithm, isomorphic_strings
- jugs, rotate_matrix, string_manipulation

### `simple_algorithmic` (Starter Set)
3 basic tasks good for quick experiments:
- ab, caesar_cipher, base_conversion

## Creating Custom Configs

Create a JSON file defining your task mix:

```json
{
  "task_name": {
    "weight": 1,
    "config": {
      "param1": "value1"
    }
  },
  "another_task": {
    "weight": 2
  }
}
```

**Available Reasoning Gym tasks**: See [Reasoning Gym documentation](https://github.com/open-thought/reasoning-gym)

**Example** (`my_config.json`):
```json
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
```

Then use it:
```bash
python src/train_reasoning_gym.py --dataset_config my_config.json
```

## Full Options

```bash
python src/train_reasoning_gym.py --help
```

**Key options**:
- `--dataset_config`: Config name or JSON file path (required)
- `--dataset_size`: Number of examples to generate (default: 20000)
- `--seed`: Random seed (default: 42)
- `--model-name`: Model to use (default: unsloth/Qwen2.5-3B-Instruct-bnb-4bit)
- `--max_steps`: Training steps (default: 1000)
- `--num_generations`: Samples per prompt (default: 8)
- `--batch_size`: Training batch size (default: 4)
- `--beta`: KL penalty (default: 0.001)
- `--max_completion_length`: Max tokens for completion (default: 512)

## Examples

### Quick Test Run
```bash
python src/train_reasoning_gym.py \
    --dataset_config simple_algorithmic \
    --dataset_size 5000 \
    --max_steps 100 \
    --num_generations 4
```

### Full Training Run
```bash
python src/train_reasoning_gym.py \
    --dataset_config algorithmic \
    --dataset_size 20000 \
    --max_steps 1000 \
    --num_generations 8 \
    --model-name unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
    --batch_size 4 \
    --save_steps 100
```

### Custom Model
```bash
python src/train_reasoning_gym.py \
    --dataset_config algorithmic \
    --model-name unsloth/Llama-3.1-8B-Instruct-bnb-4bit \
    --max_steps 500
```

## Checkpoints

Checkpoints are saved to:
```
checkpoints/reasoning_gym_{config_name}/{run_name}/
```

Access the final model at:
```
checkpoints/reasoning_gym_{config_name}/{run_name}/final/
```

## Monitoring

Training logs to W&B automatically:
- Project: `GRPO_REASONING_GYM` (or custom with `--project`)
- Metrics: reward, loss, cumulative successes

## Comparison with Original Scripts

| Feature | train.py | train_reasoning_gym.py |
|---------|----------|------------------------|
| Datasets | Fixed JSON files | Procedural generation |
| Model | Quantized + LoRA | Quantized + LoRA |
| Algorithm | GRPO | GRPO |
| Memory | Efficient (4-bit) | Efficient (4-bit) |
| Dataset Size | Limited by files | Unlimited |
| Task Variety | Single task | Multiple tasks mixed |

## Tips

1. **Start small**: Test with `--dataset_size 5000 --max_steps 100` first
2. **Task weights**: Adjust weights in config to focus on specific tasks
3. **Seed variation**: Use different `--seed` values to generate different datasets
4. **Monitor W&B**: Watch reward trends to see which tasks are learned
5. **Difficulty tuning**: Adjust task-specific configs to control difficulty

## Troubleshooting

**Import Error**: `No module named 'reasoning_gym'`
```bash
cd reasoning-gym && pip install -e .
```

**Out of Memory**: Reduce batch size or max_completion_length
```bash
--batch_size 2 --max_completion_length 256
```

**Slow training**: Reduce dataset size or number of generations
```bash
--dataset_size 10000 --num_generations 4
```

## Next Steps

1. Train on a small dataset to verify it works
2. Experiment with different task mixes
3. Compare performance across different configs
4. Evaluate on held-out tasks to test generalization

## Questions?

- Original train.py: Fixed datasets, difficulty selection
- train_reasoning_gym.py (this): Procedural Reasoning Gym datasets
- Both use the same Unsloth + LoRA + GRPO infrastructure!

