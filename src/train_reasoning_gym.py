import unsloth
from unsloth import FastLanguageModel
import torch
import os
from trl import GRPOConfig, GRPOTrainer
import click
import wandb
import numpy as np
import json
import subprocess
from datasets import Dataset as HFDataset
from src_utils import (
    CumulativeSuccessCallback,
    extract_boxed_content,
    _get_checkpoint_dir,
)

# Import Reasoning Gym
try:
    import reasoning_gym
    from reasoning_gym.composite import DatasetSpec
    REASONING_GYM_AVAILABLE = True
except ImportError:
    REASONING_GYM_AVAILABLE = False
    print("Warning: reasoning_gym not installed. Install with: cd reasoning-gym && pip install -e .")


# ============================================================================
# REASONING GYM DATASET LOADING
# ============================================================================

def load_reasoning_gym_dataset(
    dataset_configs: dict,
    size: int = 20000,
    seed: int = 42,
    developer_prompt: str = None
) -> HFDataset:
    """
    Load a Reasoning Gym procedural dataset.
    
    Args:
        dataset_configs: Dict of dataset names to config dicts
            Example: {
                'ab': {'weight': 1},
                'caesar_cipher': {'weight': 1, 'config': {'max_words': 10}},
                'jugs': {'weight': 1, 'config': {'difficulty': 6}}
            }
        size: Total number of examples to generate
        seed: Random seed
        developer_prompt: Optional system prompt
    
    Returns:
        HuggingFace Dataset with 'question' and 'answer' columns
    """
    if not REASONING_GYM_AVAILABLE:
        raise ImportError("reasoning_gym not installed")
    
    # Build dataset specs from config
    dataset_specs = []
    for name, config in dataset_configs.items():
        weight = config.get('weight', 1)
        task_config = config.get('config', {})
        dataset_specs.append(
            DatasetSpec(name=name, weight=weight, config=task_config)
        )
    
    # Create procedural dataset
    procedural_dataset = reasoning_gym.create_dataset(
        'composite',
        seed=seed,
        size=size,
        datasets=dataset_specs
    )
    
    # Convert to HuggingFace format
    data = []
    for i in range(len(procedural_dataset)):
        item = procedural_dataset[i]
        data.append({
            'question': item['question'],
            'answer': str(item['answer'])  # Ensure answer is string
        })
    
    return HFDataset.from_list(data)


# Pre-defined dataset configurations
REASONING_GYM_CONFIGS = {
    'algorithmic': {
        'ab': {'weight': 1},
        'base_conversion': {'weight': 1},
        'binary_alternation': {'weight': 1, 'config': {'p_solvable': 0.9}},
        'binary_matrix': {'weight': 1, 'config': {'min_n': 2, 'max_n': 6}},
        'caesar_cipher': {'weight': 1, 'config': {'max_words': 10}},
        'cryptarithm': {'weight': 1},
        'isomorphic_strings': {'weight': 1, 'config': {'max_string_length': 8}},
        'jugs': {'weight': 1, 'config': {'difficulty': 6}},
        'rotate_matrix': {'weight': 1, 'config': {'min_n': 2, 'max_n': 6}},
        'string_manipulation': {'weight': 1, 'config': {'max_string_length': 15, 'max_num_rules': 6}},
    },
    'simple_algorithmic': {
        'ab': {'weight': 1},
        'caesar_cipher': {'weight': 1, 'config': {'max_words': 5}},
        'base_conversion': {'weight': 1},
    },
    # Add more configs as needed
}


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def create_reasoning_gym_reward_func():
    """
    Generic reward function for Reasoning Gym tasks.
    Checks if extracted answer matches ground truth.
    """
    def reward_func(completions, answer, **kwargs):
        predictions = np.array([extract_boxed_content(a) for a in completions])
        # Normalize answers (lowercase, strip)
        answer = np.array([str(a).lower().strip() for a in answer])
        predictions = np.array([str(p).lower().strip() if p is not None else "" for p in predictions])
        
        scores = answer == predictions
        return scores.astype(int)
    
    return reward_func


# ============================================================================
# FORMATTING
# ============================================================================

def format_reasoning_gym_prompt(question: str, tokenizer) -> str:
    """Format a Reasoning Gym question for the model."""
    prompt = f"{question}\nThink step by step and put your final answer within \\boxed{{}}."
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True
    )


def format_reasoning_gym_dataset(dataset: HFDataset, tokenizer) -> HFDataset:
    """Format dataset for training."""
    def _format(example):
        prompt = format_reasoning_gym_prompt(example['question'], tokenizer)
        return {'prompt': prompt, 'answer': example['answer']}
    
    dataset = dataset.map(_format)
    dataset = dataset.remove_columns(['question'])
    return dataset


# ============================================================================
# MODEL & TRAINING (same as original train.py)
# ============================================================================

def load_train_model_and_tokenizer(model_name, max_seq_length: int = 2048, lora_rank: int = 32, load_in_4bit = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        gpu_memory_utilization=0.9,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


def train(model,
          tokenizer,
          dataset,
          run_name: str,
          reward_fn, 
          max_completion_length: int = 250,
          num_generations: int = 4,
          batch_size: int = 4,
          max_steps: int = 1000,
          checkpoint_dir: str = 'runs',
          save_steps: int = 100,
          beta: float = 0.001):
    config = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=checkpoint_dir,
        run_name=run_name,
        save_steps=save_steps,
        beta=beta
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=dataset,
        callbacks=[CumulativeSuccessCallback()],
    )
    
    trainer.train()
    model.save_pretrained(f'{checkpoint_dir}/final')


def setup_wandb(project, name):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name
    wandb.init(project=project, name=name)


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option(
    '--dataset_config',
    type=str,
    required=True,
    help="Reasoning Gym config name (e.g., 'algorithmic', 'simple_algorithmic') or path to custom JSON config"
)
@click.option(
    '--dataset_size',
    type=int,
    default=20000,
    help='Number of examples to generate'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for dataset generation'
)
@click.option('--project', type=str, default='GRPO_REASONING_GYM')
@click.option('--num_generations', '-n', type=int, default=8, help='Number of generations per iteration')
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen2.5-3B-Instruct-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option('--max_steps', type=int, default=1000, help='Number of training steps')
@click.option('--save_steps', type=int, default=100, help='How often to save')
@click.option('--load_4bit', '-l', is_flag=True, default=True, help='Load model in 4-bit mode')
@click.option('--beta', type=float, default=0.001, help='Beta Term for KL-Divergence')
@click.option('--batch_size', type=int, default=4, help='Training batch size')
@click.option('--max_completion_length', type=int, default=512, help='Max tokens for model completion')
def main(
    dataset_config: str,
    dataset_size: int,
    seed: int,
    project: str,
    num_generations: int,
    model_name: str,
    max_steps: int,
    save_steps: int,
    load_4bit: bool,
    beta: float,
    batch_size: int,
    max_completion_length: int,
):
    """
    Train with Reasoning Gym procedural datasets using Unsloth + LoRA.
    
    Examples:
    
    \b
    # Train on algorithmic tasks
    python train_reasoning_gym.py --dataset_config algorithmic --dataset_size 20000
    
    \b
    # Train on simple algorithmic tasks with custom settings
    python train_reasoning_gym.py --dataset_config simple_algorithmic \\
        --dataset_size 10000 --max_steps 500 --num_generations 4
    
    \b
    # Use custom config JSON file
    python train_reasoning_gym.py --dataset_config my_config.json
    """
    
    # Check if reasoning_gym is available
    if not REASONING_GYM_AVAILABLE:
        raise ImportError(
            "reasoning_gym not installed. Install with:\n"
            "  cd reasoning-gym && pip install -e ."
        )
    
    # Load dataset config
    if dataset_config.endswith('.json'):
        # Load custom config from file
        with open(dataset_config) as f:
            dataset_configs = json.load(f)
        config_name = os.path.basename(dataset_config).replace('.json', '')
    elif dataset_config in REASONING_GYM_CONFIGS:
        # Use pre-defined config
        dataset_configs = REASONING_GYM_CONFIGS[dataset_config]
        config_name = dataset_config
    else:
        raise ValueError(
            f"Unknown dataset_config: {dataset_config}\n"
            f"Available: {list(REASONING_GYM_CONFIGS.keys())} or path to JSON file"
        )
    
    # Create run name
    name = f'{num_generations}gen_{max_steps}steps_{model_name}_beta{beta}_rg_{config_name}_size{dataset_size}'.replace('/', '-')
    
    # Setup W&B
    setup_wandb(project=project, name=name)
    
    # Setup checkpoint directory
    checkpoint_dir = _get_checkpoint_dir(f'reasoning_gym_{config_name}', name)
    click.echo(f'Checkpoint directory: {checkpoint_dir}')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Generate Reasoning Gym dataset
    click.echo(f'Generating Reasoning Gym dataset: {config_name}')
    click.echo(f'  Size: {dataset_size}, Seed: {seed}')
    dataset = load_reasoning_gym_dataset(
        dataset_configs=dataset_configs,
        size=dataset_size,
        seed=seed
    )
    click.echo(f'Generated {len(dataset)} examples')
    
    # Load model
    click.echo(f'Loading model: {model_name}')
    model, tokenizer = load_train_model_and_tokenizer(
        model_name=model_name,
        load_in_4bit=load_4bit
    )
    
    # Format dataset
    click.echo('Formatting dataset...')
    dataset = format_reasoning_gym_dataset(dataset, tokenizer)
    
    # Train
    click.echo(f'Starting training for {max_steps} steps...')
    train(
        model,
        tokenizer,
        dataset,
        run_name=name,
        num_generations=int(num_generations),
        max_steps=max_steps,
        save_steps=save_steps,
        checkpoint_dir=checkpoint_dir,
        reward_fn=create_reasoning_gym_reward_func(),
        beta=beta,
        batch_size=batch_size,
        max_completion_length=max_completion_length
    )
    
    click.echo(f'✅ Training complete! Model saved to: {checkpoint_dir}/final')
    
    # Clear memory
    model.to('cpu')
    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

