#!/usr/bin/env python3
"""
Training script for Reasoning Gym GRPO experiments.

This script combines the Reasoning Gym training approach with the working
patterns from existing train.py and burton_test.py scripts.

Based on:
- reasoning-gym/training/README.md setup instructions
- src/train.py working patterns
- AIME_evals/burton_test.py environment setup
"""

import os
import sys
import subprocess
import click
from pathlib import Path

# Add reasoning-gym to path if needed
SCRIPT_DIR = Path(__file__).parent.absolute()
REASONING_GYM_DIR = SCRIPT_DIR / "reasoning-gym"
if REASONING_GYM_DIR.exists():
    sys.path.insert(0, str(REASONING_GYM_DIR))


def check_environment():
    """Check if the environment is properly set up."""
    issues = []
    
    # Check if reasoning-gym directory exists
    if not REASONING_GYM_DIR.exists():
        issues.append(f"reasoning-gym directory not found at {REASONING_GYM_DIR}")
    
    # Check if verl is installed
    try:
        import verl
        click.echo(f"✓ verl is installed")
    except ImportError:
        issues.append("verl is not installed. Run: pip install git+https://github.com/volcengine/verl.git@c34206925e2a50fd452e474db857b4d488f8602d")
    
    # Check if reasoning_gym is installed
    try:
        import reasoning_gym
        click.echo(f"✓ reasoning_gym is installed")
    except ImportError:
        issues.append("reasoning_gym is not installed. Run: cd reasoning-gym/ && pip install -e .")
    
    # Check if flash-attn is installed
    try:
        import flash_attn
        click.echo(f"✓ flash-attn is installed")
    except ImportError:
        issues.append("flash-attn is not installed. Run: pip install flash-attn==2.7.3 --no-build-isolation")
    
    # Check if wandb is configured
    try:
        import wandb
        if wandb.api.api_key:
            click.echo(f"✓ wandb is configured")
        else:
            issues.append("wandb is not logged in. Run: wandb login")
    except:
        issues.append("wandb is not configured. Run: wandb login")
    
    # Check if huggingface is configured
    try:
        from huggingface_hub import HfFolder
        if HfFolder.get_token():
            click.echo(f"✓ huggingface-cli is configured")
        else:
            issues.append("huggingface is not logged in. Run: huggingface-cli login")
    except:
        issues.append("huggingface is not configured. Run: huggingface-cli login")
    
    if issues:
        click.echo("\n⚠️  Environment issues found:")
        for issue in issues:
            click.echo(f"  - {issue}")
        return False
    
    click.echo("\n✓ Environment check passed!")
    return True


def run_training(
    config_path: str,
    config_name: str,
    n_gpus: int = None,
    tensor_parallel_size: int = None,
    project_name: str = None,
    experiment_name: str = None,
    additional_args: list = None
):
    """Run the Reasoning Gym GRPO training."""
    
    training_dir = REASONING_GYM_DIR / "training"
    if not training_dir.exists():
        raise ValueError(f"Training directory not found at {training_dir}")
    
    # Build the command
    cmd = [
        "python3", "-u",
        str(training_dir / "train_grpo.py"),
        f"--config-path={config_path}",
        f"--config-name={config_name}"
    ]
    
    # Add optional overrides
    if n_gpus is not None:
        cmd.append(f"trainer.n_gpus_per_node={n_gpus}")
    
    if tensor_parallel_size is not None:
        cmd.append(f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}")
    
    if project_name is not None:
        cmd.append(f"trainer.project_name={project_name}")
    
    if experiment_name is not None:
        cmd.append(f"trainer.experiment_name={experiment_name}")
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    click.echo(f"\n🚀 Running training command:")
    click.echo(f"   {' '.join(cmd)}")
    click.echo(f"\n   Working directory: {training_dir}\n")
    
    # Set CUDA_VISIBLE_DEVICES if specified
    env = os.environ.copy()
    
    # Run the training
    result = subprocess.run(
        cmd,
        cwd=str(training_dir),
        env=env
    )
    
    return result.returncode


def convert_checkpoint_to_hf(
    fsdp_checkpoint_path: str,
    output_path: str,
    model_name: str
):
    """Convert FSDP checkpoint to HuggingFace format for easier evaluation."""
    
    training_dir = REASONING_GYM_DIR / "training"
    conversion_script = training_dir / "utils" / "load_fsdp_to_hf.py"
    
    if not conversion_script.exists():
        raise ValueError(f"Conversion script not found at {conversion_script}")
    
    cmd = [
        "python",
        str(conversion_script),
        fsdp_checkpoint_path,
        output_path,
        model_name
    ]
    
    click.echo(f"\n🔄 Converting checkpoint to HuggingFace format:")
    click.echo(f"   {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    return result.returncode


def run_evaluation(
    config_path: str,
    eval_config: str = None
):
    """Run evaluation on trained model."""
    
    eval_dir = REASONING_GYM_DIR / "training" / "evaluations"
    if not eval_dir.exists():
        raise ValueError(f"Evaluation directory not found at {eval_dir}")
    
    # Set environment variable for vLLM
    env = os.environ.copy()
    env["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    
    if eval_config:
        config_path = eval_config
    
    cmd = [
        "python",
        str(eval_dir / "evaluate_model.py"),
        "--config", config_path
    ]
    
    click.echo(f"\n📊 Running evaluation:")
    click.echo(f"   {' '.join(cmd)}\n")
    
    result = subprocess.run(
        cmd,
        cwd=str(eval_dir),
        env=env
    )
    
    return result.returncode


@click.group()
def cli():
    """Reasoning Gym Training Script - Train LLMs with GRPO on procedural datasets."""
    pass


@cli.command()
def check():
    """Check if the environment is properly set up."""
    check_environment()


@cli.command()
@click.option(
    '--config-path',
    default='configs/inter_generalisation',
    help='Path to config directory (relative to training/)'
)
@click.option(
    '--config-name',
    default='algorithmic_qwen_3b',
    help='Name of the config file (without .yaml)'
)
@click.option(
    '--n-gpus',
    type=int,
    default=None,
    help='Number of GPUs to use (overrides config)'
)
@click.option(
    '--tensor-parallel-size',
    type=int,
    default=None,
    help='Tensor parallel size for vLLM rollouts (overrides config)'
)
@click.option(
    '--project-name',
    default=None,
    help='W&B project name (overrides config)'
)
@click.option(
    '--experiment-name',
    default=None,
    help='W&B experiment name (overrides config)'
)
@click.option(
    '--cuda-devices',
    default=None,
    help='Comma-separated CUDA device IDs (e.g., "0,1,2,3")'
)
@click.option(
    '--check-env',
    is_flag=True,
    default=False,
    help='Check environment before training'
)
@click.argument('additional_args', nargs=-1, type=str)
def train(
    config_path: str,
    config_name: str,
    n_gpus: int,
    tensor_parallel_size: int,
    project_name: str,
    experiment_name: str,
    cuda_devices: str,
    check_env: bool,
    additional_args: tuple
):
    """
    Train a model using Reasoning Gym GRPO.
    
    Examples:
    
    \b
    # Train with default settings (4 GPUs)
    python train_reasoning_gym.py train --config-name algorithmic_qwen_3b
    
    \b
    # Train with 2 GPUs, 1 for vLLM
    python train_reasoning_gym.py train --config-name algorithmic_qwen_3b --n-gpus 2 --tensor-parallel-size 1
    
    \b
    # Train with custom W&B project
    python train_reasoning_gym.py train --config-name algorithmic_qwen_3b --project-name my-project --experiment-name my-exp
    
    \b
    # Train with specific GPUs
    python train_reasoning_gym.py train --config-name algorithmic_qwen_3b --cuda-devices "0,1"
    
    \b
    # Train with config overrides
    python train_reasoning_gym.py train --config-name algorithmic_qwen_3b trainer.total_training_steps=1000
    """
    
    if check_env:
        if not check_environment():
            click.echo("\n❌ Environment check failed. Please fix the issues above.")
            sys.exit(1)
    
    # Set CUDA_VISIBLE_DEVICES if specified
    if cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        click.echo(f"🔧 Setting CUDA_VISIBLE_DEVICES={cuda_devices}")
        
        # Adjust n_gpus if not explicitly set
        if n_gpus is None:
            n_gpus = len(cuda_devices.split(','))
            click.echo(f"🔧 Auto-detected {n_gpus} GPUs from CUDA_VISIBLE_DEVICES")
    
    returncode = run_training(
        config_path=config_path,
        config_name=config_name,
        n_gpus=n_gpus,
        tensor_parallel_size=tensor_parallel_size,
        project_name=project_name,
        experiment_name=experiment_name,
        additional_args=list(additional_args)
    )
    
    if returncode == 0:
        click.echo("\n✅ Training completed successfully!")
    else:
        click.echo(f"\n❌ Training failed with return code {returncode}")
        sys.exit(returncode)


@cli.command()
@click.option(
    '--fsdp-checkpoint',
    required=True,
    help='Path to FSDP checkpoint directory (e.g., checkpoints/.../global_step_400/actor)'
)
@click.option(
    '--output-path',
    required=True,
    help='Output path for HuggingFace checkpoint'
)
@click.option(
    '--model-name',
    default='converted_model',
    help='Name for the converted model'
)
def convert(fsdp_checkpoint: str, output_path: str, model_name: str):
    """
    Convert FSDP checkpoint to HuggingFace format.
    
    Example:
    
    \b
    python train_reasoning_gym.py convert \\
        --fsdp-checkpoint checkpoints/my-project/my-exp/global_step_400/actor \\
        --output-path checkpoints/my-project/my-exp/global_step_400/actor/huggingface \\
        --model-name qwen3b_trained
    """
    
    returncode = convert_checkpoint_to_hf(
        fsdp_checkpoint_path=fsdp_checkpoint,
        output_path=output_path,
        model_name=model_name
    )
    
    if returncode == 0:
        click.echo(f"\n✅ Checkpoint converted successfully to {output_path}")
    else:
        click.echo(f"\n❌ Conversion failed with return code {returncode}")
        sys.exit(returncode)


@cli.command()
@click.option(
    '--config',
    required=True,
    help='Path to evaluation config YAML file'
)
def evaluate(config: str):
    """
    Evaluate a trained model on Reasoning Gym tasks.
    
    Example:
    
    \b
    python train_reasoning_gym.py evaluate \\
        --config reasoning-gym/training/evaluations/inter_generalisation/algorithmic.yaml
    """
    
    returncode = run_evaluation(config_path=config)
    
    if returncode == 0:
        click.echo("\n✅ Evaluation completed successfully!")
    else:
        click.echo(f"\n❌ Evaluation failed with return code {returncode}")
        sys.exit(returncode)


@cli.command()
def setup():
    """
    Print setup instructions for Reasoning Gym training.
    """
    instructions = """
╔════════════════════════════════════════════════════════════════════════╗
║           Reasoning Gym Training Environment Setup                    ║
╚════════════════════════════════════════════════════════════════════════╝

Follow these steps to set up your environment:

1️⃣  Clone and install Reasoning Gym (if not already done):
    
    cd /path/to/grpo_difficulty
    git clone https://github.com/open-thought/reasoning-gym.git
    cd reasoning-gym/
    pip install -e .

2️⃣  Install dependencies:
    
    pip install wheel fire
    pip install git+https://github.com/volcengine/verl.git@c34206925e2a50fd452e474db857b4d488f8602d

3️⃣  Install flash-attention:
    
    pip install flash-attn==2.7.3 --no-build-isolation

4️⃣  Login to services:
    
    huggingface-cli login
    wandb login

5️⃣  Check your environment:
    
    python train_reasoning_gym.py check

6️⃣  Run your first training:
    
    python train_reasoning_gym.py train --config-name algorithmic_qwen_3b --check-env

═══════════════════════════════════════════════════════════════════════

📚 Available config templates:

  Inter-domain generalisation:
    - algorithmic_qwen_3b
    - algebra_qwen_3b
    - games_qwen_3b
    - logic_qwen_3b

  Intra-domain generalisation:
    - algebra_qwen_3b (in intra_generalisation/)
    - algorithmic_qwen_3b (in intra_generalisation/)
    - arithmetic_qwen_3b
    - cognition_qwen_3b
    - games_qwen_3b (in intra_generalisation/)
    - graphs_qwen_3b

  Curriculum learning:
    - knights_knaves_curriculum.yaml
    - spell_backward.yaml

═══════════════════════════════════════════════════════════════════════

💡 Tips:

  • For 2 GPU training: --n-gpus 2 --tensor-parallel-size 1
  • For specific GPUs: --cuda-devices "0,1"
  • For config overrides: Add them as arguments, e.g.:
    trainer.total_training_steps=1000
  • Check GPU usage with: nvidia-smi
  • Monitor training at: wandb.ai

═══════════════════════════════════════════════════════════════════════
"""
    click.echo(instructions)


@cli.command()
def list_configs():
    """List available training configurations."""
    
    configs_dir = REASONING_GYM_DIR / "training" / "configs"
    
    if not configs_dir.exists():
        click.echo(f"❌ Configs directory not found at {configs_dir}")
        return
    
    click.echo("\n📋 Available training configurations:\n")
    
    for config_type in configs_dir.iterdir():
        if config_type.is_dir() and not config_type.name.startswith('.'):
            click.echo(f"\n  {config_type.name}/")
            for config_file in config_type.glob("*.yaml"):
                click.echo(f"    - {config_file.stem}")
    
    click.echo("\n")


if __name__ == '__main__':
    cli()

