#!/usr/bin/env python3
"""
Generate a train.json file from a Reasoning Gym curriculum yaml config.

Usage:
    python generate_dataset_from_yaml.py knights_knaves_curriculum.yaml --output train.json --size 20000
"""

import argparse
import json
import yaml
from pathlib import Path
import sys

# Add reasoning-gym to path if needed
SCRIPT_DIR = Path(__file__).parent.absolute()
REASONING_GYM_DIR = SCRIPT_DIR / "reasoning-gym"
if not REASONING_GYM_DIR.exists():
    REASONING_GYM_DIR = SCRIPT_DIR.parent / "reasoning-gym"

if REASONING_GYM_DIR.exists():
    sys.path.insert(0, str(REASONING_GYM_DIR))

try:
    import reasoning_gym
    from reasoning_gym.composite import DatasetSpec, CompositeConfig, CompositeDataset
    from reasoning_gym.utils import SYSTEM_PROMPTS
except ImportError as e:
    print("❌ Error: reasoning_gym not installed")
    print("Install with: cd reasoning-gym && pip install -e .")
    sys.exit(1)


def load_yaml_config(yaml_path: str) -> dict:
    """Load and parse the yaml configuration file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_dataset_config(config: dict) -> tuple:
    """
    Extract dataset configuration from yaml.
    
    Returns:
        tuple: (dataset_configs dict, dataset_size int, developer_prompt str)
    """
    reasoning_gym_config = config.get('reasoning_gym', {})
    
    # Get dataset configurations
    dataset_configs = reasoning_gym_config.get('datasets', {})
    if not dataset_configs:
        print("⚠️  Warning: No datasets defined in yaml config")
        print("    Add datasets to the 'reasoning_gym.datasets' section")
        print("    Example:")
        print("      reasoning_gym:")
        print("        datasets:")
        print("          knights_knaves:")
        print("            weight: 1")
        return None, None, None
    
    # Get dataset size
    dataset_size = reasoning_gym_config.get('dataset_size', 20000)
    
    # Get developer prompt
    developer_prompt = reasoning_gym_config.get('developer_prompt', None)
    
    return dataset_configs, dataset_size, developer_prompt


def generate_dataset(
    dataset_configs: dict,
    size: int = 20000,
    seed: int = 42,
    developer_prompt: str = None
) -> list:
    """
    Generate dataset from Reasoning Gym tasks.
    
    Args:
        dataset_configs: Dict of task names to config dicts
        size: Total number of examples to generate
        seed: Random seed
        developer_prompt: Optional system prompt
    
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    print(f"\n🔧 Generating dataset with {size} examples...")
    print(f"   Tasks: {list(dataset_configs.keys())}")
    print(f"   Seed: {seed}")
    if developer_prompt:
        print(f"   System prompt: {developer_prompt}")
    
    # Build dataset specs from config
    dataset_specs = []
    for name, config in dataset_configs.items():
        if config is None:
            config = {}
        weight = config.get('weight', 1) if isinstance(config, dict) else 1
        task_config = config.get('config', {}) if isinstance(config, dict) else {}
        
        print(f"   - {name}: weight={weight}, config={task_config}")
        dataset_specs.append(
            DatasetSpec(name=name, weight=weight, config=task_config)
        )
    
    # Create composite config
    composite_config = CompositeConfig(
        size=size,
        seed=seed,
        datasets=dataset_specs
    )
    
    # Create procedural dataset
    procedural_dataset = CompositeDataset(composite_config)
    
    # Convert to list of dicts
    data = []
    for i in range(len(procedural_dataset)):
        item = procedural_dataset[i]
        
        # Add system prompt if specified
        question = item['question']
        if developer_prompt and developer_prompt in SYSTEM_PROMPTS:
            system_text = SYSTEM_PROMPTS[developer_prompt]
            question = f"{system_text}\n\n{question}"
        
        data.append({
            'question': question,
            'answer': str(item['answer'])
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"   Generated {i + 1}/{size} examples...")
    
    print(f"✅ Generated {len(data)} examples")
    return data


def save_to_json(data: list, output_path: str):
    """Save dataset to JSON file."""
    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train.json from Reasoning Gym curriculum yaml"
    )
    parser.add_argument(
        'yaml_config',
        help='Path to yaml configuration file (e.g., knights_knaves_curriculum.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        default='train.json',
        help='Output JSON file path (default: train.json)'
    )
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=None,
        help='Number of examples to generate (overrides yaml config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for generation (default: 42)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help='Also generate test.json with this many examples'
    )
    
    args = parser.parse_args()
    
    # Load yaml config
    print(f"📄 Loading config from {args.yaml_config}...")
    config = load_yaml_config(args.yaml_config)
    
    # Extract dataset configuration
    dataset_configs, dataset_size, developer_prompt = extract_dataset_config(config)
    
    if dataset_configs is None:
        sys.exit(1)
    
    # Use command-line size if provided, otherwise use config
    size = args.size if args.size is not None else dataset_size
    
    # Generate training dataset
    train_data = generate_dataset(
        dataset_configs=dataset_configs,
        size=size,
        seed=args.seed,
        developer_prompt=developer_prompt
    )
    
    # Save training dataset
    save_to_json(train_data, args.output)
    
    # Generate test dataset if requested
    if args.test_size:
        print(f"\n🔧 Generating test dataset with {args.test_size} examples...")
        test_data = generate_dataset(
            dataset_configs=dataset_configs,
            size=args.test_size,
            seed=args.seed + 1,  # Different seed for test set
            developer_prompt=developer_prompt
        )
        
        test_output = args.output.replace('train.json', 'test.json')
        if test_output == args.output:
            test_output = args.output.replace('.json', '_test.json')
        
        save_to_json(test_data, test_output)
    
    print("\n✅ Dataset generation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the generated file: {args.output}")
    print(f"  2. Use with your training script")


if __name__ == '__main__':
    main()

