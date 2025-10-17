#!/usr/bin/env python3
"""
Generate train.json and/or test.json datasets from a Reasoning Gym curriculum yaml config.

Usage Examples:
    # Generate training set with size from yaml config
    python generate_dataset_from_yaml.py cognition_aiw.yaml
    
    # Generate training set with custom size
    python generate_dataset_from_yaml.py cognition_aiw.yaml --train-size 2000
    
    # Generate both train and test sets
    python generate_dataset_from_yaml.py cognition_aiw.yaml --generate-both --train-size 2000 --test-size 500
    
    # Generate to specific directory
    python generate_dataset_from_yaml.py cognition_aiw.yaml --output-dir ./dsets/my_dataset --generate-both
    
    # Custom seeds for reproducibility
    python generate_dataset_from_yaml.py cognition_aiw.yaml --seed 123 --test-seed 456 --generate-both
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
        description="Generate train.json and/or test.json from Reasoning Gym curriculum yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate only training set
  python generate_dataset_from_yaml.py cognition_aiw.yaml --train-size 2000

  # Generate both train and test sets
  python generate_dataset_from_yaml.py cognition_aiw.yaml --generate-both --train-size 2000 --test-size 500

  # Generate to specific directory
  python generate_dataset_from_yaml.py cognition_aiw.yaml --output-dir ./dsets/my_dataset --generate-both

  # Use custom output filename
  python generate_dataset_from_yaml.py cognition_aiw.yaml --output my_train.json --test-size 500
        """
    )
    parser.add_argument(
        'yaml_config',
        help='Path to yaml configuration file (e.g., cognition_aiw.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output JSON file path (default: train.json or <output-dir>/train.json)'
    )
    parser.add_argument(
        '--output-dir', '-d',
        default=None,
        help='Output directory for train.json and test.json (creates if not exists)'
    )
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=None,
        help='Number of training examples (deprecated, use --train-size)'
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=None,
        help='Number of training examples to generate (overrides yaml config)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help='Number of test examples to generate'
    )
    parser.add_argument(
        '--generate-both',
        action='store_true',
        help='Generate both train and test sets (uses dataset_size from yaml if sizes not specified)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for generation (default: 42)'
    )
    parser.add_argument(
        '--test-seed',
        type=int,
        default=None,
        help='Random seed for test set (default: train_seed + 1)'
    )
    
    args = parser.parse_args()
    
    # Load yaml config
    print(f"📄 Loading config from {args.yaml_config}...")
    config = load_yaml_config(args.yaml_config)
    
    # Extract dataset configuration
    dataset_configs, dataset_size, developer_prompt = extract_dataset_config(config)
    
    if dataset_configs is None:
        sys.exit(1)
    
    # Determine output paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_output = str(output_dir / 'train.json')
        test_output = str(output_dir / 'test.json')
    else:
        train_output = args.output if args.output else 'train.json'
        # Derive test output from train output
        test_output = train_output.replace('train.json', 'test.json')
        if test_output == train_output:
            test_output = train_output.replace('.json', '_test.json')
    
    # Determine train size (priority: --train-size > --size > yaml config)
    train_size = args.train_size or args.size or dataset_size
    
    # Determine test size
    test_size = args.test_size
    if args.generate_both and test_size is None:
        # Default to same size as training set if --generate-both is used
        test_size = train_size
    
    # Determine test seed
    test_seed = args.test_seed if args.test_seed is not None else args.seed + 1
    
    # Generate training dataset
    print(f"\n{'='*60}")
    print(f"📊 TRAINING DATASET")
    print(f"{'='*60}")
    train_data = generate_dataset(
        dataset_configs=dataset_configs,
        size=train_size,
        seed=args.seed,
        developer_prompt=developer_prompt
    )
    
    # Save training dataset
    save_to_json(train_data, train_output)
    
    # Generate test dataset if requested
    if test_size or args.generate_both:
        print(f"\n{'='*60}")
        print(f"📊 TEST DATASET")
        print(f"{'='*60}")
        test_data = generate_dataset(
            dataset_configs=dataset_configs,
            size=test_size,
            seed=test_seed,
            developer_prompt=developer_prompt
        )
        
        save_to_json(test_data, test_output)
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ Dataset generation complete!")
    print(f"{'='*60}")
    print(f"\n📁 Generated files:")
    print(f"  • Training: {train_output} ({train_size} examples)")
    if test_size or args.generate_both:
        print(f"  • Test: {test_output} ({test_size} examples)")
    print(f"\n💡 Next steps:")
    print(f"  1. Review the generated files")
    print(f"  2. Use with your training script")
    if args.output_dir:
        print(f"  3. All datasets are in: {args.output_dir}")


if __name__ == '__main__':
    main()

