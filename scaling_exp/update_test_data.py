#!/usr/bin/env python3
"""
Script to create test_data CSV matching test_data_cleaned.csv format

This script:
1. Fetches training metrics (num_successes_seen, average_reward) from W&B
2. Loads test evaluation results from checkpoint test_results.json files
3. Combines them into the same format as test_data_cleaned.csv

Usage:
    # From W&B run ID, will look for test_results.json in checkpoints/
    python update_test_data.py \\
        --run_ids YOUR_RUN_ID \\
        --project GRPO_REASONING_GYM \\
        --checkpoint_base_dir checkpoints
"""

import pandas as pd
import wandb
import argparse
import json
from pathlib import Path
from datetime import datetime
import re


def extract_model_size(model_name: str) -> float:
    """Extract model size in billions from model name"""
    # Try to find number followed by B
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
    if match:
        return float(match.group(1))
    
    # Try common model patterns
    if 'phi-4' in model_name.lower():
        return 14.0
    elif 'qwen3-8b' in model_name.lower() or 'qwen2.5-8b' in model_name.lower():
        return 8.0
    elif 'qwen3-4b' in model_name.lower() or 'qwen2.5-4b' in model_name.lower():
        return 4.0
    elif 'qwen2.5-3b' in model_name.lower():
        return 3.0
    elif '7b' in model_name.lower():
        return 7.0
    
    return 0.0


def get_training_metrics_from_wandb(run_id: str, project: str = 'GRPO_DIFFICULTY') -> pd.DataFrame:
    """
    Fetch training metrics from W&B
    
    Returns DataFrame with columns: global_step, num_successes_seen, average_reward
    """
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    
    # Download the history
    history = run.history(samples=100000)
    
    # Process reward data
    df = history[["train/reward", 'train/global_step']]
    df = df[df['train/reward'].notnull()]
    
    # Keep only the last entry per global_step
    df = df.groupby('train/global_step', as_index=False).last().reset_index(drop=True)
    
    # Calculate cumulative successes
    num_gen = run.config.get('num_generations', 8)
    df['train/reward'] = (df['train/reward'] * num_gen).astype(int)
    df['num_successes_seen'] = df['train/reward'].cumsum()
    
    # Calculate average reward (EMA with span=100)
    df['average_reward'] = df['train/reward'].ewm(span=100, adjust=False).mean() / num_gen
    
    # Calculate train_score (recent training reward average)
    # This is the percentage of examples getting reward in recent steps
    df['train_score'] = df['train/reward'].rolling(window=min(100, len(df)), min_periods=1).mean() / num_gen
    
    # Rename column
    df = df.rename(columns={'train/global_step': 'checkpoint'})
    
    return df[['checkpoint', 'num_successes_seen', 'average_reward', 'train_score']], run


def load_test_results_json(json_path: Path) -> dict:
    """Load test_results.json file"""
    if not json_path.exists():
        raise FileNotFoundError(f"Test results not found: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)


def find_checkpoint_dir(run, checkpoint_base_dir: str = 'checkpoints') -> Path:
    """
    Find checkpoint directory for a W&B run
    
    Looks for patterns like:
    - checkpoints/dataset_name/8gen_1000steps_model_beta0.001/
    - checkpoints/8gen_1000steps_model_beta0.001/
    - checkpoints/dataset_name/full_run_name/
    """
    checkpoint_base = Path(checkpoint_base_dir)
    
    run_name = run.name
    dataset_name = run.config.get('dataset_name', '')
    model_name = run.config.get('model_name', '')
    num_gen = run.config.get('num_generations', 8)
    max_steps = run.config.get('max_steps', 1000)
    beta = run.config.get('beta', 0.001)
    
    # Extract model short name
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    
    # Build expected checkpoint dir name pattern
    # e.g., "8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_beta0.001"
    checkpoint_pattern = f"{num_gen}gen_{max_steps}steps_{model_short}_beta{beta}"
    
    # Try multiple patterns
    candidates = []
    
    # Pattern 1: checkpoint_base/checkpoint_pattern/
    candidates.append(checkpoint_base / checkpoint_pattern)
    
    # Pattern 2: checkpoint_base/dataset/checkpoint_pattern/
    if dataset_name:
        candidates.append(checkpoint_base / checkpoint_pattern)
    
    # Pattern 3: Full run name
    candidates.append(checkpoint_base / run_name)
    if dataset_name:
        candidates.append(checkpoint_base / run_name)
    
    # Check if any candidate exists
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    
    # Pattern 4: Search for directories matching the pattern
    if checkpoint_base.exists():
        # Look for any directory containing key parts of the checkpoint pattern
        search_patterns = [
            f"{num_gen}gen_{max_steps}steps",
            checkpoint_pattern,
            model_short,
        ]
        
        for subdir in checkpoint_base.rglob('*'):
            if not subdir.is_dir():
                continue
            
            subdir_name = subdir.name
            # Check if directory name matches our patterns
            if any(pattern in subdir_name for pattern in search_patterns):
                # Additional check: make sure it has test_results.json
                if (subdir / 'test_results.json').exists():
                    return subdir
    
    # If we still haven't found it, list what's actually there
    available_dirs = []
    if checkpoint_base.exists():
        available_dirs = [str(d.relative_to(checkpoint_base)) for d in checkpoint_base.rglob('*') 
                         if d.is_dir() and (d / 'test_results.json').exists()]
    
    raise FileNotFoundError(
        f"Could not find checkpoint directory for run: {run_name}\n"
        f"Searched in: {checkpoint_base}\n"
        f"Expected pattern: {checkpoint_pattern}\n"
        f"Available directories with test_results.json:\n" + 
        '\n'.join(f"  - {d}" for d in available_dirs[:10])
    )


def combine_run_data(run_id: str, project: str, checkpoint_base_dir: str) -> pd.DataFrame:
    """
    Combine training metrics from W&B with test evaluation results
    
    Returns DataFrame matching test_data_cleaned.csv format
    """
    print(f"\nProcessing run: {run_id}")
    
    # 1. Get training metrics from W&B
    print("  ├─ Fetching training metrics from W&B...")
    training_df, run = get_training_metrics_from_wandb(run_id, project)
    print(f"  │  ✓ Got {len(training_df)} training steps")
    
    # 2. Find checkpoint directory and test_results.json
    print("  ├─ Looking for test_results.json...")
    checkpoint_dir = find_checkpoint_dir(run, checkpoint_base_dir)
    test_results_path = checkpoint_dir / 'test_results.json'
    
    test_results = load_test_results_json(test_results_path)
    print(f"  │  ✓ Found: {test_results_path}")
    
    # 3. Extract test metrics
    checkpoints = test_results.get('checkpoint', [])
    accuracies = test_results.get('accuracy', [])
    base_accuracy = test_results.get('base accuracy', 0.0)
    
    print(f"  │  ✓ Got {len(checkpoints)} checkpoint evaluations")
    print(f"  │  ✓ Base accuracy: {base_accuracy:.4f}")
    
    if not checkpoints or not accuracies:
        raise ValueError(f"No checkpoint data in test_results.json: {test_results_path}")
    
    # 4. Extract metadata from run config
    dataset_name = run.config.get('dataset_name', 'unknown')
    model_name = run.config.get('model_name', None)
    
    # Fallback to run name if model_name not in config
    if not model_name or model_name == 'unknown':
        model_name = run.name
        print(f"  │  ⚠️  model_name not in config, using run name: {model_name}")
    
    strategy = run.config.get('strategy', 'unknown')
    
    # Simplify model name
    if 'unsloth/' in model_name:
        model_name = model_name.split('unsloth/')[1]
    if '-bnb-4bit' in model_name:
        model_name = model_name.replace('-bnb-4bit', '')
    if '-unsloth' in model_name:
        model_name = model_name.replace('-unsloth', '')
    
    model_size = extract_model_size(model_name)
    
    # 5. Calculate derived metrics
    final_acc = accuracies[-1] if accuracies else 0.0
    
    # 6. Create rows for each checkpoint
    rows = []
    for checkpoint, accuracy in zip(checkpoints, accuracies):
        # Find corresponding training metrics
        # Get the training metrics at or before this checkpoint
        train_row = training_df[training_df['checkpoint'] <= checkpoint]
        
        if len(train_row) > 0:
            train_row = train_row.iloc[-1]  # Get last row before/at checkpoint
            num_successes_seen = train_row['num_successes_seen']
            average_reward = train_row['average_reward']
            train_score = train_row['train_score']
        else:
            # Checkpoint before any training data (shouldn't happen normally)
            num_successes_seen = 0
            average_reward = 0.0
            train_score = 0.0
        
        # Calculate improvement metrics
        abs_improv = accuracy - base_accuracy
        
        # perc_learnable: what fraction of the learnable gap was closed
        # learnable gap = (1.0 - base_accuracy)
        learnable_gap = 1.0 - base_accuracy
        perc_learnable = abs_improv / learnable_gap if learnable_gap > 0 else 0.0
        
        # relative_improvement: improvement relative to base
        relative_improvement = abs_improv / base_accuracy if base_accuracy > 0 else 0.0
        
        row = {
            'dataset': dataset_name,
            'strategy': strategy,
            'model_name': model_name,
            'accuracy': accuracy,
            'base': base_accuracy,
            'final_acc': final_acc,
            'checkpoint': checkpoint,
            'num_successes_seen': int(num_successes_seen),
            'average_reward': average_reward,
            'perc_learnable': perc_learnable,
            'abs_improv': abs_improv,
            'model_size': model_size,
            'relative_improvement': relative_improvement,
            'train_score': train_score,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print(f"  └─ ✓ Created {len(df)} rows (checkpoints: {checkpoints[0]} to {checkpoints[-1]})")
    print(f"     Final accuracy: {final_acc:.4f} | Improvement: {final_acc - base_accuracy:+.4f}")
    
    return df


def create_combined_csv(run_ids: list, project: str, checkpoint_base_dir: str,
                       output_path: str = None, base_csv: str = None, merge: bool = False):
    """
    Create CSV combining W&B training metrics with test evaluation results
    
    Args:
        run_ids: List of W&B run IDs
        project: W&B project name
        checkpoint_base_dir: Base directory where checkpoints are stored
        output_path: Output CSV path (auto-generated if None)
        base_csv: Path to existing CSV to merge with (optional)
        merge: If True, merge with base_csv data
    """
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"scaling_exp/test_data_{timestamp}.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load base CSV if merging
    existing_df = pd.DataFrame()
    if merge and base_csv:
        base_csv_path = Path(base_csv)
        if base_csv_path.exists():
            existing_df = pd.read_csv(base_csv_path)
            print(f"📂 Loaded base CSV with {len(existing_df)} rows from: {base_csv}")
            print(f"   Existing runs: {len(existing_df.groupby(['dataset', 'model_name', 'strategy']))}")
        else:
            print(f"⚠️  Base CSV not found: {base_csv}")
    
    # Process each run
    print(f"\n🔄 Processing {len(run_ids)} run(s)...")
    print("=" * 70)
    
    new_dfs = []
    failed_runs = []
    
    for i, run_id in enumerate(run_ids, 1):
        print(f"\n[{i}/{len(run_ids)}] Run ID: {run_id}")
        try:
            df = combine_run_data(run_id, project, checkpoint_base_dir)
            new_dfs.append(df)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed_runs.append((run_id, str(e)))
    
    # Combine data
    if new_dfs:
        if merge and not existing_df.empty:
            combined_df = pd.concat([existing_df] + new_dfs, ignore_index=True)
            print(f"\n📊 Merged with existing data")
        else:
            combined_df = pd.concat(new_dfs, ignore_index=True)
            print(f"\n📊 Created new dataset")
        
        # Sort for consistency
        combined_df = combined_df.sort_values(
            ['dataset', 'strategy', 'model_name', 'checkpoint']
        ).reset_index(drop=True)
        
        # Save to CSV
        combined_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 70)
        print(f"✅ SUCCESS! CSV saved to: {output_path}")
        print("=" * 70)
        print(f"   📈 Total rows: {len(combined_df)}")
        print(f"   🏃 Successful runs: {len(new_dfs)}/{len(run_ids)}")
        print(f"   📊 Unique configurations: {len(combined_df.groupby(['dataset', 'model_name', 'strategy']))}")
        
        # Show sample
        print(f"\n📋 Sample (first 3 rows):")
        print(combined_df[['dataset', 'model_name', 'checkpoint', 'accuracy', 'num_successes_seen']].head(3).to_string(index=False))
        
        if failed_runs:
            print(f"\n⚠️  Failed runs ({len(failed_runs)}):")
            for run_id, error in failed_runs:
                print(f"   - {run_id}: {error}")
    else:
        print("\n✗ No data to save - all runs failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create CSV matching test_data_cleaned.csv format from W&B + test_results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single run
  python update_test_data.py \\
      --run_ids abc123 \\
      --project GRPO_REASONING_GYM \\
      --checkpoint_base_dir checkpoints
  
  # Process multiple runs
  python update_test_data.py \\
      --run_ids run1 run2 run3 \\
      --project GRPO_REASONING_GYM
  
  # Merge with existing CSV
  python update_test_data.py \\
      --run_ids new_run \\
      --merge \\
      --base scaling_exp/test_data_cleaned.csv
  
  # Custom output location
  python update_test_data.py \\
      --run_ids abc123 \\
      --output my_results.csv

This script:
1. Fetches training metrics (num_successes_seen, average_reward) from W&B
2. Loads test evaluation results from checkpoint/test_results.json
3. Combines them with calculated metrics (perc_learnable, abs_improv, etc.)
4. Outputs in the same format as test_data_cleaned.csv
        """
    )
    parser.add_argument('--run_ids', nargs='+', required=True,
                        help='W&B run IDs (space-separated)')
    parser.add_argument('--project', default='GRPO_DIFFICULTY',
                        help='W&B project name (default: GRPO_DIFFICULTY)')
    parser.add_argument('--checkpoint_base_dir', default='checkpoints',
                        help='Base directory for checkpoints (default: checkpoints)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path (default: timestamped file in scaling_exp/)')
    parser.add_argument('--base', default=None,
                        help='Base CSV to merge with (requires --merge)')
    parser.add_argument('--merge', action='store_true',
                        help='Merge new runs with base CSV')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("🔬 TEST DATA CSV GENERATOR")
    print("=" * 70)
    print(f"Format: Matches test_data_cleaned.csv")
    print(f"Project: {args.project}")
    print(f"Checkpoint dir: {args.checkpoint_base_dir}")
    print("=" * 70)
    
    create_combined_csv(
        run_ids=args.run_ids,
        project=args.project,
        checkpoint_base_dir=args.checkpoint_base_dir,
        output_path=args.output,
        base_csv=args.base,
        merge=args.merge
    )
