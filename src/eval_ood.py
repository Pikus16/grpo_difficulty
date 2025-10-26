from src_utils import run_on_all_checkpoints, _get_checkpoint_dir
import click
import os
import json
from datasets import load_dataset

def load_aime():
    return load_dataset('opencompass/AIME2025', 'AIME2025-I', split='test')

@click.command()
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option(
    '--split',
    default='train',
    show_default=True,
    help="Which split to run on"
)
@click.option('--dataset_name', type=str, default=None, help='Dataset name (used for both training and eval if train_dataset_name not specified)')
@click.option('--train_dataset_name', type=str, default=None, help='Dataset used for training (for finding checkpoints)')
@click.option('--eval_dataset_name', type=str, default=None, help='Dataset to evaluate on (if different from training dataset)')
@click.option('--batch_size', '-b', type=int, default=16, help='Batch size to use')
@click.option('--num_repeat',  type=int, default=10, help='Number of answers per question')
@click.option('--run_name',  type=str, default=None, help='Run name to load adapters')
def main(
    model_name: str,
    split: str,
    dataset_name: str,
    train_dataset_name: str,
    eval_dataset_name: str,
    batch_size: int,
    num_repeat: int,
    run_name: str
):
    assert split in ['train','test']
    
    # Handle backward compatibility and parameter logic
    if train_dataset_name is None and eval_dataset_name is None:
        # Old behavior: dataset_name used for both
        if dataset_name is None:
            click.echo('Error: Must specify either --dataset_name or both --train_dataset_name and --eval_dataset_name')
            return
        train_dataset_name = dataset_name
        eval_dataset_name = dataset_name
    elif train_dataset_name is None or eval_dataset_name is None:
        # Partial specification
        if dataset_name is not None:
            # Use dataset_name as fallback
            train_dataset_name = train_dataset_name or dataset_name
            eval_dataset_name = eval_dataset_name or dataset_name
        else:
            click.echo('Error: Must specify both --train_dataset_name and --eval_dataset_name if not using --dataset_name')
            return

    results = run_on_all_checkpoints(
        model_name=model_name,
        num_repeat=num_repeat,
        batch_size=batch_size,
        split=split,
        train_dataset_name=train_dataset_name,
        eval_dataset_name=eval_dataset_name,
        run_name=run_name
    )
    if run_name is not None:
        # save inference info - use train_dataset_name for checkpoint location
        checkpoint_dir = _get_checkpoint_dir(train_dataset_name, run_name)
        
        # Include eval dataset name in results filename if different
        if train_dataset_name != eval_dataset_name:
            results_filename = f'test_results_{eval_dataset_name}.json'
        else:
            results_filename = 'test_results.json'
            
        with open(os.path.join(checkpoint_dir, results_filename), 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()