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
@click.option('--dataset_name', type=str, required=True)
@click.option('--batch_size', '-b', type=int, default=16, help='Batch size to use')
@click.option('--num_repeat',  type=int, default=10, help='Number of answers per question')
@click.option('--run_name',  type=str, default=None, help='Run name to load adapters')
def main(
    model_name: str,
    split: str,
    dataset_name: str,
    batch_size: int,
    num_repeat: int,
    run_name: str
):
    assert split in ['train','test']

    results = run_on_all_checkpoints(
        model_name=model_name,
        num_repeat=num_repeat,
        batch_size=batch_size,
        split=split,
        dataset_name=dataset_name,
        run_name=run_name
    )
    if run_name is not None:
        # save inference info
        checkpoint_dir = _get_checkpoint_dir(dataset_name, run_name)
        with open(os.path.join(checkpoint_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()