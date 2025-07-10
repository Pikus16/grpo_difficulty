import click
from function_utils import run_on_all_checkpoints
import os
import json

@click.command()
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option(
    '--num-repeat', '-k',
    default=1,
    show_default=True,
    help="Number of samples to generate per question"
)
@click.option('--batch-size', '-b', default=16, 
              show_default=True, help="Number of questions to batch together")
@click.option('--adapter-folder', '-a', default='runs', 
              show_default=True, help="Folder with checkpoints")
@click.option(
    '--subset', '-s',
    default='test',
    show_default=True,
    help="Dataset split to use"
)
def main(
    model_name: str,
    num_repeat: int,
    batch_size: int,
    adapter_folder: str,
    subset: str
):
    results = run_on_all_checkpoints(
        model_name,
        num_repeat,
        batch_size,
        adapter_folder,
        subset
    )

    with open(os.path.join(adapter_folder, f'{subset}_results.json'), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
