import click
from gsm8k_utils import run_on_all_checkpoints
import os
import json

@click.command()
@click.option(
    '--dataset-name', '-d',
    default='openai/gsm8k',
    show_default=True,
    help="HuggingFace dataset name"
)
@click.option(
    '--subset', '-s',
    default='test',
    show_default=True,
    help="Dataset split to use"
)
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option(
    '--num-repeat', '-k',
    default=8,
    show_default=True,
    help="Number of samples to generate per question"
)
@click.option(
    '--output_folder', '-o',
    default=None,
    show_default=True,
    help="Output folder to write all responses"
)
@click.option('--batch-size', '-b', default=32, 
              show_default=True, help="Number of questions to batch together")
@click.option('--adapter-folder', '-a', default='runs', 
              show_default=True, help="Folder with checkpoints")
@click.option(
    '--subset_folder',
    default='accuracy_subset',
    show_default=True,
    help="Folder where dataset accuracy subsets should be read from"
)
@click.option(
    '--difficulty_level',
    type=int, 
    default=None,
    help='difficulty for gsm8k')
def main(
    dataset_name: str,
    subset: str,
    model_name: str,
    num_repeat: int,
    output_folder: str,
    batch_size: int,
    adapter_folder: str, 
    subset_folder: str,
    difficulty_level:int
):
    results = run_on_all_checkpoints(
        dataset_name,
        subset,
        model_name,
        num_repeat,
        output_folder,
        batch_size,
        adapter_folder,
        subset_folder,
        difficulty_level
    )

    if output_folder is not None:
       with open(os.path.join(output_folder, f'{subset}_results.json'), 'w') as f:
          json.dump(results, f)
    


if __name__ == '__main__':
    main()

