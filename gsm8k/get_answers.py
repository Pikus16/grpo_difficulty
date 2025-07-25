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
@click.option(
    '--subset_folder',
    default='accuracy_subset',
    show_default=True,
    help="Folder where dataset accuracy subsets should be read from"
)
@click.option(
    '--difficulty_level',
    type=int, 
    default=0,
    help='difficulty for gsm8k')
@click.option('--num_generations', '-n', type=int, default=8, help='Number of generations per iteration')
@click.option('--max_steps',
              type=int,
              default=1000,
              help='Number of generations per iteration')
@click.option('--no-checkpoints', is_flag=True, default=False,
              help='If set, do not run on checkpoints (checkpoint_dir=None)')
def main(
    dataset_name: str,
    subset: str,
    model_name: str,
    num_repeat: int,
    output_folder: str,
    batch_size: int,
    subset_folder: str,
    difficulty_level:int,
    num_generations: int,
    max_steps:int,
    no_checkpoints: bool,
):
    
    name = f'{dataset_name}_difficulty{difficulty_level}_{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    checkpoint_dir = None if no_checkpoints else os.path.join('models', name, 'checkpoints')
    results = run_on_all_checkpoints(
        dataset_name=dataset_name,
        subset=subset,
        model_name=model_name,
        num_repeat=num_repeat,
        batch_size=batch_size,
        adapter_folder=checkpoint_dir,
        subset_folder=subset_folder,
        difficulty_level=difficulty_level
    )

    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, f'test_results.json'), 'w') as f:
            json.dump(results, f)
    


if __name__ == '__main__':
    main()

