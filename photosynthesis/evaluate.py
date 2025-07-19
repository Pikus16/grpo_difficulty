import click
from photo_utils import run_on_all_checkpoints
import os
import json

@click.command()
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option('--num_to_gen',
              type=int,
              default=1000,
              help='Number of generations')
@click.option('--batch-size', '-b', default=8, 
              show_default=True, help="Number of questions to batch together")
@click.option('--flesch_threshold', '-f', type=float, default=4, help='Flesch-Kincaid grade level threshold for reward function')
@click.option('--num_generations', '-n', type=int, default=8, help='Number of generations per iteration')
@click.option('--max_steps',
              type=int,
              default=1000,
              help='Number of generations per iteration')
def main(
    model_name: str,
    num_to_gen: int,
    batch_size: int,
    flesch_threshold: float,
    num_generations: int,
    max_steps: int
):
    name = f'photosynthesis{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    checkpoint_dir = os.path.join('models',name,'checkpoints')
    results = run_on_all_checkpoints(
        model_name=model_name,
        num_to_gen=num_to_gen,
        batch_size=batch_size,
        adapter_folder=checkpoint_dir,
        desired_threshold=flesch_threshold
    )

    with open(os.path.join(checkpoint_dir, f'test_results.json'), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
