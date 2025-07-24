import click
from shuffleobj_utils import run_on_all_checkpoints
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
@click.option(
    '--split', '-s',
    default='test',
    show_default=True,
    help="Dataset split to use"
)
@click.option('--num_generations', '-n', type=int, default=8, help='Number of generations per iteration')
@click.option('--max_steps',
              type=int,
              default=1000,
              help='Number of generations per iteration')
@click.option('--no-checkpoints', is_flag=True, default=False,
              help='If set, do not run on checkpoints (checkpoint_dir=None)')
@click.option(
    '--difficulty_level',
    type=int, 
    default=None,
    help='If specified, will load difficulty subset')
def main(
    model_name: str,
    num_repeat: int,
    batch_size: int,
    split: str,
    num_generations: int,
    max_steps: int,
    no_checkpoints: bool,
    difficulty_level: int
):
    name = f'bbeh-shuffleobj_{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    if difficulty_level is not None:
        name += f'_difficulty{difficulty_level}'
    checkpoint_dir = None if no_checkpoints else os.path.join('models', name, 'checkpoints')
    results = run_on_all_checkpoints(
        model_name,
        num_repeat,
        batch_size,
        checkpoint_dir,
        split,
        difficulty_level=difficulty_level
    )

    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, f'{split}_results.json'), 'w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()
