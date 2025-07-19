from photo_utils import do_single_run
import click
import numpy as np
import json
import os

@click.command()
@click.option('--save_dir', type=str, default='responses')
@click.option('--batch_size', '-n', type=int, default=16)
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
def main(save_dir, batch_size, model_name: str,
         num_to_gen: int):
    
    avg_score, std_score, _, scores = do_single_run(
        model_name,
        adapter_name=None,
        batch_size=batch_size,
        num_to_gen=num_to_gen,
        desired_threshold=None
    )

    lowest_score = np.min(scores)
    highest_score = np.max(scores)
    quantiles = {}
    for quantile in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
        quantiles[f'Quantile {quantile}'] = np.quantile(scores, quantile)
    
    print(f'Lowest score: {lowest_score}')
    print(f'Highest score: {highest_score}')
    print(f'Average Score: {avg_score} +/- {std_score}')
    for q, s in quantiles.items():
        print(f'{q}: {s}')

    result = {
        'highest_score' : highest_score,
        'lowest_score' : lowest_score,
        'average_score' : avg_score,
        **quantiles,
    }
    model_name = model_name.replace('/','-')
    save_dir = f'{save_dir}/{model_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(f'{save_dir}/scores.json', 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()