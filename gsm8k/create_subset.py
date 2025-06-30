import json
import numpy as np
from collections import defaultdict
import os
from gsm8k_utils import load_gsm8k_dataset, extract_boxed_content
import click

@click.command()
@click.option(
    '--dataset-name', '-d',
    default='openai/gsm8k',
    show_default=True,
    help="HuggingFace dataset name"
)
@click.option(
    '--subset', '-s',
    default='train',
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
    '--output_folder', '-o',
    default='gsm8k_output',
    show_default=True,
    help="Output folder where raw responses are written to"
)
@click.option(
    '--subset_folder',
    default='subset_train',
    show_default=True,
    help="Folder where dataset accuracy subsets should be written to"
)
def main(
    dataset_name: str,
    subset: str,
    model_name: str,
    output_folder: str,
    subset_folder: str
):
    ds = load_gsm8k_dataset(dataset_name, subset=subset)

    output_file = os.path.join(output_folder,
                               f"{dataset_name.replace('/','-')}_{subset}",
                               f"{model_name.replace('/','-')}.json")

    # Get accuracies
    with open(output_file) as f:
        js = json.load(f)

    print(f'{len(js)} number of questions processing')
    accs_inds = defaultdict(list)
    for i,responses in enumerate(js):
        preds = [extract_boxed_content(r) for r in responses]
        answer = ds[i]['parsed']
        score = np.mean(np.array(preds) == answer)
        accs_inds[score].append(i)

    
    # Write to subset

    if not os.path.exists(subset_folder):
        os.mkdir(subset_folder)
    
    subset_folder = os.path.join(
        subset_folder, 
        f"{dataset_name.replace('/','-')}_{subset}_{model_name.replace('/','-')}"
    )
    if not os.path.exists(subset_folder):
        os.mkdir(subset_folder)

    if not os.path.exists(subset_folder):
        os.mkdir(subset_folder)

    for acc, inds in accs_inds.items():
        subset_ds = ds.select(inds)
        acc = int(acc * 100)
        print(f'Accuracy: {acc}%, length = {len(subset_ds)}')
        subset_ds.to_json(f'{subset_folder}/{acc}.json')

if __name__ == '__main__':
    main()