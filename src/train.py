import unsloth
from unsloth import FastLanguageModel
import torch
import os
from trl import GRPOConfig, GRPOTrainer
import click
import wandb
import numpy as np
import json
import subprocess
from src_utils import (
    CumulativeSuccessCallback,
    get_dataset_subset,
    load_whole_dataset,
    extract_boxed_content,
    _get_base_path,
    _get_checkpoint_dir,
    format_dataset_
)

def create_reward_func(dataset_name):
    def shuffle_correctness_reward_func(completions, answer, **kwargs):
        predictions = np.array([extract_boxed_content(a) for a in completions])
        answer = np.array([a.lower() for a in answer])
        scores = answer == predictions
        return scores.astype(int)
    
    def kegg_reward_func(completions, answer, **kwargs):
        predictions = np.array([extract_boxed_content(a) for a in completions])
        answer = np.array([a.lower() for a in answer])
        scores = np.array([
            a in p if p is not None else False
            for a, p in zip(answer, predictions)
        ])
        return scores.astype(int)
    
    def gsm8k_reward_func(completions, answer, **kwargs):
        predictions = []
        for a in completions:
            try:
                predictions.append(
                    int(extract_boxed_content(a))
                )
            except:
                predictions.append(None)
        predictions = np.array(predictions)
        scores = np.array(answer) == predictions
        return scores.astype(int)


    if dataset_name == 'kegg':
        return kegg_reward_func
    elif dataset_name == 'gsm8k':
        return gsm8k_reward_func
    elif dataset_name == 'shuffleobj':
        return shuffle_correctness_reward_func
    else:
        raise ValueError(f'Unknown dataset name {dataset_name}')

# ---------- Main Functions ----------
def load_train_model_and_tokenizer(model_name, max_seq_length: int = 2048, lora_rank: int = 32, load_in_4bit = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        #fast_inference=True,
        #max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer

def train(model,
          tokenizer,
          dataset,
          run_name: str,
          reward_fn, 
          max_completion_length: int = 250,
          num_generations: int = 4,
          batch_size: int = 4,
          max_steps: int = 1000,
          checkpoint_dir: str = 'runs',
          save_steps: int = 100,
          just_get_data_order: bool = True,
          dataset_name: str = None):
    config = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=checkpoint_dir,
        run_name=run_name,
        save_steps=save_steps
    )

    if just_get_data_order:
        dataset = dataset.add_column("example_id", list(range(len(dataset))))
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_fn,
        ],
        args=config,
        train_dataset=dataset,
        callbacks=[CumulativeSuccessCallback()],
    )
    if just_get_data_order:
        ids_ = []
        train_dataloader = trainer.get_train_dataloader()
        for _, batch in enumerate(train_dataloader):
            assert batch[0]['example_id'] == batch[-1]['example_id']
            ids_.append(batch[0]['example_id'])

        path_ = os.path.join(_get_base_path(), 'misc', 'train_data_order', f'{dataset_name}_{run_name}.json')
        with open(path_,'w') as f:
            json.dump(ids_, f)
    else:
        trainer.train()
        
        model.save_pretrained(f'{checkpoint_dir}/final')

def setup_wandb(project, name, skip_train):
    if skip_train:
        # resume previous run
        api = wandb.Api()
        runs = api.runs(f"{wandb.api.default_entity}/{project}")
        matched = [run.id for run in runs if run.name == name]
        if len(matched) == 0:
            raise ValueError(f"No W&B run with name '{name}' found in project '{project}'")
        id_ = matched[-1]
        print(f'Resume run {name} with id {id_}')
        wandb.init(project=project, name=name, id = id_, resume="must")
    else:
        os.environ['WANDB_PROJECT'] = project
        os.environ['WANDB_NAME'] = name

        # calling init now to save both train and test
        wandb.init(
            project=project,
            name=name
        )

def log_inference_results(results_path):
    """Log inference results to the active wandb run"""
    if wandb.run is None:
        print("Warning: No active wandb run found for logging inference results")
        return
    
    if not os.path.exists(results_path):
        raise ValueError(f'{results_path} not found')
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Extract data from results dictionary
    checkpoint_numbers = results.get('checkpoint', [])
    accuracies = results.get('accuracy', [])
    pass_at_k_key = [k for k in results.keys() if k.startswith('pass@')][0] if any(k.startswith('pass@') for k in results.keys()) else None
    pass_at_k_values = results.get(pass_at_k_key, []) if pass_at_k_key else []
    
    final_acc = accuracies[-1]
    best_acc = max(accuracies)
    final_pass = pass_at_k_values[-1]
    best_pass = max(pass_at_k_values)

    base_accuracy = results.get('base accuracy', 0)
    base_pass_at_k_key = [k for k in results.keys() if k.startswith('base pass@')][0] if any(k.startswith('base pass@') for k in results.keys()) else None
    base_pass_at_k = results.get(base_pass_at_k_key, 0) if base_pass_at_k_key else 0
    
    metric_dict =  {
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "base_acc": base_accuracy,
        f'final {pass_at_k_key}' : final_pass,
        f'best {pass_at_k_key}' : best_pass,
        f'base {pass_at_k_key}' : base_pass_at_k, 
        'checkpoints': checkpoint_numbers,
        'accuracies' : accuracies,
        pass_at_k_key: pass_at_k_values
    }

    wandb.run.summary.update(metric_dict)

    print(f"Logged inference results for {len(checkpoint_numbers)} checkpoints to wandb")

@click.command()
@click.option('--dataset_name', type=str, required=True)
@click.option(
    '--strategy',
    default=None,
    show_default=True,
    help="Strategy to use for selection (if none will use whole dataset)"
)
@click.option(
    '--subset_perc',
    type=float,
    default=None,
    help='Percentage of the dataset to grab as subset (if none will use whole dataset)'
)
@click.option('--project', type=str, default='GRPO_DIFFICULTY')
@click.option('--num_generations', '-n', type=int, default=8, help='Number of generations per iteration')
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option('--max_steps',
              type=int,
              default=1000,
              help='Number of generations per iteration')
@click.option('--load_4bit', '-l',
              is_flag=True,
              help='Load model in 4-bit mode (flag)')
@click.option('--skip_train', is_flag=True, default=False, help="Skip training and directly evaluate")
@click.option('--eval_last', is_flag=True, default=False, help="Only evaluate last checkpoint")
@click.option('--just_get_order', is_flag=True, default=False, help="Only save the order of train data samples")
def main(
    dataset_name: str,
    strategy: str,
    subset_perc: float,
    project: str,
    num_generations: int,
    model_name: str,
    max_steps: int,
    load_4bit: bool,
    skip_train: bool,
    eval_last: bool,
    just_get_order: bool
):
    name = f'{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    if strategy is not None and subset_perc is not None:
        name += f'_strategy{strategy}_subsetperc{subset_perc}'
    
    if not just_get_order:
        setup_wandb(project=project, name=f'{dataset_name}_{name}', skip_train=skip_train)

    checkpoint_dir = _get_checkpoint_dir(dataset_name, name)
    click.echo(f'Checkpoint directory: {checkpoint_dir}')

    if not skip_train:
        dataset = load_whole_dataset(
            dataset_name=dataset_name,
            split='train',
            model_name=model_name
        )
        if strategy is not None and subset_perc is not None:
            click.echo(f'Loading {subset_perc} size subset with strategy {strategy}')
            dataset = get_dataset_subset(
                whole_dataset=dataset,
                strategy = strategy,
                size = subset_perc
            )
        else:
            click.echo('Using whole dataset')
        
        click.echo(f'Loaded train dataset of size {len(dataset)}')
        model, tokenizer = load_train_model_and_tokenizer(model_name=model_name, load_in_4bit=load_4bit)
        dataset = format_dataset_(dataset, tokenizer, dataset_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        train(model,
            tokenizer, 
            dataset,
            run_name=name,
            num_generations=int(num_generations),
            max_steps=max_steps,
            checkpoint_dir=checkpoint_dir,
            reward_fn=create_reward_func(dataset_name),
            just_get_data_order=just_get_order,
            dataset_name=dataset_name
        )
        
        # clear up memory before inference
        model.to('cpu')
        del model
        del tokenizer
        torch.cuda.empty_cache()

    if not just_get_order:
        # Run inference
        cmd = f'python get_answers.py -m {model_name} --split test --dataset_name {dataset_name} -b 32 --num_repeat 1 --run_name {name}'
        if eval_last:
            cmd += ' --eval_last'
        click.echo(f'Runnng command: {cmd}')
        subprocess.run(cmd, shell=True)

        # Log inference results to the same wandb run
        log_inference_results(
            os.path.join(checkpoint_dir, 'test_results.json')
        )

if __name__ == '__main__':
    main()