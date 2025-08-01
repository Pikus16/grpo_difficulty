import unsloth
from unsloth import FastLanguageModel
import torch
import os
from trl import GRPOConfig, GRPOTrainer
import click
import wandb
import numpy as np
import json
from src_utils import (
    CumulativeSuccessCallback,
    get_dataset_subset,
    load_whole_dataset,
    extract_boxed_content,
    run_on_all_checkpoints,
    _get_checkpoint_dir,
    format_dataset_
)
    
def correctness_reward_func(completions, answer, **kwargs):
    predictions = np.array([extract_boxed_content(a) for a in completions])
    answer = np.array([a.lower() for a in answer])
    scores = answer == predictions
    return scores.astype(int)

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
          max_completion_length: int = 250,
          num_generations: int = 4,
          batch_size: int = 4,
          max_steps: int = 1000,
          checkpoint_dir: str = 'runs',
          save_steps: int = 100):
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
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
        ],
        args=config,
        train_dataset=dataset,
        callbacks=[CumulativeSuccessCallback()],
    )
    trainer.train()
    
    model.save_pretrained(f'{checkpoint_dir}/final')

def setup_wandb(project, name):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name

    # calling init now to save both train and test
    wandb.init(
        project=project,
        name=name
    )

def log_inference_results(results):
    """Log inference results to the active wandb run"""
    if wandb.run is None:
        print("Warning: No active wandb run found for logging inference results")
        return
    
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
    wandb.run.summary.update() 

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
def main(
    dataset_name: str,
    strategy: str,
    subset_perc: float,
    project: str,
    num_generations: int,
    model_name: str,
    max_steps: int,
    load_4bit: bool
):
    name = f'{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')

    dataset = load_whole_dataset(
        dataset_name=dataset_name,
        split='train',
        model_name=model_name
    )
    if strategy is not None and subset_perc is not None:
        click.echo(f'Loading {subset_perc} size subset with strategy {strategy}')
        name += f'_strategy{strategy}_subsetperc{subset_perc}'
        dataset = get_dataset_subset(
            whole_dataset=dataset,
            strategy = strategy,
            size = subset_perc
        )
    else:
        click.echo('Using whole dataset')
    
    setup_wandb(project=project, name=f'{dataset_name}_{name}')
    click.echo(f'Loaded train dataset of size {len(dataset)}')
    model, tokenizer = load_train_model_and_tokenizer(model_name=model_name, load_in_4bit=load_4bit)
    dataset = format_dataset_(dataset, tokenizer, dataset_name)

    checkpoint_dir = _get_checkpoint_dir(dataset_name, name)
    click.echo(f'Checkpoint directory: {checkpoint_dir}')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train(model,
          tokenizer, 
          dataset,
          run_name=name,
          num_generations=int(num_generations),
          max_steps=max_steps,
          checkpoint_dir=checkpoint_dir  
        )
    
    # clear up memory before inference
    model.to('cpu')
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # Run inference
    results =run_on_all_checkpoints(
        model_name = model_name,
        num_repeat=1, # hard coded to 1 for now
        batch_size=32,
        split ='test',
        dataset_name=dataset_name,
        run_name=name
    )

    # Log inference results to the same wandb run
    log_inference_results(results)
    
    # save inference info
    with open(os.path.join(checkpoint_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()