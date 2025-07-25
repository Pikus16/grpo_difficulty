import unsloth
from unsloth import FastLanguageModel
import torch
import textstat
from vllm import SamplingParams
import os
from trl import GRPOConfig, GRPOTrainer
import click
import re
from datasets import Dataset
import wandb
import numpy as np
import json
from shuffleobj_utils import load_shuffleobj_dataset, format_single_question_qwen, extract_boxed_content, run_on_all_checkpoints
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from grpo_utils import CumulativeSuccessCallback

    
def correctness_reward_func(completions, answer, **kwargs):
    predictions = np.array([extract_boxed_content(a) for a in completions])
    scores = np.array(answer) == predictions
    return scores.astype(int)

# ---------- Main Functions ----------
def load_model_and_tokenizer(model_name, max_seq_length: int = 2048, lora_rank: int = 32, load_in_4bit = True):
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

def train(model, tokenizer, dataset,
          save_path: str,
          run_name: str,
          max_completion_length: int = 250,
          num_generations: int = 4,
          batch_size: int = 4,
          max_steps: int = 1000,
          checkpoint_dir: str = 'runs'):
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
        save_steps=100
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
    
    model.save_pretrained(save_path)
    #model.save_pretrained_merged(f"{save_path}/merged", tokenizer, save_method="lora")

def setup_wandb(project='GRPO_DIFFICULTY', name='gsm8k'):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name

    # calling init now to save both train and test
    wandb.init(
        project=project,
        name=name
    )

def format_dataset(ds, tokenizer):
    def _format_prompt(example):
        new_prompt = format_single_question_qwen(example['question'], tokenizer)
        return {'prompt' : new_prompt}
    ds = ds.map(_format_prompt)
    return ds

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
    
    base_accuracy = results.get('base accuracy', 0)
    base_pass_at_k_key = [k for k in results.keys() if k.startswith('base pass@')][0] if any(k.startswith('base pass@') for k in results.keys()) else None
    base_pass_at_k = results.get(base_pass_at_k_key, 0) if base_pass_at_k_key else 0
    
    # Log base model performance
    wandb.log({
        f'test/base_accuracy': base_accuracy,
        f'test/base_{pass_at_k_key}': base_pass_at_k
    })
    
    # Log per-checkpoint metrics
    for i, (checkpoint_num, accuracy) in enumerate(zip(checkpoint_numbers, accuracies)):
        log_data = {
            f'test/accuracy': accuracy,
            f'test/checkpoint_step': checkpoint_num
        }
        
        # Add pass@k if available
        if pass_at_k_values and i < len(pass_at_k_values):
            log_data[f'test/{pass_at_k_key}'] = pass_at_k_values[i]
        
        wandb.log(log_data, step=checkpoint_num)
    
    print(f"Logged inference results for {len(checkpoint_numbers)} checkpoints to wandb")

@click.command()
@click.option('--project', type=str, default='GRPO_DIFFICULTY')
@click.option('--save_dir', type=str, default='models')
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
@click.option(
    '--difficulty_level',
    type=str, 
    default=None,
    help='If specified, will load difficulty subset')
def main(project: str,
         save_dir: str, 
         num_generations: int,
         model_name: str,
         max_steps: int,
         load_4bit: bool,
         difficulty_level: int):
    name = f'bbeh-shuffleobj_{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    if difficulty_level is not None:
        name += f'_difficulty{difficulty_level}'
    setup_wandb(project=project, name=name)

    dataset = load_shuffleobj_dataset(
        split='train',
        difficulty_level=difficulty_level,
        model_name=model_name
    )
    
    click.echo(f'Loaded train dataset of size {len(dataset)}')
    model, tokenizer = load_model_and_tokenizer(model_name=model_name, load_in_4bit=load_4bit)
    dataset = format_dataset(dataset, tokenizer)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, name)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    train(model,
          tokenizer, 
          dataset, 
          save_path=save_dir,
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

    results = run_on_all_checkpoints(
        model_name,
        num_repeat=1, # hard coded to 1 for now
        output_folder=None, # not saving outputs for now
        batch_size=32,
        adapter_folder=checkpoint_dir,
        subset='test'
    )

    # Log inference results to the same wandb run
    log_inference_results(results)
    
    # save inference info
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()