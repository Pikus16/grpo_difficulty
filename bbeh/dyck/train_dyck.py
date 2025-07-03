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
from datasets import load_dataset
from dyck_utils import load_dyck_dataset, format_single_question_qwen, extract_boxed_content

    
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
          max_steps: int = 1000):
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
        output_dir="runs",
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
    )
    trainer.train()
    
    model.save_pretrained(save_path)
    #model.save_pretrained_merged(f"{save_path}/merged", tokenizer, save_method="lora")

def setup_wandb(project='GRPO_DIFFICULTY', name='gsm8k'):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name

def format_dataset(ds, tokenizer):
    def _format_prompt(example):
        new_prompt = format_single_question_qwen(example['question'], tokenizer)
        return {'prompt' : new_prompt}
    ds = ds.map(_format_prompt)
    return ds

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
def main(project: str,
         save_dir: str, 
         num_generations: int,
         model_name: str,
         max_steps: int,
         load_4bit: bool):
    name = f'bbeh-dyck_{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    setup_wandb(project=project, name=name)

    dataset = load_dyck_dataset(subset='train')
    click.echo(f'Loaded train dataset of size {len(dataset)}')
    model, tokenizer = load_model_and_tokenizer(model_name=model_name, load_in_4bit=load_4bit)
    dataset = format_dataset(dataset, tokenizer)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train(model,
          tokenizer, 
          dataset, 
          save_path=os.path.join(save_dir, name),
          run_name=name,
          num_generations=int(num_generations),
          max_steps=max_steps)

if __name__ == '__main__':
    main()