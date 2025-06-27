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

# ---------- Constants ----------
PROMPT = """Solve the following math word problem.

{q}

Think step-by-step. Then, provide the final answer as a single integer in the format "Answer: XXX" with no extra formatting."""
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


# ---------- Utility Functions ----------
def make_dataset(difficulty_level, dir_path='outputs/gsm8k_platinum/accuracy_subset', subset='train'):
    ds = load_dataset("json", data_files=f'{dir_path}/{difficulty_level}_{subset}.jsonl', split="train")
    def format_prompt(example):
        new_prompt = PROMPT.format(q=example['question'])
        return {'prompt' : new_prompt}
    ds = ds.map(format_prompt)

    ds = ds.map(lambda x: {"answer": x["parsed"]})
    ds = ds.remove_columns("parsed")
    return ds

def parse_llm_answer(text):
    """
    Extracts the final answer from the LLM output.
    Expects the format: "Answer: XXX" where XXX is an integer.
    
    Args:
        text (str): The output from the LLM.
    
    Returns:
        int or None: The extracted integer answer, or None if not found.
    """
    match = re.search(r"Answer:\s*(-?\d+)", text)
    try:
        if match:
            return int(match.group(1))
    except:
        return None
    
def correctness_reward_func(completions, answer, **kwargs):
    predictions = np.array([parse_llm_answer(a) for a in completions])
    scores = np.array(answer) == predictions
    return scores.astype(int)

# ---------- Main Functions ----------
def load_model_and_tokenizer(max_seq_length: int = 2048, lora_rank: int = 32, load_in_4bit = False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        max_lora_rank=lora_rank,
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
          num_generations: int = 4):
    config = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        max_steps=1000,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="runs",
        run_name=run_name,
        save_steps=250
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
    model.save_lora(f"{save_path}/lora")
    #model.save_pretrained_merged(f"{save_path}/merged", tokenizer, save_method="lora")

def setup_wandb(project='GRPO_DIFFICULTY', name='gsm8k'):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name

@click.command()
@click.option('--project', type=str, default='GRPO_DIFFICULTY')
@click.option('--name', type=str, default='gsm8k')
@click.option('--save_path', type=str, default='runs')
@click.option('--difficulty_level', '-d', type=int, default=25, help='difficulty for gsm8k')
@click.option('--num_generations', '-n', type=int, default=4, help='Number of generations per iteration')
def main(project, name, save_path, difficulty_level, num_generations):
    click.echo(f'Using difficulty {difficulty_level}')
    setup_wandb(project=project, name=name)

    dataset = make_dataset(difficulty_level=difficulty_level)
    model, tokenizer = load_model_and_tokenizer()

    train(model,
          tokenizer, 
          dataset, 
          save_path=save_path,
          run_name=name,
          num_generations=int(num_generations))

if __name__ == '__main__':
    main()