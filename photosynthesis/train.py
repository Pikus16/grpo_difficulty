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

# ---------- Constants ----------
PROMPT = "Describe photosynthesis. Use as simple terms as possible"
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


# ---------- Utility Functions ----------
def create_dataset_from_str(text: str, tokenizer) -> Dataset:
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}],
        tokenize=False, add_generation_prompt=True)
    data = {'prompt' : [text], 'answer' : ['']}
    return Dataset.from_dict(data)


# # ---------- Reward Functions ----------
def create_flesch_kincaid_reward_func(threshold: float):
    def flesch_kincaid_reward_func(completions, **kwargs):
        scores = [textstat.flesch_kincaid_grade(r) for r in completions]
        # Log average flesch score to W&B
        avg_flesch_score = sum(scores) / len(scores) if scores else 0
        wandb.log({"avg_flesch_kincaid_score": avg_flesch_score})
        
        return [1 if s < threshold else 0 for s in scores]
    return flesch_kincaid_reward_func


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
          flesch_threshold: float,
          max_seq_length: int = 2048,
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
        num_train_epochs=1000,
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
            create_flesch_kincaid_reward_func(flesch_threshold),
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
@click.option('--flesch_threshold', '-f', type=float, default=4, help='Flesch-Kincaid grade level threshold for reward function')
@click.option('--num_generations', '-n', type=int, default=4, help='Number of generations per iteration')
def main(project, name, save_path, flesch_threshold, num_generations):
    #os.environ['WANDB_API_KEY'] = '54a50cbe22da3f857fcb7812dd80fedc2ef01ad4'
    click.echo(f'Using threshold {flesch_threshold}')
    setup_wandb(project=project, name=name)
    
    model, tokenizer = load_model_and_tokenizer()
    dataset = create_dataset_from_str(text=PROMPT, tokenizer=tokenizer)

    train(model,
          tokenizer, 
          dataset, 
          save_path=save_path,
          run_name=name,
          flesch_threshold=flesch_threshold,
          num_generations=int(num_generations))

if __name__ == '__main__':
    main()