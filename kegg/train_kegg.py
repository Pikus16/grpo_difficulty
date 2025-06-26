from kegg_utils import load_model_and_tokenizer, load_kegg_dataset, parse_llm_responses
import os
import click
from trl import GRPOConfig, GRPOTrainer

def correctness_reward_func(completions, answer, **kwargs):
    predictions = [parse_llm_responses(a) for a in completions]
    scores = []
    for r,a in zip(predictions, answer):
        if r is None:
            scores.append(0.0)
        else:
            if a.lower() in r.lower():
                scores.append(1.0)
            else:
                scores.append(0.0)
    return scores

def train(model,
          tokenizer,
          dataset,
          save_path: str,
          run_name: str,
          max_completion_length: int = 600,
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
        save_steps=200
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

def setup_wandb(project='GRPO_DIFFICULTY', name='gsm8k'):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name

@click.command()
@click.option('--project', type=str, default='GRPO_DIFFICULTY')
@click.option('--name', type=str, default='gsm8k')
@click.option('--save_path', type=str, default='runs')
@click.option('--num_generations', '-n', type=int, default=4, help='Number of generations per iteration')
def main(project, name, save_path, num_generations):
    setup_wandb(project=project, name=name)
    
    model, tokenizer = load_model_and_tokenizer()
    dataset = load_kegg_dataset(split='train')

    train(model,
          tokenizer, 
          dataset, 
          save_path=save_path,
          run_name=name,
          num_generations=int(num_generations))

if __name__ == '__main__':
    main()