import unsloth
from unsloth import FastLanguageModel
from kegg_utils import load_kegg_dataset, extract_boxed_content, format_single_question_qwen
import os
import click
from trl import GRPOConfig, GRPOTrainer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grpo_utils import CumulativeSuccessCallback

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

def correctness_reward_func(completions, answer, **kwargs):
    predictions = [extract_boxed_content(a) for a in completions]
    
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
        save_steps=20
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

def setup_wandb(project='GRPO_DIFFICULTY', name='gsm8k'):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name

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
def main(project, save_dir, num_generations, model_name: str,
         max_steps: int):
    name = f'kegg_{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    setup_wandb(project=project, name=name)
    
    model, tokenizer = load_model_and_tokenizer(model_name=model_name)
    dataset = load_kegg_dataset(split='train', tokenizer=tokenizer)
    click.echo(f'Loaded train dataset of size {len(dataset)}')

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

if __name__ == '__main__':
    main()
