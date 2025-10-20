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
    _get_order_file,
    _get_checkpoint_dir,
    format_dataset_
)

def create_reward_func(dataset_name, regression_reward: bool = False):
    def shuffle_correctness_reward_func(completions, answer, **kwargs):
        predictions = np.array([extract_boxed_content(a) for a in completions])
        answer = np.array([a.lower() for a in answer])
        scores = answer == predictions
        if regression_reward:
            raise NotImplementedError()
        return scores.astype(int)
    
    def kegg_reward_func(completions, answer, **kwargs):
        predictions = np.array([extract_boxed_content(a) for a in completions])
        answer = np.array([a.lower() for a in answer])
        scores = np.array([
            a in p if p is not None else False
            for a, p in zip(answer, predictions)
        ])
        if regression_reward:
            raise NotImplementedError()
        return scores.astype(int)
    
    def gsm8k_reward_func(completions, answer, **kwargs):
        scores = []
        for c,a in zip(completions, answer):
            try:
                pred = int(extract_boxed_content(c))
                if regression_reward:
                    scores.append(
                        -np.abs(pred - a)
                    )
                else:
                    scores.append(
                        pred == a
                    )
            except:
                if regression_reward:
                    # Set to arbitrarily large negative number
                    scores.append(-1000000)
                else:
                    scores.append(0)
        return np.array(scores).astype(int)

    def cruxo_reward_func(completions, answer, **kwargs):
        def _process_fn(x):
            x = extract_boxed_content(x)
            try:
                return eval(x)
            except:
                try:
                    return json.loads(x)
                except:
                    return None

        scores = []
        for c, a in zip(completions, answer):
            a = eval(a)
            c = _process_fn(c)
            scores.append(1 if c == a else 0)
        if regression_reward:
            raise NotImplementedError()
        return np.array(scores).astype(int)

    def musique_reward_func(completions, answer, **kwargs):
        predictions = np.array([extract_boxed_content(a) for a in completions])
        answer = np.array([a.lower() for a in answer])
        scores = np.array([
            a == p if p is not None else False
            for a, p in zip(answer, predictions)
        ])
        if regression_reward:
            raise NotImplementedError()
        return scores.astype(int)

    def reasoning_gym_reward_func(completions, answer, **kwargs):
        """Generic reward function for Reasoning Gym tasks"""
        predictions = np.array([extract_boxed_content(a) for a in completions])
        # Normalize answers (lowercase, strip)
        answer = np.array([str(a).lower().strip() for a in answer])
        predictions = np.array([str(p).lower().strip() if p is not None else "" for p in predictions])
        
        scores = answer == predictions
        if regression_reward:
            raise NotImplementedError()
        return scores.astype(int)

    if dataset_name == 'kegg':
        return kegg_reward_func
    elif dataset_name == 'gsm8k':
        return gsm8k_reward_func
    elif dataset_name == 'shuffleobj':
        return shuffle_correctness_reward_func
    elif dataset_name == 'cruxo':
        return cruxo_reward_func
    elif dataset_name == 'musique':
        return musique_reward_func
    elif dataset_name == 'cognition_reasoning_gym':
        return reasoning_gym_reward_func
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
          dataset_name: str = None,
          beta: float = 0.001):
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
        save_steps=save_steps,
        beta=beta
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
        train_dataloader = trainer.get_train_dataloader()
        sampler = train_dataloader.sampler  # usually RandomSampler

        ids_ = []
        step = 0
        for _ in range(10000):  # big number, will break early
            for batch in train_dataloader:
                assert batch[0]['example_id'] == batch[-1]['example_id']
                ids_.append(batch[0]['example_id'])
                step += 1
                if step >= config.max_steps:  # stop once you've simulated all steps
                    break
            if step >= config.max_steps:
                break

        assert len(ids_) == max_steps
        with open(_get_order_file(dataset_name),'w') as f:
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
@click.option('--save_steps',
              type=int,
              default=100,
              help='How often to save')
@click.option('--load_4bit', '-l',
              is_flag=True,
              help='Load model in 4-bit mode (flag)')
@click.option('--skip_train', is_flag=True, default=False, help="Skip training and directly evaluate")
@click.option('--eval_last', is_flag=True, default=False, help="Only evaluate last checkpoint")
@click.option('--just_get_order', is_flag=True, default=False, help="Only save the order of train data samples")
@click.option('--test_batch_size',
              type=int,
              default=32,
              help='Batch size to use during evaluation')
@click.option('--test_num_repeat',
              type=int,
              default=1,
              help='Number of times to sample during test')
@click.option('--beta',
              type=float,
              default=0.001,
              help='Beta Term for KL-Divergence')
@click.option('--regression_reward', is_flag=True, default=False,
    help="Modify the reward to be regression rather than 0/1")
def main(
    dataset_name: str,
    strategy: str,
    subset_perc: float,
    project: str,
    num_generations: int,
    model_name: str,
    max_steps: int,
    save_steps: int,
    load_4bit: bool,
    skip_train: bool,
    eval_last: bool,
    just_get_order: bool,
    test_batch_size: int,
    test_num_repeat: int,
    beta: float,
    regression_reward: bool,
):
    name = f'{num_generations}gen_{max_steps}steps_{model_name}_beta{beta}'.replace('/','-')
    if strategy is not None:
        name += f'_strategy{strategy}'
    if subset_perc is not None:
        name += f'_subsetperc{subset_perc}'
    if regression_reward:
        click.echo(f"Using regression reward")
        name += '_regressionreward'
        
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
        if strategy is not None:
            click.echo(f'Loading {subset_perc} size')
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
            save_steps=save_steps,
            checkpoint_dir=checkpoint_dir,
            reward_fn=create_reward_func(dataset_name, regression_reward=regression_reward),
            just_get_data_order=just_get_order,
            dataset_name=dataset_name,
            beta=beta
        )
        
        # clear up memory before inference
        model.to('cpu')
        del model
        del tokenizer
        torch.cuda.empty_cache()

    if not just_get_order:
        # Run inference - determine correct path to get_answers.py
        if os.path.exists('get_answers.py'):
            get_answers_path = 'get_answers.py'
        elif os.path.exists('src/get_answers.py'):
            get_answers_path = 'src/get_answers.py'
        else:
            click.echo('Error: Cannot find get_answers.py')
            return
        
        cmd = f'python {get_answers_path} -m {model_name} --split test --dataset_name {dataset_name} -b {test_batch_size} --num_repeat {test_num_repeat} --run_name {name}'
        if eval_last:
            cmd += ' --eval_last'
        click.echo(f'Running command: {cmd}')
        subprocess.run(cmd, shell=True)

        # Log inference results to the same wandb run
        log_inference_results(
            os.path.join(checkpoint_dir, 'test_results.json')
        )

if __name__ == '__main__':
    main()