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
from transformers import TrainerCallback, TrainerState, TrainerControl

class CurriculumLearningCallback(TrainerCallback):
    """
    Callback that implements curriculum learning by removing examples with average reward >= 0.25
    at every epoch (evaluated every save_steps).
    """
    def __init__(self, reward_threshold=0.25, save_steps=100):
        super().__init__()
        self.reward_threshold = reward_threshold
        self.save_steps = save_steps
        self.example_rewards = {}  # Track rewards per example
        self.example_counts = {}   # Track how many times each example was seen
        self.removed_examples = set()  # Track which examples have been removed
        self.last_curriculum_update = 0
        
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after each logging step to track rewards and update curriculum."""
        # Only update curriculum every save_steps
        if state.global_step - self.last_curriculum_update >= self.save_steps:
            self._update_curriculum(state, control)
            self.last_curriculum_update = state.global_step
        return control
    
    def _update_curriculum(self, state: TrainerState, control: TrainerControl):
        """Update the curriculum by removing high-performing examples."""
        if not hasattr(self, 'trainer') or self.trainer is None:
            assert False
            
        # Calculate average rewards for each example
        examples_to_remove = []
        for example_id, reward in self.example_rewards.items():
            if example_id in self.removed_examples:
                continue

            if reward >= self.reward_threshold:
                examples_to_remove.append(example_id)
        # for example_id, total_reward in self.example_rewards.items():
        #     if example_id in self.removed_examples:
        #         continue
                
        #     count = self.example_counts.get(example_id, 1)
        #     avg_reward = total_reward / count
            
        #     if avg_reward >= self.reward_threshold:
        #         examples_to_remove.append(example_id)
        
        if examples_to_remove:
            print(f"Removing {len(examples_to_remove)} examples with avg reward >= {self.reward_threshold}")
            self.removed_examples.update(examples_to_remove)
            
            # Update the trainer's dataset by filtering out removed examples
            self._update_trainer_dataset()
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    "curriculum/examples_removed": len(examples_to_remove),
                    "curriculum/total_removed": len(self.removed_examples),
                    "curriculum/remaining_examples": len(self.trainer.train_dataset) - len(self.removed_examples)
                })
    
    def _update_trainer_dataset(self):
        """Update the trainer's dataset by removing examples that have been marked for removal."""
        if not hasattr(self, 'trainer') or self.trainer is None:
            assert False
            
        # Create a new dataset without the removed examples
        original_dataset = self.trainer.train_dataset
        
        # Filter out removed examples
        def filter_examples(example):
            example_id = example.get('example_id', None)
            return example_id not in self.removed_examples
        
        filtered_dataset = original_dataset.filter(filter_examples)
        
        # Update the trainer's dataset
        self.trainer.train_dataset = filtered_dataset
        
        # Recreate the dataloader
        self.trainer._remove_unused_columns(self.trainer.train_dataset, description="training")
        self.trainer._get_train_sampler()
        
        print(f"Updated dataset: {len(filtered_dataset)} examples remaining (removed {len(self.removed_examples)})")
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step to track example rewards."""
        # This would need to be implemented to track rewards per example
        # The exact implementation depends on how GRPO provides example-level rewards
        pass
    
    def set_trainer(self, trainer):
        """Set the trainer reference for dataset updates."""
        self.trainer = trainer
    
    def track_example_reward(self, example_id, reward):
        """Track reward for a specific example."""
        assert example_id not in self.removed_examples
        if example_id not in self.example_rewards:
            self.example_rewards[example_id] = 0
            self.example_counts[example_id] = 0
        
        self.example_rewards[example_id] = reward
        self.example_counts[example_id] += 1

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
          dataset_name: str = None,
          use_curriculum: bool = False,
          curriculum_threshold: float = 0.25):
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

    # Always add example_id column for curriculum learning
    dataset = dataset.add_column("example_id", list(range(len(dataset))))
    
    # Set up callbacks
    callbacks = [CumulativeSuccessCallback()]
    
    # Add curriculum learning callback if enabled
    curriculum_callback = None
    if use_curriculum and not just_get_data_order:
        curriculum_callback = CurriculumLearningCallback(
            reward_threshold=curriculum_threshold,
            save_steps=len(dataset)
        )
        callbacks.append(curriculum_callback)
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_fn,
        ],
        args=config,
        train_dataset=dataset,
        callbacks=callbacks,
    )
    
    # Set trainer reference for curriculum callback
    if curriculum_callback is not None:
        curriculum_callback.set_trainer(trainer)
        
        # Create a curriculum-aware reward function
        original_reward_fn = reward_fn
        def curriculum_reward_fn(completions, answer, **kwargs):
            rewards = original_reward_fn(completions, answer, **kwargs)
            
            # Track rewards per example if we have example IDs
            if 'example_id' in kwargs:
                example_ids = kwargs['example_id']
                if isinstance(example_ids, (list, tuple)):
                    for example_id, reward in zip(example_ids, rewards):
                        curriculum_callback.track_example_reward(example_id, reward)
                else:
                    # Single example case
                    curriculum_callback.track_example_reward(example_ids, rewards[0] if len(rewards) > 0 else 0)
            
            return rewards
        
        # Replace the reward function in the trainer
        trainer.reward_funcs = [curriculum_reward_fn]
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
@click.option('--use_curriculum', is_flag=True, default=False, help="Enable curriculum learning")
@click.option('--curriculum_threshold', type=float, default=0.25, help="Reward threshold for removing examples in curriculum learning")
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
    use_curriculum: bool,
    curriculum_threshold: float
):
    name = f'{num_generations}gen_{max_steps}steps_{model_name}'.replace('/','-')
    if strategy is not None:
        name += f'_strategy{strategy}'
    if subset_perc is not None:
        name += f'_subsetperc{subset_perc}'
    if use_curriculum:
        name += f'_curriculum{curriculum_threshold}'
    
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
            reward_fn=create_reward_func(dataset_name),
            just_get_data_order=just_get_order,
            dataset_name=dataset_name,
            use_curriculum=use_curriculum,
            curriculum_threshold=curriculum_threshold
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