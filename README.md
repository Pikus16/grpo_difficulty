# GRPO Difficulty

A project to analyze GRPO task difficulty and performance.

## Setup Instructions

Follow the steps below to set up the environment:

### 1. Clone the repository

```bash
git clone https://github.com/Pikus16/grpo_difficulty
cd grpo_difficulty
```

### 2. Create the Conda environment

Make sure you have Conda installed. If you dont, run:
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash ~/Miniconda3-latest-Linux-x86_64.sh`

```bash
conda env create -f environment.yml
conda activate grpo
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### 3. Train on Subset

To train on a dataset, run the below command:

`python src/train.py --dataset_name DATASET_NAME -m MODEL_NAME --strategy STRATEGY_NAME`

This will do both a single training run on the specified dataset subset, and then evaluate every checkpoint.

As an example:

`python src/train.py --dataset_name shuffleobj -m unsloth/phi-4-bnb-4bit --strategy singlehard`

If you just want to run eval, you can add the flag `--skip_train`. Note this is fairly opinionated on where checkpoints are located, and should only be run if a train run finished.

There are more arguments, below is the output of `python train.py --help`

```
Usage: train.py [OPTIONS]

Options:
  --dataset_name TEXT            [required]
  --strategy TEXT                Strategy to use for selection (if none will
                                 use whole dataset)
  --subset_perc FLOAT            Percentage of the dataset to grab as subset
                                 (if none will use whole dataset)
  --project TEXT
  -n, --num_generations INTEGER  Number of generations per iteration
  -m, --model-name TEXT          Model name or path  [default:
                                 unsloth/Qwen3-4B-unsloth-bnb-4bit]
  --max_steps INTEGER            Number of generations per iteration
  --save_steps INTEGER           How often to save
  -l, --load_4bit                Load model in 4-bit mode (flag)
  --skip_train                   Skip training and directly evaluate
  --eval_last                    Only evaluate last checkpoint
  --just_get_order               Only save the order of train data samples
  --help                         Show this message and exit.
```