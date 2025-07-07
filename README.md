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

### 3. Runs

Each dataset has a separate run. Below are the instructions for each

#### GSM8K

Both train and eval should be run in `gsm8k` directory.
To run both train and test, run:

`python train_gsm8k.py --difficulty_level DIFFICULTY_LEVEL`

To run just test, run:

`python get_answers.py -k 1 --difficulty_level 0`

Both have arguments that can be overwritten.