# GRPO Difficulty

A project to analyze GRPO task difficulty and performance.

## Setup Instructions

Follow the steps below to set up the environment:

### 1. Clone the repository

```bash
git clone https://github.com/Pikus16/grpo_difficulty
cd grpo_difficulty
```

### 2 Create the Conda environment

Make sure you have Conda installed. If you dont, run:
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash ~/Miniconda3-latest-Linux-x86_64.sh`

```bash
conda env create -f environment.yml
conda activate grpo
pip install flash-attn==2.7.4.post1 --no-build-isolation
```