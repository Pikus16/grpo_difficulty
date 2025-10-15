#!/bin/bash
# Setup script for Reasoning Gym training environment
# Based on reasoning-gym/training/README.md

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Reasoning Gym Training Environment Setup                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Current directory: $SCRIPT_DIR${NC}"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"

# Check if we're in a conda environment
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${GREEN}✓ Conda environment active: $CONDA_DEFAULT_ENV${NC}"
else
    echo -e "${YELLOW}⚠ No conda environment detected${NC}"
    echo "  It's recommended to use a conda environment"
    echo "  Run: conda activate grpo"
fi
echo ""

# Clone Reasoning Gym if not exists
if [ ! -d "reasoning-gym" ]; then
    echo "📦 Cloning Reasoning Gym repository..."
    git clone https://github.com/open-thought/reasoning-gym.git
    echo -e "${GREEN}✓ Reasoning Gym cloned${NC}"
else
    echo -e "${GREEN}✓ Reasoning Gym directory already exists${NC}"
fi
echo ""

# Install basic dependencies
echo "📦 Installing basic dependencies (wheel, fire)..."
pip install wheel fire
echo -e "${GREEN}✓ Basic dependencies installed${NC}"
echo ""

# Install Reasoning Gym
echo "📦 Installing Reasoning Gym..."
cd reasoning-gym/
pip install -e .
cd ..
echo -e "${GREEN}✓ Reasoning Gym installed${NC}"
echo ""

# Install verl
echo "📦 Installing verl (specific commit for compatibility)..."
echo "   This may take a few minutes..."
pip install git+https://github.com/volcengine/verl.git@c34206925e2a50fd452e474db857b4d488f8602d
echo -e "${GREEN}✓ verl installed${NC}"
echo ""

# Install flash-attn
echo "📦 Installing flash-attention..."
echo "   This may take several minutes to compile..."
pip install flash-attn==2.7.3 --no-build-isolation
echo -e "${GREEN}✓ flash-attention installed${NC}"
echo ""

# Install other dependencies if missing
echo "📦 Checking other dependencies..."
pip install -q wandb huggingface_hub transformers datasets torch vllm --upgrade
echo -e "${GREEN}✓ Other dependencies installed/updated${NC}"
echo ""

# Check if HF is logged in
echo "🔑 Checking Hugging Face login..."
if python -c "from huggingface_hub import HfFolder; exit(0 if HfFolder.get_token() else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ Hugging Face is configured${NC}"
else
    echo -e "${YELLOW}⚠ Hugging Face not logged in${NC}"
    echo "   Please run: huggingface-cli login"
fi
echo ""

# Check if W&B is logged in
echo "🔑 Checking Weights & Biases login..."
if python -c "import wandb; exit(0 if wandb.api.api_key else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ W&B is configured${NC}"
else
    echo -e "${YELLOW}⚠ W&B not logged in${NC}"
    echo "   Please run: wandb login"
fi
echo ""

# Make training script executable
if [ -f "train_reasoning_gym.py" ]; then
    chmod +x train_reasoning_gym.py
    echo -e "${GREEN}✓ train_reasoning_gym.py is executable${NC}"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. If not already done, login to services:"
echo "   $ huggingface-cli login"
echo "   $ wandb login"
echo ""
echo "2. Verify your setup:"
echo "   $ python train_reasoning_gym.py check"
echo ""
echo "3. List available configs:"
echo "   $ python train_reasoning_gym.py list-configs"
echo ""
echo "4. Run your first training:"
echo "   $ python train_reasoning_gym.py train --config-name algorithmic_qwen_3b --n-gpus 2 --tensor-parallel-size 1"
echo ""
echo "5. Read the full guide:"
echo "   $ cat REASONING_GYM_TRAINING.md"
echo ""
echo -e "${GREEN}Happy training! 🚀${NC}"

