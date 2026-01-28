# Python Environment Setup on Sony Cluster

This guide explains how to create and manage Python virtual environments on the Sony AI Crusoe cluster.

## Storage Locations

Based on the cluster README, here are the recommended locations for Python environments:

| Location | Size | Speed | Recommended For |
|----------|------|-------|-----------------|
| `/scratch2/$USER` | 100GB/user | Fast | Personal environments, pip cache |
| `/home/$USER` | 100GB/user | Slow | Small environments, configs |
| `/music-shared-disk/group/ct/yiwen` | 5TB/team | Fast | Shared team environments |

**Recommendation**: Use `/scratch2/$USER` for personal environments (faster I/O).

## Method 1: Python venv (Recommended)

### Create Virtual Environment

```bash
# SSH to cluster
ssh mfml1

# Navigate to your project directory
cd /music-shared-disk/group/ct/yiwen/codes/FLUX_finetune

# Create virtual environment in scratch space (fast)
python3 -m venv /scratch2/$USER/venv/flux_finetune

# Or create in project directory
python3 -m venv venv

# Activate the environment
source /scratch2/$USER/venv/flux_finetune/bin/activate
# OR
source venv/bin/activate
```

### Install Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install your requirements
pip install -r requirements.txt

# Or install specific packages
pip install torch torchvision diffusers transformers accelerate
```

### Deactivate Environment

```bash
deactivate
```

## Method 2: Conda/Mamba (If Available)

If conda or mamba is available on the cluster:

```bash
# Create conda environment
conda create -n flux_finetune python=3.10

# Activate
conda activate flux_finetune

# Install packages
conda install pytorch torchvision -c pytorch
pip install diffusers transformers accelerate
```

## Method 3: Using Singularity Containers

For reproducible environments, consider using Singularity containers (see README for Singularity tutorial).

## Best Practices

### 1. Environment Location

**For personal use:**
```bash
# Fast storage, good for active development
/scratch2/$USER/venv/project_name
```

**For shared team use:**
```bash
# Shared with team members
/music-shared-disk/group/ct/yiwen/venv/project_name
```

**For project-specific:**
```bash
# In project directory (convenient but slower if in /home)
/music-shared-disk/group/ct/yiwen/codes/FLUX_finetune/venv
```

### 2. Environment Variables

Add to your `~/.bashrc` or `~/.bash_profile`:

```bash
# Python environment
export VENV_DIR=/scratch2/$USER/venv

# Cache directories (recommended by README)
export HF_HOME=/scratch2/$USER/hf_cache
export TORCH_HOME=/scratch2/$USER/torch_cache
export WANDB_DIR=/scratch2/$USER/wandb
export XDG_CACHE_HOME=/scratch2/$USER/.cache
```

### 3. Activate in Slurm Jobs

In your `sbatch` scripts, activate the environment:

```bash
#!/bin/bash
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err

# Activate virtual environment
source /scratch2/$USER/venv/flux_finetune/bin/activate

# Or if in project directory
source /music-shared-disk/group/ct/yiwen/codes/FLUX_finetune/venv/bin/activate

# Your training command
python train.py
```

### 4. Create Environment Script

Create a setup script `setup_env.sh`:

```bash
#!/bin/bash
# Create and setup Python environment for FLUX finetune

ENV_DIR="/scratch2/$USER/venv/flux_finetune"
PROJECT_DIR="/music-shared-disk/group/ct/yiwen/codes/FLUX_finetune"

# Create virtual environment
echo "Creating virtual environment at $ENV_DIR..."
python3 -m venv $ENV_DIR

# Activate
source $ENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements if exists
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r $PROJECT_DIR/requirements.txt
else
    echo "No requirements.txt found. Installing basic packages..."
    pip install torch torchvision diffusers transformers accelerate wandb
fi

echo "Environment created! Activate with:"
echo "source $ENV_DIR/bin/activate"
```

Make it executable and run:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

## Quick Reference

```bash
# Create environment
python3 -m venv /scratch2/$USER/venv/flux_finetune

# Activate
source /scratch2/$USER/venv/flux_finetune/bin/activate

# Install packages
pip install package_name

# Save requirements
pip freeze > requirements.txt

# Deactivate
deactivate

# Remove environment (if needed)
rm -rf /scratch2/$USER/venv/flux_finetune
```

## Troubleshooting

### Python Version

Check available Python versions:
```bash
python3 --version
which python3
```

### Permission Issues

If you get permission errors, ensure you're using your user space:
```bash
# Use /scratch2/$USER instead of system directories
python3 -m venv /scratch2/$USER/venv/myenv
```

### Disk Space

Check available space:
```bash
df -h /scratch2/$USER
```

### Environment Not Found in Slurm Jobs

Make sure to use absolute paths in `sbatch` scripts:
```bash
# Good: absolute path
source /scratch2/$USER/venv/flux_finetune/bin/activate

# Bad: relative path (may not work in jobs)
source venv/bin/activate
```

## Example: Complete Setup for FLUX Finetune

```bash
# 1. SSH to cluster
ssh mfml1

# 2. Navigate to project
cd /music-shared-disk/group/ct/yiwen/codes/FLUX_finetune

# 3. Create environment in fast storage
python3 -m venv /scratch2/$USER/venv/flux_finetune

# 4. Activate
source /scratch2/$USER/venv/flux_finetune/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate wandb

# 6. Verify installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 7. Save requirements
pip freeze > requirements.txt
```

