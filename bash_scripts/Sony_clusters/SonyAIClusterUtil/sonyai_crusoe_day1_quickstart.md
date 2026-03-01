# SonyAI Crusoe Slurm – Day‑1 Quick Start

This is a concise checklist-style summary of the SonyAI Crusoe cluster onboarding and basic usage.

---

## 1. First‑time account setup (do once)

SSH into the login node (recommended):

```bash
ssh mfml1
```

Enable Crusoe Slurm helper tools by adding this to `~/.bashrc`:

```bash
export SLURM_USER_TOOL_ROOT=/usr/local/share/slurm-user/
if [ -d $SLURM_USER_TOOL_ROOT ]; then
  export PATH="${SLURM_USER_TOOL_ROOT}:$PATH"
  source ${SLURM_USER_TOOL_ROOT}/slurm-crusoe.bash_profile
fi
```

Reload:
```bash
source ~/.bashrc
```

This enables commands like `sua`, `sq`, `sip`, `sig`, `sbash`, `slog`.

---

## 2. Where to work & storage best practices

Recommended directories:

- `/home/$USER` – small scripts, configs (slow, quota-limited)
- `/scratch2/$USER` – fast personal scratch (HF cache, temp files)
- `/group2/ct` – shared team datasets and checkpoints

Suggested environment variables:

```bash
export HF_HOME=/scratch2/$USER/hf_cache
export TORCH_HOME=/scratch2/$USER/torch_cache
export WANDB_DIR=/scratch2/$USER/wandb
```

---

## 3. Checking what resources you have

Check accounts and partitions:
```bash
sua
```

Check cluster status:
```bash
sig   # GPU availability
sip   # partition status
sin   # node-level detail
```

Rules of thumb:
- Use `ct` / `ct_l40s` when possible (priority)
- Use `sharedp` only for resumable jobs

---

## 4. Running jobs with `srun` (quick tests)

Quick GPU test:

```bash
srun --partition=ct --account=ct --gres=gpu:1 nvidia-smi
```

Python CUDA test:

```bash
srun --partition=ct --account=ct --gres=gpu:1 python - << 'EOF'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF
```

---

## 5. Running jobs with `sbatch` (real training)

Minimal single‑GPU batch script:

```bash
#!/bin/bash
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_debug
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err

mkdir -p slurm_logs
python train.py
```

Submit with:
```bash
sbatch train.sh
```

For shared partitions, add:
```bash
#SBATCH --requeue
```

---

## 6. Interactive GPU sessions

Use `sbash` (interactive shell on a compute node):

```bash
sbash --partition=ct --account=ct --nodes=1 --gpus=1
```

Always exit when done:
```bash
exit
```

---

## 7. Mounting / accessing data correctly

No manual mounting required. Filesystems are globally available:

- `/group2/ct` – shared persistent data
- `/scratch2/$USER` – fast personal storage
- `$SLURM_SCRATCH` – job‑local NVMe (fastest, auto‑deleted)

Inside a job:
```bash
cp -r /group2/ct/datasets/mydata $SLURM_SCRATCH/
```

Train from `$SLURM_SCRATCH` for best I/O performance.

---

## 8. Minimal day‑1 sanity checklist

- `sua` shows `ct` account
- `srun --partition=ct --gres=gpu:1 nvidia-smi` works
- Logs are written correctly
- HF / Torch caches point to `/scratch2`
- Checkpointing works (important for `sharedp`)

---

**You are ready to run real experiments on SonyAI Crusoe.**
