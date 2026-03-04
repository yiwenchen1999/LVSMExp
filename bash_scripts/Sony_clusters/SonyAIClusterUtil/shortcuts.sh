# login to the cluster
ssh mfml1
# cd to the project root
cd /music-shared-disk/group/ct/yiwen/codes/FLUX_finetune
# cd to the data root

# sync data directly from neu to sony
# Note: This will resume partial transfers and skip already-transferred files
cd /music-shared-disk/group/ct/yiwen/data/objaverse
rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  chen.yiwe@xfer.discovery.neu.edu:/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/dense_relight_env/ckpt_0000000000026000.pt  \
  ckpt/dense_relight_env/

rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  /music-shared-disk/group/ct/yiwen/data/objaverse/lvsm_scenes_dense \
  chen.yiwe@xfer.discovery.neu.edu:/scratch/chen.yiwe/temp_objaverse/
  

# 直接运行命令
rsync -avzP --human-readable \
  mfml1:/music-shared-disk/group/ct/yiwen/codes/LVSMExp/experiments/evaluation/polyhaven_dense_inference \
  result_previews/
rsync -avzP --human-readable \
  chen.yiwe@xfer.discovery.neu.edu:/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/experiments/evaluation/test_scenes_dense \
  result_previews/
# env creation:
sbash --partition=ct --account=ct --nodes=1 --gpus=2
sbash --partition=ct_l40s --account=ct --nodes=1 --gpus=1
sbash --partition=sharedp --account=ct --nodes=1 --gpus=1

export PROJ=/music-shared-disk/group/ct/yiwen/codes/FLUX_finetune   # or Neural_Gaffer later
export PY_SITE=/scratch2/$USER/ng_py
export PIP_CACHE_DIR=/scratch2/$USER/cache/pip
export SIF_DIR=/scratch2/$USER/singularity_images
export SIF=$SIF_DIR/pytorch_24.01-py3.sif
export BIND="-B /group2,/scratch2,/music-shared-disk"
mkdir -p "$PY_SITE" "$PIP_CACHE_DIR"
singularity exec --nv $BIND $SIF bash -lc "
python3 -m pip install -U pip wheel setuptools --cache-dir $PIP_CACHE_DIR
"
singularity exec --nv $BIND $SIF bash -lc "
python3 -m pip install -r $PROJ/requirements.txt --target $PY_SITE --cache-dir $PIP_CACHE_DIR
"
singularity exec --nv $BIND $SIF bash -lc "
export PYTHONPATH=$PY_SITE:\$PYTHONPATH
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
"



#env creation:
# See python_env_setup.md for detailed instructions
# Quick reference:
#   python3 -m venv /scratch2/$USER/venv/flux_finetune
#   source /scratch2/$USER/venv/flux_finetune/bin/activate

# request interactive node
sbash --partition=ct --account=ct --nodes=1 --gpus=1
# sbatch the job
# small job:
srun --partition=ct --account=ct --gres=gpu:1 \
     nvidia-smi
#full command:

#srun with venv:
SRUN_OPTS="--account=ct --partition=ct --gres=gpu:1 --job-name=ng_test"
SIF=/scratch2/$USER/singularity_images/pytorch_24.01-py3.sif
BIND="-B /group2,/scratch2,/music-shared-disk"

srun ${SRUN_OPTS} singularity exec --nv ${BIND} ${SIF} \
  bash -lc "source /scratch2/$USER/ng_venv/bin/activate && cd /group2/ct/yiwen/src/Neural_Gaffer && python -c 'import torch; print(torch.cuda.get_device_name(0))'"

#!/bin/bash
#SBATCH --partition=ct
#SBATCH --account=ct
#SBATCH --job-name=train_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x.%j.out
#SBATCH --error=slurm_logs/%x.%j.err
mkdir -p slurm_logs
python train.py

# transfer data to the cluster(neu to SONY)

# setup env

#############################################
# Git Configuration - Avoid password prompts
#############################################
# See github_auth.md for detailed instructions on setting up SSH keys
# Quick reference:
#   - Switch to SSH: git remote set-url origin git@github.com:yiwenchen1999/FLUX_finetune.git
#   - Test connection: ssh -T git@github.com
#   - Display public key: cat ~/.ssh/id_ed25519.pub

#post-process stanford-ORB::
cd /music-shared-disk/group/ct/yiwen/codes/LVSMExp
singularity exec --nv $BIND $SIF bash -lc "
python preprocess_scripts/preprocess_stanford_orb.py \
  --input-root /music-shared-disk/group/ct/yiwen/data/blender_LDR \
  --output-root /music-shared-disk/group/ct/yiwen/data/stanford_ORB_lvsm \
  --env-gt-root /music-shared-disk/group/ct/yiwen/data/ground_truth \
  --n-context 4 \
  --n-target 8
"

#iterative relighting:
SCENE_IDX=0 NUM_ITER=50 NUM_INPUT_VIEWS=4 bash bash_scripts/Sony_clusters/SonyAIClusterUtil/interactive_degrade_test.sh
