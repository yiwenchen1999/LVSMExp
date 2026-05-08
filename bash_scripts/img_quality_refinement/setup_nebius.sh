# 1) 添加 deadsnakes（提供多版本 Python）
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update

# 2) 安装 Python 3.10 + venv
sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils

# 3) 验证
python3.10 --version

python3.10 -m venv ~/venv/lvsmexp310
source ~/venv/lvsmexp310/bin/activate
git clone https://github.com/yiwenchen1999/LVSMExp.git
cd LVSMExp

python -m pip install -U pip setuptools wheel
# B200 (sm_100) needs newer CUDA/PyTorch wheels than cu118.
# 2) 先装项目其余依赖（当前 requirements 已去掉 torch 固定）
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
# 3) 最后强制装“同一来源、同一 CUDA 通道”的 torch 三件套 + xformers
python -m pip install --upgrade --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio
python -m pip install --upgrade --force-reinstall --no-cache-dir xformers lpips

python - <<'PY'
import torch
print("torch", torch.__version__)
print("compiled_cuda", torch.version.cuda)
print("arch_list", torch.cuda.get_arch_list())
PY

mkdir ckpt/dpt_decoder_256
rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  chen.yiwe@xfer.discovery.neu.edu:/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/dense_relight_env_progressive/ckpt_0000000000026000.pt  \
  ckpt/relight_finetune/

# filetransfer
rsync -avhP --partial --append-verify --info=progress2 --mkpath \
  -e "ssh -T" \
  chen.yiwe@xfer.discovery.neu.edu:/scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse/test/envmaps/ \
  /mnt/data-disk/lvsmPlus_objaverse/test/envmaps/


# mount disks:
# Format it (this will erase anything on it - fine for a new disk)
sudo mkfs.ext4 /dev/vdc

# Create mount point
sudo mkdir -p /mnt/data-disk

# Mount it
sudo mount /dev/vdc /mnt/data-disk

# Give yourself access
sudo chmod a+w /mnt/data-disk