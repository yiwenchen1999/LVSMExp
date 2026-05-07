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
python -m pip install -r requirements.txt

mkdir ckpt/dpt_decoder_256
rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  chen.yiwe@xfer.discovery.neu.edu:/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/dense_relight_env_progressive/ckpt_0000000000026000.pt  \
  ckpts/dpt_decoder_256/
