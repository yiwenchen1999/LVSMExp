cd /music-shared-disk/group/ct/yiwen/codes/LVSMExp

sbash --partition=ct --account=ct --nodes=1 --gpus=1
sbash --partition=ct_l40s --account=ct --nodes=1 --gpus=1
sbash --partition=sharedp --account=ct --nodes=1 --gpus=1


# ckpt pull: explorer -> sony:
rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  chen.yiwe@xfer.discovery.neu.edu:/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/dense_relight_env/ckpt_0000000000026000.pt  \
  ckpt/dense_relight_env/
