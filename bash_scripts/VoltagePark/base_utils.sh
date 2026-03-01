# rsync data from neu to voltage park
rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  chen.yiwe@xfer.discovery.neu.edu:/scratch/chen.yiwe/temp_objaverse/lvsm_scenes/test \
    lvsm_scenes/


rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  ckpt_0000000000026000.pt \
  chen.yiwe@xfer.discovery.neu.edu:/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/dense_relight_env
python3 preprocess_scripts/update_paths.py \
  --old-path /scratch/chen.yiwe/temp_objaverse \
  --new-path /data \
  --root-dir /data/lvsm_scenes/test \
  --extensions json txt \
  --backup

export FULL_LIST_TEST=/data/lvsmPlus_objaverse/test/full_list.txt
export PREVIEW_OUTPUT=/data/lvsmPlus_objaverse/test/preview.png

python3 preprocess_scripts/preview_scenes.py \
  --full-list $FULL_LIST_TEST \
  --output preview_scenes/preview.png \
  --image-idx 64 \
  --grid-cols 8 \
  --grid-rows 4 \
  --images-per-grid 32

python3 preprocess_scripts/remove_broken_scenes.py \
  --broken-scene preview_scenes/broken_scene.txt \
  --full-list /data/lvsmPlus_objaverse/train/full_list.txt
 
 scp ubuntu@147.185.41.15:/home/ubuntu/LVSMExp/ckpt/LVSM_object_encoder_decoder_sparse/iter_00000001/supervision_2cb7026ffacf4889bb265099b8460c7f_env_2.jpg bash_scripts/VoltagePark/previews/

tmux new -s finetune

# 在 tmux 里运行
cd ~/LVSMExp
source /data/venv/lvsmexp/bin/activate
bash bash_scripts/VoltagePark/finetune_objaverse.sh

# 断开会话（保留运行）：Ctrl+B 然后按 D
# 之后重连 SSH 并恢复会话：
tmux attach -t finetune