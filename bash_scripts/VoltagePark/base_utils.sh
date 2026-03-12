# rsync data from neu to voltage park
rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  chen.yiwe@xfer.discovery.neu.edu:/scratch/chen.yiwe/temp_objaverse/lvsm_scenes_dense \
    .

rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  ubuntu@147.185.41.15:/home/ubuntu/LVSMExp/experiments/evaluation/combined_scenes \
    result_previews

rsync -avz ubuntu@147.185.41.15:/home/ubuntu/LVSMExp/experiments/evaluation/demo_scene_env_variations demo_material/

rsync -avh --partial --inplace --progress \
  -e "ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes" \
  ckpt_0000000000010000.pt \
  chen.yiwe@xfer.discovery.neu.edu:/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp/ckpt/relight_dense_env_scene

python3 preprocess_scripts/update_paths.py \
  --old-path /music-shared-disk/group/ct/yiwen/data/objaverse \
  --new-path /data \
  --root-dir /data/polyhaven_lvsm/test \
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
  --full-list /data/lvsmPlus_objaverse/test/full_list.txt
 
scp ubuntu@147.185.41.15:~/Neural_Gaffer_exp/polyhaven_relighting_results result_previews/NeuralGaffer
rsync -avz --progress ubuntu@147.185.41.15:~/Neural_Gaffer_exp/polyhaven_relighting_results result_previews/NeuralGaffer

tmux new -s finetune# 默认路径：result_previews/eval/gt_samples 和 result_previews/eval/lightSwitch



python scripts/eval_gt_vs_lightswitch_masked.py result_previews/eval/gt_samples eval/lightSwitch --all 
# 在 tmux 里运行
cd ~/LVSMExp
source /data/venv/lvsmexp/bin/activate
bash bash_scripts/VoltagePark/finetune_objaverse.sh

# 断开会话（保留运行）：Ctrl+B 然后按 D
# 之后重连 SSH 并恢复会话：
tmux attach -t finetune

#preprocess stanford orb
python3 preprocess_scripts/preprocess_stanford_orb.py \
  --input-root /data/dataset \
  --output-root /data/stanford_ORB_processed \
  --split both \
  --target-size 512 \

python3 preprocess_scripts/preprocess_obj_with_light.py \
  --input-root /data/dataset \
  --output-root /data/obj_with_light_processed \
  --split test \
  --target-size 512

python3 preprocess_scripts/preprocess_obj_with_light.py \
  --crop-mode square \
  --target-size 512 \
  --input-root /data/dataset \
  --output-root /data/obj_with_light_processed \
  --split test


  python3 scripts/fetch_source_data.py \
    --metadata_dir  metadata_polyhaven \
    --dataset_root  /data/polyhaven_lvsm/test \
    --out_dir      source_data_polyhaven


python3 preprocess_scripts/preprocess_tenIR.py \
    --input /data/benchmark_tensoIR \
    --output /data/benchmark_tensoIR_processed \
    --hdri-dir /data/high_res_envmaps_1k \
    --split test



python preprocess_scripts/create_env_variations_from_processed.py \
    --data-root /data/lvsm_scenes_dense \
    --output-root /data/lvsm_scenes_dense_env_variations \
    --split test \
    --n-variations 100 \
    --scene-list /data/lvsm_scenes_dense/test/full_list_scenes.txt

torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
exp_rotate_env.py --config "configs/LVSM_scene_encoder_decoder_wEditor_envmap_pointlight.yaml" \
training.dataset_path = "/data/lvsm_scenes_dense_env_variations/test/full_list.txt" \
training.checkpoint_dir = ckpt/relight_combined_scenes \
training.batch_size_per_gpu = 1 \
training.target_has_input = false \
training.num_views = 16 \
training.square_crop = true \
training.num_input_views = 8 \
training.num_target_views = 8 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = false \
inference.view_idx_file_path = "data/demo_scene_rotate_dense_env_variations.json" \
inference_out_dir = experiments/evaluation/demo_scene_env_variations
