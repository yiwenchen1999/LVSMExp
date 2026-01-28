export DATA_LIST="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/data_samples/objaverse_processed_with_envmaps/test/full_list.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"

singularity exec --nv $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
  cd $PROJ

  python3 preprocess_scripts/create_evaluation_index.py \
    --full-list data_samples/objaverse_processed_with_envmaps/test/full_list.txt \
    --output data/evaluation_index_objaverse_dense_samples.json \
    --n-input 4 \
    --n-target 8 \
    --max-scenes 125 \
    --seed 42
"

export DATA_LIST="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/data_samples/objaverse_processed_with_envmaps/test/full_list.txt"
export CKPT_DIR="/music-shared-disk/group/ct/yiwen/codes/LVSMExp/ckpt/LVSM_scene_encoder_decoder"

singularity exec --nv $BIND $SIF bash -lc "
  set -euo pipefail
  export PYTHONPATH=\"$PY_SITE:${PYTHONPATH:-}\"
  cd $PROJ

  torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
    inference.py --config configs/LVSM_scene_encoder_decoder.yaml \
    training.dataset_path = \"$DATA_LIST\" \
    training.batch_size_per_gpu = 4 \
    training.target_has_input = false \
    training.num_views = 12 \
    training.square_crop = true \
    training.num_input_views = 4 \
    training.num_target_views = 8 \
    training.checkpoint_dir = \"$CKPT_DIR\" \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_video = false \
    inference.view_idx_file_path = \"./data/evaluation_index_objaverse_dense.json\" \
    inference_out_dir = ./experiments/evaluation/test_dense_reconstruction
"
