torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
  inference_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
  training.dataset_path = /data/polyhaven_lvsm/test/full_list_demo.txt \
  training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4_singleMap \
  training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
  training.batch_size_per_gpu = 4 \
  training.target_has_input = false \
  training.num_views = 48 \
  training.square_crop = true \
  training.num_input_views = 16 \
  training.num_target_views = 32 \
  inference.if_inference = true \
  inference.compute_metrics = true \
  inference.render_video = false \
  inference.condition_reverse = true \
  inference.all_views = true \
  inference.view_idx_file_path = data/evaluation_index_polyhaven_demo.json \
  inference_out_dir = experiments/evaluation/demo_scenes
