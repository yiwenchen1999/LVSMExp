#!/bin/bash
# Benchmark: single-scene relight + 100 novel views
# Measures per-stage timing and peak GPU/CPU memory.

torchrun --nproc_per_node 1 --nnodes 1 \
  --rdzv_id 28635 --rdzv_backend c10d --rdzv_endpoint localhost:29507 \
  inference_editor_benchmark.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
  training.dataset_path = /data/polyhaven_lvsm/test/full_list.txt \
  training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4_singleMap \
  training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
  training.batch_size_per_gpu = 1 \
  training.target_has_input = false \
  training.num_views = 12 \
  training.square_crop = true \
  training.num_input_views = 4 \
  training.num_target_views = 8 \
  inference.if_inference = true \
  inference.compute_metrics = false \
  inference.render_video = false \
  inference.same_pose = True \
  inference.benchmark_num_views = 100 \
  inference_out_dir = experiments/benchmark_relight
