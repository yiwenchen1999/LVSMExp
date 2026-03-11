#!/bin/bash
# Render ALL camera poses of each scene and export individual frames + video.
# Requires batch_size=1 (enforced automatically) because frame counts vary per scene.

torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    inference_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense.yaml \
    training.batch_size_per_gpu = 1 \
    training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense \
    training.LVSM_checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
    training.dataset_path = data_samples/objaverse_processed_with_envmaps/test/full_list.txt \
    training.num_input_views = 4 \
    training.square_crop = true \
    inference.if_inference = true \
    inference.compute_metrics = true \
    inference.render_all_views = true \
    inference.view_chunk_size = 4 \
    inference.view_idx_file_path = data/evaluation_index_objaverse.json \
    inference_out_dir = experiments/evaluation/render_all_views
