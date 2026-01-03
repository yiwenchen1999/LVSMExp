python process_data.py --base_path re10k --output_dir re10k_processed --mode test

torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
inference.py --config "configs/LVSM_scene_encoder_decoder.yaml" \
training.dataset_path = "./re10k_processed/test/full_list.txt" \
training.batch_size_per_gpu = 4 \
training.target_has_input =  false \
training.num_views = 5 \
training.square_crop = true \
training.num_input_views = 2 \
training.num_target_views = 3 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = true \
inference_out_dir = ./experiments/evaluation/test