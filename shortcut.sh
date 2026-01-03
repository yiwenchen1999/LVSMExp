# env setup
cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate


python process_data.py --base_path re10k --output_dir re10k_processed --mode test

python preprocess_scripts/preprocess_objaverse.py \
    --input data_samples/sample_objaverse \
    --output data_samples/objaverse_processed \
    --split test

python preprocess_scripts/create_evaluation_index.py \
    --full-list data_samples/objaverse_processed/test/full_list.txt \
    --output data/evaluation_index_objaverse.json \
    --n-input 2 \
    --n-target 3 \
    --seed 42

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