# env setup
cd /Users/yiwenchen/Desktop/ResearchProjects/scripts
source venv/bin/activate

cd /projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/LVSMExp
conda activate /projects/vig/yiwenc/all_env/rayzer

# data preprocessing
# re10k
python process_data.py --base_path re10k --output_dir re10k_processed --mode test

# objaverse
python preprocess_scripts/preprocess_objaverse.py \
    --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense \
    --output /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_format \
    --split test
python preprocess_scripts/create_evaluation_index.py \
    --full-list /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_format/test/full_list.txt \
    --output data/evaluation_index_objaverse_test.json \
    --n-input 4 \
    --n-target 8 \
    --seed 42

python preprocess_scripts/preprocess_objaverse.py \
    --input data_samples/sample_objaverse \
    --output data_samples/objaverse_processed \
    --split train
python preprocess_scripts/create_evaluation_index.py \
    --full-list data_samples/objaverse_processed/test/full_list.txt \
    --output data/evaluation_index_objaverse_test_4i3o.json \
    --n-input 4 \
    --n-target 3 \
    --seed 42

# preprocess objaverse with envmaps
python preprocess_scripts/preprocess_objaverse.py \
  --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense \
  --output /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps \
  --split train \
  --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs \
  --max-objects 10

  # preprocess objaverse with envmaps
python preprocess_scripts/preprocess_objaverse.py \
  --input data_samples/sample_objaverse \
  --output data_samples/objaverse_processed_with_envmaps \
  --split train \
  --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs \
  --max-objects 10

# preview the scenes:
python preprocess_scripts/preview_scenes.py \
    --full-list /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps/test/full_list.txt \
    --output scene_preview/preview.png \
    --image-idx 64 \
    --grid-cols 8 \
    --grid-rows 4 \
    --images-per-grid 32
# visualize data
python test_data_visualization.py --config configs/LVSM_scene_encoder_decoder_wEditor.yaml --output-dir ./test_output --subsample 10 --env-sample-num 5000 --batch-idx 8
# train-og:
torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train.py --config configs/LVSM_scene_encoder_decoder.yaml \
    training.batch_size_per_gpu = 8 \
    training.checkpoint_dir = ckpt/LVSM_object_encoder_decoder_sparse

# train-editor, objaverse
torchrun --nproc_per_node 1 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29502 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_overfit.yaml \
    training.batch_size_per_gpu = 4 \
    training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_overfit \
    training.LVSM_checkpoint_dir = ckpt/LVSM_object_encoder_decoder \
    training.dataset_path = data_samples/objaverse_processed_with_envmaps/train/full_list.txt


find /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps/train/images -maxdepth 1 -type d | wc -l

# inference
# base, re10k
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
inference.view_idx_file_path = "./data/evaluation_index_re10k.json" \
inference_out_dir = ./experiments/evaluation/test

# finetune, objaverse
torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
inference_editor.py --config "configs/LVSM_scene_encoder_decoder_wEditor.yaml" \
training.dataset_path = "./data_samples/objaverse_processed/test/full_list.txt" \
training.batch_size_per_gpu = 4 \
training.target_has_input =  false \
training.num_views = 7 \
training.square_crop = true \
training.num_input_views = 4 \
training.num_target_views = 3 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = true \
inference.view_idx_file_path = "./data/evaluation_index_objaverse_test_4i3o.json" \
inference_out_dir = ./experiments/evaluation/test_obj_4i3o_wEditorTest