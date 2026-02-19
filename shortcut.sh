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
    --input data_samples/objaverse_with_pointLights \
    --output data_samples/processed_objaverse_with_pointLights \
    --split test
python preprocess_scripts/create_evaluation_index.py \
    --full-list /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps/test/full_list.txt \
    --output data/evaluation_index_objaverse_dense.json \
    --n-input 4 \
    --n-target 8 \
    --max-scenes 100 \
    --seed 42

python preprocess_scripts/preprocess_objaverse.py \
    --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense \
    --output /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps \
    --split test
python preprocess_scripts/create_evaluation_index.py \
    --full-list data_samples/objaverse_processed/test/full_list.txt \
    --output data/evaluation_index_objaverse_test_4i3o.json \
    --n-input 4 \
    --n-target 3 \
    --seed 42

# preprocess objaverse with envmaps
python preprocess_scripts/preprocess_objaverse.py \
  --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense_lightPlus \
  --output /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse \
  --output-tar /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsmPlus_objaverse_tar \
  --split test \
  --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs 

  # preprocess objaverse with envmaps
python preprocess_scripts/preprocess_objaverse.py \
  --input data_samples/sample_objaverse \
  --output data_samples/objaverse_processed_with_envmaps \
  --output-tar data_samples/objaverse_processed_with_envmaps_tar \
  --split test \
  --hdri-dir data_samples/hdirs \
  --max-objects 10

# preview the scenes:
python preprocess_scripts/preview_scenes.py \
    --full-list /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse/test/full_list_point_light.txt \
    --output scene_preview/preview.png \
    --image-idx 64 \
    --grid-cols 8 \
    --grid-rows 4 \
    --images-per-grid 32
# visualize data
python test_data_visualization.py --config configs/LVSM_scene_encoder_decoder_wEditor.yaml --output-dir ./test_output --subsample 10 --env-sample-num 5000 --batch-idx 8
# delete broken scenes
python preprocess_scripts/remove_broken_scenes.py \
    --broken-scene scene_preview/broken_scene.txt \
    --full-list /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse/train/full_list.txt

# generate full list of point light scenes
python preprocess_scripts/generate_full_list_point_light.py \
  -i /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse/test/full_list.txt \
  -o /scratch/chen.yiwe/temp_objaverse/lvsmPlus_objaverse/test/full_list_point_light.txt

# create evaluation indices
python preprocess_scripts/create_evaluation_index.py \
    --full-list /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps/test/full_list.txt \
    --output data/evaluation_index_objaverse_dense.json \
    --n-input 4 \
    --n-target 8 \
    --max-scenes 125 \
    --seed 42
# create evaluation index of a consecutive traj
python preprocess_scripts/create_rotation_traj.py \
    --full-list /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps/test/full_list.txt \
    --output data/test_rotation_traj_index.json \
    --window-size 50 \
    --n-input 4 \
    --max-scenes 100 
# create evaluation scene of scenes with rotating env:
python preprocess_scripts/preprocess_objaverse_env_variations.py \
    --input /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense \
    --output /projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps_rotating_env \
    --split test \
    --hdri-dir /projects/vig/Datasets/objaverse/envmaps_256/hdirs \
    --n-variations 36 \
    --scene-list env_varation_scene_list.json



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
training.dataset_path = "/projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps/test/full_list.txt" \
training.batch_size_per_gpu = 4 \
training.target_has_input =  false \
training.num_views = 12 \
training.square_crop = true \
training.num_input_views = 4 \
training.num_target_views = 8 \
training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = false \
inference.view_idx_file_path = "./data/evaluation_index_objaverse_dense.json" \
inference_out_dir = ./experiments/evaluation/test_dense_reconstruction

# inference, objaverse
torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
inference_editor.py --config "configs/LVSM_scene_encoder_decoder_wEditor.yaml" \
training.dataset_path = "/projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps_test_split/test/full_list.txt" \
training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4 \
training.batch_size_per_gpu = 4 \
training.target_has_input = false \
training.num_views = 12 \
training.square_crop = true \
training.num_input_views = 4 \
training.num_target_views = 8 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = false \
inference.view_idx_file_path = "./data/evaluation_index_objaverse_dense_test_split.json" \
inference_out_dir = ./experiments/evaluation/test_obj_dense_wEditor_test_split

# env variations inference
torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29506 \
exp_rotate_env.py --config "configs/LVSM_scene_encoder_decoder_wEditor.yaml" \
training.dataset_path = "/projects/vig/Datasets/objaverse/hf-objaverse-v1/lvsm_with_envmaps_rotating_env/test/full_list.txt" \
training.checkpoint_dir = ckpt/LVSM_scene_encoder_decoder_wEditor_general_dense_lr1e4 \
training.batch_size_per_gpu = 1 \
training.target_has_input = false \
training.num_views = 12 \
training.square_crop = true \
training.num_input_views = 4 \
training.num_target_views = 8 \
inference.if_inference = true \
inference.compute_metrics = true \
inference.render_video = false \
inference.view_idx_file_path = "./data/evaluation_index_objaverse_dense_env_variations.json" \
inference_out_dir = ./experiments/evaluation/test_obj_dense_wEditor_env_variations
