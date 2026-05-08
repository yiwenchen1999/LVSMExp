OG_DATASET_BASE=${OG_DATASET_BASE:-/scratch/chen.yiwe/temp_objaverse}
LOCAL_DATASET_BASE=${LOCAL_DATASET_BASE:-/mnt/data-disk}

torchrun --nproc_per_node 8 --nnodes 1 \
    --rdzv_id 18635 --rdzv_backend c10d --rdzv_endpoint localhost:29501 \
    train_editor.py --config configs/LVSM_scene_encoder_decoder_wEditor_general_dense_512_singleMap.yaml \
    training.batch_size_per_gpu = 16 \
    training.checkpoint_dir = /mnt/data-disk/ckpt/finetune512 \
    training.dataset_path = /mnt/data-disk/lvsmPlus_objaverse/test/full_list.txt \
    training.LVSM_checkpoint_dir = /mnt/data-disk/ckpt/finetune512 \
    training.wandb_exp_name = LVSM_edit_dense_general_512 \
    training.warmup = 3000 \
    training.vis_every = 1000 \
    training.lr = 0.0001 \
    training.og_dataset_base = ${OG_DATASET_BASE} \
    training.local_dataset_base = ${LOCAL_DATASET_BASE} 
