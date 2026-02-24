python3 preprocess_scripts/update_paths.py \
  --old-path /scratch/chen.yiwe/temp_objaverse \
  --new-path /data \
  --root-dir /data/lvsmPlus_objaverse/test \
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
  --full-list /data/lvsmPlus_objaverse/train/full_list.txt
 
 scp ubuntu@147.185.41.15:/home/ubuntu/LVSMExp/ckpt/LVSM_object_encoder_decoder_sparse/iter_00000001/supervision_2cb7026ffacf4889bb265099b8460c7f_env_2.jpg bash_scripts/VoltagePark/previews/