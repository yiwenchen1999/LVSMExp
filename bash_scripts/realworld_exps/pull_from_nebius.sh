REMOTE_HOST=yiwen@204.12.169.196 \
REMOTE_CKPT_DIR=/home/yiwen/LVSMExp/ckpt/infer_stanfordORB_512_editor \
RUN_NAME=infer_stanfordORB_512_editor \
  bash bash_scripts/realworld_exps/pull_realworld_eval_previews.sh \
&& RUN_NAME=infer_stanfordORB_512_editor \
  bash bash_scripts/realworld_exps/reorganize_realworld_eval_previews.sh