#!/usr/bin/env bash
set -e

SRC=reece-omahoney/pick-and-place
REPO=${REPO:-${SRC}-aug}
RUN=outputs/piper/${REPO##*/}

rm -rf ~/.cache/huggingface/lerobot/$REPO
pixi run python scripts/augment.py --env.type piper \
  --src_repo $SRC --dst_repo $REPO --copies ${COPIES:-6} --bend_min 0.25 --bend_max 0.6 --ik_rot_weight 0.1 ${AUG_ARGS:-}

pixi run python scripts/train.py --env.type piper --repo_id $REPO \
  --horizon 50 --num_iters ${ITERS:-75000} --eval_every 0 --run_dir $RUN

pixi run hf upload ${HUB:-reece-omahoney/piper-pick-and-place-aug} $RUN
