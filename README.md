# Flow Planning

Obstacle-blind flow-matching policies that stay task-valid under inference-time
obstacle avoidance. Environments: `franka` (Newton), `libero` (SafeLIBERO), and
`piper` (real arm).

## Setup

```bash
pixi install
```

Set `HF_TOKEN` and `WANDB_API_KEY` in the environment.

## Pipeline

Each environment has a script that records or augments demos, trains, and
evaluates:

```bash
scripts/pipeline.sh                      # franka
scripts/pipeline_libero.sh <suite> <task> # e.g. safelibero_spatial 2
scripts/pipeline_piper.sh                # piper, from the recorded dataset
```

The steps are `scripts/augment.py`, `scripts/train.py`, and `scripts/eval.py`.
Every script takes `--env.type` and `--help` lists the rest.

```bash
pixi run python scripts/eval.py --env.type franka --checkpoint outputs/<run> \
  --env.obstacle true --cond search
```

Tests: `pixi run pytest`. Lint: `pixi run pre-commit run --files <files>`.

## Cluster

Jobs run through SkyPilot. Put the command in the `run:` block of the config,
then:

```bash
pixi run vast            # configs/vast.yaml, launches cluster "vast"
pixi run vast exec       # reuse the running cluster
pixi run launch flow     # configs/sky.yaml on the "flow" host
pixi run slurm run --cluster civo   # configs/slurm.yaml
```

Set idle autostop on Vast: `sky autostop vast -i 20 --down`. `outputs/` is
gitignored, so checkpoints stay on the cluster that made them.

## Piper

Record demos and roll out a policy on the arm:

```bash
pixi run record     # configs/record.yaml
pixi run rollout    # configs/rollout.yaml
```

### Obstacle perception

Live obstacle tracking from two RealSense cameras (overhead and wrist) with
SAM2, fused into a box in the Piper base frame and shown on a dashboard at
`http://<host>:8080`. Everything lives in `hardware/obstacle_perception/` and
runs with `pixi run python hardware/obstacle_perception/<file>`.

```bash
pixi run python hardware/obstacle_perception/track_obstacle.py
```

Calibration files are loaded from `hardware/obstacle_perception/calibration/`
(not committed). Regenerate them when the setup changes:

- `select_top_roi.py`: overhead-camera crop. Rerun if the working view changes.
- `calibrate_cameras.py`: wrist camera relative to the overhead camera. Rerun if
  either camera moves.
- `calibrate_robot_frame.py`: overhead camera relative to the Piper base. Rerun
  if the camera-to-robot geometry changes.
- `track_obstacle.py --select-targets`: object references that initialise
  tracking. Rerun when changing the tracked object.

Camera serials and the SAM2 model id are in
`hardware/obstacle_perception/rig.py`.
