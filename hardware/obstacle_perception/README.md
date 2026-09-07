# Obstacle perception

Live obstacle tracking and RGB-D point-cloud estimation for the Piper setup.

## Run

The `rick` setup is already configured, so normal use only requires:

```bash
pixi run python hardware/obstacle_perception/track_obstacle.py
```

This loads the saved camera crop, calibrations, and obstacle references; starts
both cameras, SAM2 tracking, point-cloud estimation, and robot-frame projection;
and serves the dashboard at `http://rick:8080`.

## Recalibration

- `select_top_roi.py`: selects the overhead-camera crop used for tracking
  (`calibration/top_roi.json`). Run again only if the working view changes.
- `calibrate_cameras.py`: estimates the wrist-camera pose relative to the
  overhead camera (`calibration/top_from_left.json`). Run again if either camera
  moves.
- `calibrate_robot_frame.py`: estimates the overhead-camera pose relative to the
  Piper base (`calibration/base_from_top.json`). Run again if the
  camera-to-robot geometry changes.
- `track_obstacle.py --select-targets`: records the object references used to
  initialize tracking in both camera views (`calibration/targets/`). Run again
  when changing the tracked object.

Run each with `pixi run python hardware/obstacle_perception/<file>`. The latest
accepted outputs are stored at the paths above and loaded automatically by
`track_obstacle.py`. On `rick`, they are already populated and ready to use.

## Files

```text
track_obstacle.py             Coordinates cameras, SAM2 tracking, and live output
point_cloud_stream.py         Fuses RGB-D views and estimates the obstacle in robot coordinates
camera_web.py                 Serves the live browser dashboard and shared state
policy_trajectory_stream.py   Projects recorded robot trajectories into the scene

select_top_roi.py             Selects the working crop of the overhead camera
calibrate_cameras.py          Calibrates the wrist camera relative to the overhead camera
calibrate_robot_frame.py      Calibrates the overhead camera relative to the Piper base
rig.py                        Camera serials, SAM2 model id, and the Piper pose reader

calibration/                  Generated calibration and target files; not committed
runs/                         Generated runtime output; not committed
```
