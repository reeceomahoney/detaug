import numpy as np
import torch

from detaug.perception.point_cloud_stream import ObstacleBox, obstacle_payload
from detaug.selector import PiperCollision, PiperSelector

STATS = {"mean": [0.0] * 7, "std": [1.0] * 7}


def test_tip_reaches_past_link6():
    fc = PiperCollision("cpu")
    q = torch.zeros(1, 6)
    mats = fc.frames(q)
    p6 = mats[0, -1, :3, 3]
    pts = fc.arm_points(q)
    assert torch.allclose(pts[0, -1], fc.tip_point(mats)[0])
    assert abs(float((pts[0, -1] - p6).norm()) - fc.tip) < 1e-5


def test_score_is_penetration_depth():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sel = PiperSelector(STATS, STATS, joint_start=7, device=dev)
    traj = torch.zeros(2, 20, 14, device=dev)
    sel.set_boxes([1e3, 1e3, 1e3, 0.01, 0.01, 0.01])
    assert sel.score(traj).abs().max().item() == 0.0

    tip = sel.fc.arm_points(torch.zeros(1, 6, device=dev))[0, -1].cpu().numpy()
    half = 0.03
    sel.set_boxes([*tip, half, half, half])
    # every frame is the same pose, so the cost is the per-frame depth times T
    assert abs(sel.score(traj)[0].item() - 20 * (sel.fc.radius + half)) < 1e-3


def test_degrees_are_converted():
    sel = PiperSelector(
        {"mean": [90.0] * 6 + [0.0], "std": [1.0] * 7},
        {"mean": [90.0] * 6 + [0.0], "std": [1.0] * 7},
        joint_start=7,
        device="cpu",
    )
    q = sel.joints(torch.zeros(1, 2, 14))[0]
    assert torch.allclose(q, torch.full_like(q, np.pi / 2))


def test_cloud_geometry_matches_box_surface():
    sel = PiperSelector(STATS, STATS, joint_start=7, device="cpu")
    tip = sel.fc.arm_points(torch.zeros(1, 6))[0, -1].numpy()
    face = tip + np.array([0.1, 0.0, 0.0], np.float32)
    sel.set_cloud(face[None])
    traj = torch.zeros(1, 4, 14)
    # nearest cloud point sits 0.1 away, so the capsule radius leaves clearance
    assert sel.score(traj).item() == 0.0
    sel.set_cloud((tip + np.array([0.01, 0.0, 0.0], np.float32))[None])
    assert sel.score(traj).item() > 0.0


def test_obstacle_payload_bounds_the_corners():
    box = ObstacleBox(0.0, 0.0, 0.08, 0.0, 0.2, 500)
    corners = np.array(
        [
            [x, y, z]
            for z in (0.0, 0.2)
            for x, y in ((-0.04, -0.04), (0.04, -0.04), (0.04, 0.04), (-0.04, 0.04))
        ],
        np.float32,
    )
    payload = obstacle_payload(corners, np.zeros((0, 3), np.float32), box, 10)
    assert payload["box"] == [0.0, 0.0, 0.1, 0.04, 0.04, 0.1]
    assert obstacle_payload(None, corners, None, 10)["box"] is None


def test_obstacle_payload_subsamples_the_cloud():
    box = ObstacleBox(0.0, 0.0, 0.08, 0.0, 0.2, 500)
    corners = np.zeros((8, 3), np.float32)
    payload = obstacle_payload(corners, np.zeros((900, 3), np.float32), box, 100)
    cloud = payload["cloud"]
    assert isinstance(cloud, list) and len(cloud) == 100
