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


def test_camera_rides_above_the_gripper():
    fc = PiperCollision("cpu")
    q = torch.zeros(1, 6)
    mats = fc.frames(q)
    p6 = mats[0, -1, :3, 3]
    pts = fc.arm_points(q)
    assert pts.shape[1] == len(fc.radii)
    cam = pts[0, : len(fc.t)]
    assert torch.allclose(fc.radii[: len(fc.t)], torch.full((len(fc.t),), 0.035))
    local = (cam - p6) @ mats[0, -1, :3, :3]
    assert torch.allclose(local[0], torch.tensor([-0.045, 0.0, 0.09]), atol=1e-5)
    assert torch.allclose(local[-1], torch.tensor([-0.095, 0.0, 0.11]), atol=1e-5)
    assert (cam[:, 2] > p6[2] + 0.04).all()
    bare = PiperCollision("cpu", camera_radius=0.0)
    assert bare.arm_points(q).shape[1] == pts.shape[1] - len(fc.t)


def test_camera_is_scored():
    sel = PiperSelector(STATS, STATS, joint_start=7, device="cpu")
    traj = torch.zeros(1, 5, 14)
    cam = sel.fc.arm_points(torch.zeros(1, 6))[0, len(sel.fc.t) - 1].numpy()
    sel.set_boxes([*cam, 0.01, 0.01, 0.01])
    assert abs(sel.score(traj).item() - 5 * (0.035 + 0.01)) < 1e-3
    bare = PiperSelector(STATS, STATS, joint_start=7, device="cpu", camera_radius=0.0)
    bare.set_boxes([*cam, 0.01, 0.01, 0.01])
    assert bare.score(traj).item() == 0.0


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


def test_yawed_box_rotates_its_extents():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sel = PiperSelector(STATS, STATS, joint_start=7, device=dev)
    traj = torch.zeros(1, 1, 14, device=dev)
    tip = sel.fc.arm_points(torch.zeros(1, 6, device=dev))[0, -1].cpu().numpy()
    centre = tip + np.array([0.10, 0.0, 0.0], np.float32)
    sel.set_boxes([*centre, 0.03, 0.10, 0.06])
    assert sel.score(traj).item() == 0.0
    sel.set_boxes([*centre, 0.03, 0.10, 0.06, np.pi / 2])
    assert sel.score(traj).item() > 0.0


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
    box = ObstacleBox(0.0, 0.0, 0.08, 0.08, 0.0, 0.0, 0.2, 500)
    corners = np.array(
        [
            [x, y, z]
            for z in (0.0, 0.2)
            for x, y in ((-0.04, -0.04), (0.04, -0.04), (0.04, 0.04), (-0.04, 0.04))
        ],
        np.float32,
    )
    payload = obstacle_payload(corners, np.zeros((0, 3), np.float32), box, 10)
    assert payload["box"] == [0.0, 0.0, 0.1, 0.04, 0.04, 0.1, 0.0]
    assert obstacle_payload(None, corners, None, 10)["box"] is None


def test_obstacle_payload_subsamples_the_cloud():
    box = ObstacleBox(0.0, 0.0, 0.08, 0.08, 0.0, 0.0, 0.2, 500)
    corners = np.zeros((8, 3), np.float32)
    payload = obstacle_payload(corners, np.zeros((900, 3), np.float32), box, 100)
    cloud = payload["cloud"]
    assert isinstance(cloud, list) and len(cloud) == 100


def test_fit_footprint_recovers_a_rotated_rectangle():
    from detaug.perception.point_cloud_stream import fit_footprint

    rng = np.random.default_rng(0)
    local_u = rng.uniform(-0.10, 0.10, 4000)
    local_v = rng.uniform(-0.04, 0.04, 4000)
    angle = 0.6
    table_u = 0.3 + local_u * np.cos(angle) - local_v * np.sin(angle)
    table_v = -0.1 + local_u * np.sin(angle) + local_v * np.cos(angle)
    center_u, center_v, size_u, size_v, yaw = fit_footprint(table_u, table_v, 0.0)
    assert abs(center_u - 0.3) < 0.003 and abs(center_v + 0.1) < 0.003
    assert abs(size_u - 0.20) < 0.005 and abs(size_v - 0.08) < 0.005
    assert abs(yaw - angle) < 0.02


def test_fit_footprint_snaps_a_square_to_zero_yaw():
    from detaug.perception.point_cloud_stream import fit_footprint

    rng = np.random.default_rng(1)
    table_u = rng.uniform(-0.05, 0.05, 2000)
    table_v = rng.uniform(-0.05, 0.05, 2000)
    _, _, size_u, size_v, yaw = fit_footprint(table_u, table_v, 0.0)
    assert yaw == 0.0
    assert abs(size_u - 0.10) < 0.005 and abs(size_v - 0.10) < 0.005
