import numpy as np
import torch

from detaug.cbf import PiperCBF, yaw_matrix
from detaug.selector import PIPER_TIP


def tip_of(cbf: PiperCBF, q: torch.Tensor) -> np.ndarray:
    return cbf.piper.arm_points(q)[0, -1].numpy()


def test_ellipsoid_sits_between_link6_and_the_fingertips():
    cbf = PiperCBF("cpu")
    q = torch.zeros(1, 6)
    p, r = cbf.ee(q)
    link6 = cbf.chain.forward_kinematics(q).get_matrix()[0, :3, 3]
    along = float(r[0, :, 2] @ (p[0] - link6))
    assert abs(along - PIPER_TIP / 2) < 1e-5


def test_barrier_signs():
    cbf = PiperCBF("cpu")
    q = torch.zeros(1, 6)
    tip = tip_of(cbf, q)
    cbf.set_boxes([[[*tip, 0.03, 0.03, 0.03, 0.0]]])
    assert cbf.value(q).item() < 0.0
    cbf.set_boxes([[[tip[0] + 0.3, tip[1], tip[2], 0.03, 0.03, 0.03, 0.0]]])
    h_far = cbf.value(q).item()
    assert 0.1 < h_far < 0.3
    cbf.set_boxes([[[tip[0] + 0.3, tip[1], tip[2], 0.03, 0.03, 0.03, np.pi / 2]]])
    assert abs(cbf.value(q).item() - h_far) < 1e-4
    cbf.set_boxes([[[tip[0] + 0.3, tip[1], tip[2], 0.10, 0.02, 0.03, np.pi / 2]]])
    assert cbf.value(q).item() > h_far


def test_yaw_matrix_rotates_about_z():
    r = yaw_matrix(torch.tensor([np.pi / 2]))[0]
    assert torch.allclose(
        r @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), atol=1e-6
    )


def test_filter_only_edits_moves_toward_the_obstacle():
    cbf = PiperCBF("cpu", alpha=10.0, dt=0.05)
    q = torch.zeros(1, 6)
    tip = tip_of(cbf, q)
    cbf.set_boxes([[[tip[0] + 0.12, tip[1], tip[2], 0.03, 0.03, 0.03, 0.0]]])
    h0 = cbf.value(q)
    assert 0.0 < h0.item() < 0.1
    forward = torch.tensor([[0.0, 0.3, -0.1, 0.0, 0.0, 0.0]])
    dq, h = cbf.filter(q, forward)
    assert torch.allclose(h, h0)
    assert cbf.value(q + dq).item() > cbf.value(q + forward).item()
    assert cbf.value(q + dq).item() > -1e-3
    backward = -forward
    dq, _ = cbf.filter(q, backward)
    assert torch.allclose(dq, backward)


def test_cloud_obstacle_is_its_mvee():
    cbf = PiperCBF("cpu")
    q = torch.zeros(1, 6)
    tip = tip_of(cbf, q)
    rng = np.random.default_rng(0)
    cloud = tip + [0.3, 0.0, 0.0] + rng.uniform(-0.03, 0.03, (2000, 3))
    cbf.set_cloud(cloud)
    assert cbf.ob_c is not None
    assert np.abs(cbf.ob_c[0, 0].numpy() - (tip + [0.3, 0.0, 0.0])).max() < 0.01
    h = cbf.value(q).item()
    assert 0.1 < h < 0.3
    cbf.set_cloud(tip + rng.uniform(-0.03, 0.03, (500, 3)))
    assert cbf.value(q).item() < 0.0


def test_arm_barrier_sees_the_camera():
    q = torch.zeros(1, 6)
    bare = PiperCBF("cpu")
    armed = PiperCBF("cpu", arm=True)
    cam = bare.piper.arm_points(q)[0, len(bare.piper.t) - 1].numpy()
    box = [[[*cam, 0.01, 0.01, 0.01, 0.0]]]
    bare.set_boxes(box)
    armed.set_boxes(box)
    assert bare.value(q).item() > 0.0
    assert armed.value(q).item() < 0.0
    armed.set_cloud(cam[None] + np.zeros((4, 3)))
    assert armed.value(q).item() < 0.0
    assert bare.value(q).item() > 0.0


def test_held_sphere_joins_the_barrier():
    cbf = PiperCBF("cpu")
    q = torch.zeros(1, 6)
    tip = tip_of(cbf, q)
    cbf.set_boxes([[[tip[0] + 0.09, tip[1], tip[2], 0.02, 0.02, 0.02, 0.0]]])
    free = cbf.value(q).item()
    assert free > 0.0
    held = cbf.value(q, (cbf.held(0.03), 0.05)).item()
    assert held < free
