import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.utils.constants import OBS_STATE

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rollout  # noqa: E402  # ty: ignore[unresolved-import]
from rollout import (  # noqa: E402  # ty: ignore[unresolved-import]
    DetAugConfig,
    attach_cbf,
)


class FakeClient:
    period = 0.02

    def __init__(self, box):
        self.box_value = box
        self.stopped = False

    def cloud(self):
        return None

    def stop(self):
        self.stopped = True


class FakeGuard:
    def __init__(self, client):
        self.client = client

    def current(self):
        return self.client.box_value


def fake_ctx(nominal):
    engine = SimpleNamespace(get_action=lambda obs: None if obs is None else nominal)
    policy = SimpleNamespace(config=SimpleNamespace(device="cpu"))
    return SimpleNamespace(policy=SimpleNamespace(policy=policy, inference=engine))


def run(monkeypatch, box, nominal, tmp_path=None, **overrides):
    client = FakeClient(np.asarray(box, np.float32))
    monkeypatch.setattr(
        rollout, "connect_tracker", lambda cfg: (client, FakeGuard(client))
    )
    cfg = DetAugConfig(obstacle_url="http://fake", cbf=True, **overrides)
    if tmp_path is not None:
        cfg.dump = str(tmp_path)
    ctx = fake_ctx(torch.tensor(nominal, dtype=torch.float32))
    _, save = attach_cbf(ctx, cfg, fps=20.0)
    out = ctx.policy.inference.get_action({OBS_STATE: np.zeros(7, np.float32)})
    assert ctx.policy.inference.get_action(None) is None
    save()
    return out


def test_filter_bends_a_move_into_the_obstacle(monkeypatch, tmp_path):
    cbf = rollout.PiperCBF("cpu")
    tip = cbf.piper.arm_points(torch.zeros(1, 6))[0, -1].numpy()
    box = [tip[0] + 0.12, tip[1], tip[2], 0.03, 0.03, 0.03, 0.0]
    nominal = [0.0, 17.0, -6.0, 0.0, 0.0, 0.0, 5.0]
    out = run(monkeypatch, box, nominal, tmp_path)
    assert out.shape == (7,)
    assert float(out[6]) == 5.0
    assert not torch.allclose(out[:6], torch.tensor(nominal[:6]))
    cbf.set_boxes([np.asarray(box)[None]])
    q_out = torch.deg2rad(out[:6])[None]
    q_nom = torch.deg2rad(torch.tensor(nominal[:6]))[None]
    assert cbf.value(q_out).item() > cbf.value(q_nom).item()
    dumps = list(tmp_path.glob("cbf-*.npz"))
    assert len(dumps) == 1
    data = np.load(dumps[0])
    assert data["act"].shape == (1, 7) and data["h"].shape == (1,)
    assert np.allclose(data["box"][0], box)


def test_filter_leaves_a_clear_move_alone(monkeypatch):
    box = [1.0, 1.0, 0.5, 0.03, 0.03, 0.03, 0.0]
    nominal = [0.0, 17.0, -6.0, 0.0, 0.0, 0.0, 5.0]
    out = run(monkeypatch, box, nominal)
    assert torch.allclose(out, torch.tensor(nominal), atol=1e-4)
