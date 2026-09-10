import numpy as np
import torch
from torch import Tensor

from detaug.kinematics import build_franka_chain, build_piper_chain, ee_positions
from detaug.selector import PIPER_TIP, obstacle_dist

FAR_BOX = (1e3, 1e3, 1e3, 0.01, 0.01, 0.01, 0.0)


class CapeGuidance:
    box_dim = 6

    def __init__(
        self,
        device,
        base_pos,
        act_stats,
        obs_stats,
        joint_start: int,
        radius: float = 0.08,
        margin: float = 0.06,
        n_arm: int = 7,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.device = device
        self.base = torch.as_tensor(base_pos, **f32)
        self.am = torch.as_tensor(act_stats["mean"], **f32)[:n_arm]
        self.asd = torch.as_tensor(act_stats["std"], **f32)[:n_arm]
        self.sm = torch.as_tensor(obs_stats["mean"], **f32)[:n_arm]
        self.ssd = torch.as_tensor(obs_stats["std"], **f32)[:n_arm]
        self.js = joint_start
        self.n = n_arm
        self.keep = radius + margin
        self.boxes = torch.as_tensor([[FAR_BOX[: self.box_dim]]], **f32)
        self.cloud = None
        self.build_chain(device)

    def build_chain(self, device):
        self.chain = build_franka_chain(device)[0]

    def set_boxes(self, boxes):
        b = np.asarray(boxes, np.float32).reshape(len(boxes), -1, self.box_dim)
        with torch.inference_mode(False):
            self.boxes = torch.as_tensor(b, device=self.device)
        self.cloud = None

    def set_cloud(self, points):
        with torch.inference_mode(False):
            self.cloud = (
                None
                if points is None or not len(points)
                else torch.as_tensor(np.asarray(points, np.float32), device=self.device)
            )

    def joints(self, x: Tensor) -> tuple[Tensor, Tensor]:
        n, js = self.n, self.js
        return x[..., :n] * self.ssd + self.sm, x[..., js : js + n] * self.asd + self.am

    def ee(self, q: Tensor) -> Tensor:
        b, h = q.shape[:2]
        return (
            ee_positions(self.chain, q.reshape(-1, self.n)).reshape(b, h, 3) + self.base
        )

    def cost(self, x: Tensor) -> Tensor:
        assert len(x) % len(self.boxes) == 0
        boxes = self.boxes.repeat_interleave(len(x) // len(self.boxes), 0)
        c = x.new_zeros(len(x))
        for q in self.joints(x):
            d = obstacle_dist(self.ee(q.float()), boxes, self.cloud)
            c = c + (self.keep - d).clamp(min=0.0).sum(1)
        return c

    def grad(self, x: Tensor) -> Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            return torch.autograd.grad(self.cost(x).sum(), x)[0]


class PiperCape(CapeGuidance):
    box_dim = 7

    def __init__(
        self,
        device,
        act_stats,
        obs_stats,
        joint_start: int,
        radius: float = 0.08,
        margin: float = 0.02,
        base_pos=(0.0, 0.0, 0.0),
        offset=(0.0, 0.0, PIPER_TIP / 2),
    ):
        super().__init__(
            device, base_pos, act_stats, obs_stats, joint_start, radius, margin, 6
        )
        self.offset = torch.as_tensor(offset, dtype=torch.float32, device=device)

    def build_chain(self, device):
        self.chain = build_piper_chain(device)

    def joints(self, x: Tensor) -> tuple[Tensor, Tensor]:
        obs, act = super().joints(x)
        return torch.deg2rad(obs), torch.deg2rad(act)

    def ee(self, q: Tensor) -> Tensor:
        b, h = q.shape[:2]
        m = self.chain.forward_kinematics(q.reshape(-1, self.n)).get_matrix()
        p = m[:, :3, 3] + m[:, :3, :3] @ self.offset
        return p.reshape(b, h, 3) + self.base
