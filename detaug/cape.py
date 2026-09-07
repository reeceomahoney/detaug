import numpy as np
import torch
from torch import Tensor

from detaug.kinematics import build_franka_chain, ee_positions
from detaug.selector import box_sdf


class CapeGuidance:
    def __init__(
        self,
        device,
        base_pos,
        act_stats,
        obs_stats,
        joint_start: int,
        radius: float = 0.08,
        margin: float = 0.06,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.device = device
        self.chain = build_franka_chain(device)[0]
        self.base = torch.as_tensor(base_pos, **f32)
        self.am = torch.as_tensor(act_stats["mean"], **f32)[:7]
        self.asd = torch.as_tensor(act_stats["std"], **f32)[:7]
        self.sm = torch.as_tensor(obs_stats["mean"], **f32)[:7]
        self.ssd = torch.as_tensor(obs_stats["std"], **f32)[:7]
        self.js = joint_start
        self.keep = radius + margin
        self.boxes = None

    def set_boxes(self, boxes):
        b = np.asarray(boxes, np.float32).reshape(len(boxes), -1, 6)
        self.boxes = torch.as_tensor(b, device=self.device)

    def ee(self, q: Tensor) -> Tensor:
        b, h = q.shape[:2]
        return ee_positions(self.chain, q.reshape(-1, 7)).reshape(b, h, 3) + self.base

    def cost(self, x: Tensor) -> Tensor:
        assert self.boxes is not None and len(x) % len(self.boxes) == 0
        boxes = self.boxes.repeat_interleave(len(x) // len(self.boxes), 0)
        c = x.new_zeros(len(x))
        js = self.js
        for q in (
            x[..., :7] * self.ssd + self.sm,
            x[..., js : js + 7] * self.asd + self.am,
        ):
            p = self.ee(q.float())[:, :, None]
            d = box_sdf(p, boxes[:, None, :, :3], boxes[:, None, :, 3:]).amin(-1)
            c = c + (self.keep - d).clamp(min=0.0).sum(1)
        return c

    def grad(self, x: Tensor) -> Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            return torch.autograd.grad(self.cost(x).sum(), x)[0]
