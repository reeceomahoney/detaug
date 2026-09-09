import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from detaug.kinematics import EE_FRAME, build_franka_chain, build_piper_chain
from detaug.selector import (
    PIPER_CAMERA_RADIUS,
    PIPER_RADIUS,
    PIPER_TIP,
    POINTS_PER_LINK,
    PiperCollision,
    box_sdf,
    obstacle_dist,
)

EE_AXES = (0.06, 0.12, 0.11)
EE_OFFSET = (0.0, 0.0, -0.056)
PIPER_EE_AXES = (0.05, 0.06, 0.09)
PIPER_EE_OFFSET = (0.0, 0.0, PIPER_TIP / 2)
FAR = 1e3
MVEE_POINTS = 512


def mvee(points, tol: float = 1e-4) -> tuple[Tensor, Tensor]:
    p = torch.as_tensor(points, dtype=torch.float64).cpu()
    n, d = p.shape
    q = torch.cat([p, p.new_ones(n, 1)], 1)
    u = p.new_full((n,), 1.0 / n)
    err = 1.0
    while err > tol:
        x = q.T @ (u[:, None] * q)
        m = (q @ torch.linalg.solve(x, q.T)).diagonal()
        j = int(m.argmax())
        step = (m[j] - d - 1) / ((d + 1) * (m[j] - 1))
        new = (1 - step) * u
        new[j] += step
        err = float((new - u).norm())
        u = new
    c = u @ p
    a = torch.linalg.inv(p.T @ (u[:, None] * p) - c[:, None] * c[None]) / d
    return c.float(), a.float()


def bounding_sphere(points) -> tuple[Tensor, Tensor]:
    p = torch.as_tensor(points, dtype=torch.float32).cpu()
    c = p.mean(0)
    r = (p - c).norm(dim=1).max().clamp_min(1e-3)
    return c, torch.eye(3) / (r * r)


def size_matrix(a: Tensor) -> Tensor:
    w, v = torch.linalg.eigh(a)
    return v @ torch.diag(w.clamp_min(1e-12) ** -0.5) @ v.T


def yaw_matrix(yaw: Tensor) -> Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(*yaw.shape, 3, 3)


def fibonacci_sphere(n: int) -> Tensor:
    i = torch.arange(n, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / n
    r = (1 - z * z).sqrt()
    ang = math.pi * (1 + 5**0.5) * i
    return torch.stack([r * ang.cos(), r * ang.sin(), z], 1)


class EllipsoidCBF:
    def __init__(
        self,
        device,
        base_pos,
        alpha: float = 10.0,
        dt: float = 0.05,
        n_dirs: int = 1024,
        axes: Sequence[float] = EE_AXES,
        offset: Sequence[float] = EE_OFFSET,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.device = device
        self.base_pos = torch.as_tensor(base_pos, **f32)
        self.axes = torch.as_tensor(axes, **f32)
        self.offset = torch.as_tensor(offset, **f32)
        self.dirs = fibonacci_sphere(n_dirs).to(device)
        self.alpha, self.dt = alpha, dt
        self.margin = 0.0
        self.arm_links: tuple[int, ...] = ()
        self.fc = None
        self.boxes = None
        self.ob_c = self.ob_q = None
        self.build_chain(device)

    def build_chain(self, device):
        self.chain, self.njoints, _ = build_franka_chain(device)

    def set_obstacles(self, point_sets):
        cs, qs = [], []
        for pts in point_sets:
            if pts is None or len(pts) < 4:
                cs.append(torch.full((3,), FAR))
                qs.append(torch.eye(3) * 1e-3)
                continue
            pts = np.asarray(pts, np.float64)
            if len(pts) > MVEE_POINTS:
                pts = pts[np.linspace(0, len(pts) - 1, MVEE_POINTS, dtype=np.int64)]
            try:
                c, a = mvee(pts)
            except torch.linalg.LinAlgError:
                c, a = bounding_sphere(pts)
            cs.append(c)
            qs.append(size_matrix(a))
        self.ob_c = torch.stack(cs).to(self.device)[:, None]
        self.ob_q = torch.stack(qs).to(self.device)[:, None]

    def set_boxes(self, boxes_per_world):
        g = max(len(b) if b is not None else 1 for b in boxes_per_world)
        w = len(boxes_per_world)
        c = torch.full((w, g, 3), FAR)
        q = torch.eye(3).repeat(w, g, 1, 1) * 1e-3
        boxes = torch.tensor([FAR, FAR, FAR, 0.01, 0.01, 0.01, 0.0]).repeat(w, g, 1)
        for i, b in enumerate(boxes_per_world):
            if b is None:
                continue
            b = torch.as_tensor(np.asarray(b), dtype=torch.float32)
            c[i, : len(b)] = b[:, :3]
            qi = torch.diag_embed(b[:, 3:6] * 3**0.5)
            if b.shape[1] > 6:
                r = yaw_matrix(b[:, 6])
                qi = r @ qi @ r.transpose(1, 2)
            q[i, : len(b)] = qi
            boxes[i, : len(b), : b.shape[1]] = b
        self.ob_c, self.ob_q = c.to(self.device), q.to(self.device)
        self.boxes = boxes.to(self.device)

    def h_arm(self, q: Tensor) -> Tensor:
        assert self.fc is not None and self.boxes is not None
        pts = self.fc.arm_points(q) + self.base_pos
        idx = torch.cat(
            [
                torch.arange(POINTS_PER_LINK * k, POINTS_PER_LINK * (k + 1))
                for k in self.arm_links
            ]
        ).to(q.device)
        pts = pts[:, idx]
        boxes = self.boxes
        if len(pts) != len(boxes):
            boxes = boxes.repeat_interleave(len(pts) // len(boxes), 0)
        d = torch.stack(
            [
                box_sdf(pts, boxes[:, k, None, :3], boxes[:, k, None, 3:6])
                for k in range(boxes.shape[1])
            ]
        ).amin(0)
        return (d - self.fc.radius).amin(1)

    def ee(self, q: Tensor) -> tuple[Tensor, Tensor]:
        pad = q.new_zeros(q.shape[0], self.njoints - q.shape[1])
        m = self.chain.forward_kinematics(torch.cat([q, pad], 1))[EE_FRAME].get_matrix()
        r = m[:, :3, :3]
        p = m[:, :3, 3] + self.base_pos + (r @ self.offset)
        return p, r

    def attach_offset(self, q: Tensor, world_pt: Tensor) -> Tensor:
        p, r = self.ee(q)
        p_tcp = p - (r @ self.offset)
        return (r.transpose(1, 2) @ (world_pt - p_tcp)[..., None])[..., 0]

    def h_ellipsoid(self, p: Tensor, qe_inv: Tensor) -> Tensor:
        assert self.ob_c is not None and self.ob_q is not None
        ob_c, ob_q = self.ob_c, self.ob_q
        if len(p) != len(ob_c):
            rep = len(p) // len(ob_c)
            ob_c = ob_c.repeat_interleave(rep, 0)
            ob_q = ob_q.repeat_interleave(rep, 0)
        n = self.dirs[None] @ qe_inv
        num = (
            -(n[:, None] @ ob_q).norm(dim=-1)
            + ((ob_c[:, :, None] - p[:, None, None]) * n[:, None]).sum(-1)
            - 1.0
        )
        return (num / n.norm(dim=-1)[:, None]).amax(2).amin(1)

    def h(self, p_ep: Tensor, r: Tensor) -> Tensor:
        qe_inv = r @ torch.diag(1.0 / self.axes) @ r.transpose(1, 2)
        return self.h_ellipsoid(p_ep, qe_inv)

    def value(self, q: Tensor, attach=None) -> Tensor:
        p, r = self.ee(q)
        h = self.h(p, r)
        if self.arm_links:
            h = torch.minimum(h, self.h_arm(q))
        if attach is not None:
            offset, radius = attach
            p_a = p - (r @ self.offset) + (r @ offset[..., None])[..., 0]
            eye = torch.eye(3, device=q.device).expand(len(q), 3, 3) / radius
            h = torch.minimum(h, self.h_ellipsoid(p_a, eye))
        return h

    def project_chunk(self, q0: Tensor, targets: Tensor, attach=None) -> Tensor:
        q, out = q0, []
        for i in range(targets.shape[1]):
            dq, _ = self.filter(q, targets[:, i] - q, attach)
            q = q + dq
            out.append(q)
        return torch.stack(out, 1)

    def constrain(self, dq_nom: Tensor, g: Tensor, h: Tensor) -> Tensor:
        slack = -self.alpha * self.dt * (h - self.margin) - (g * dq_nom).sum(1)
        lam = (slack / (g * g).sum(1).clamp_min(1e-9)).clamp_min(0.0)
        return dq_nom + lam[:, None] * g

    def filter(self, q: Tensor, dq_nom: Tensor, attach=None) -> tuple[Tensor, Tensor]:
        q = q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            h = self.value(q, attach)
            g = torch.autograd.grad(h.sum(), q)[0]
        return self.constrain(dq_nom, g, h.detach()), h.detach()


class PiperCBF(EllipsoidCBF):
    def __init__(
        self,
        device,
        base_pos=(0.0, 0.0, 0.0),
        alpha: float = 10.0,
        dt: float = 0.05,
        n_dirs: int = 1024,
        axes: Sequence[float] = PIPER_EE_AXES,
        offset: Sequence[float] = PIPER_EE_OFFSET,
        arm: bool = False,
        radius: float = PIPER_RADIUS,
        camera_radius: float = PIPER_CAMERA_RADIUS,
    ):
        super().__init__(device, base_pos, alpha, dt, n_dirs, axes, offset)
        self.arm_links = (0,) if arm else ()
        self.piper = PiperCollision(
            device, base_pos, radius=radius, camera_radius=camera_radius
        )
        self.fc = self.piper
        self.cloud = None

    def build_chain(self, device):
        self.chain = build_piper_chain(device)
        self.njoints = 6

    def held(self, offset: float) -> Tensor:
        return torch.tensor([[0.0, 0.0, self.piper.tip + offset]], device=self.device)

    def set_cloud(self, points):
        self.cloud = (
            None
            if points is None or not len(points)
            else torch.as_tensor(np.asarray(points, np.float32), device=self.device)
        )
        self.boxes = None
        self.set_obstacles([points])

    def set_boxes(self, boxes_per_world):
        self.cloud = None
        super().set_boxes(boxes_per_world)

    def ee(self, q: Tensor) -> tuple[Tensor, Tensor]:
        m = self.chain.forward_kinematics(q).get_matrix()
        r = m[:, :3, :3]
        p = m[:, :3, 3] + self.base_pos + (r @ self.offset)
        return p, r

    def h_arm(self, q: Tensor) -> Tensor:
        pts = self.piper.arm_points(q)
        if self.cloud is not None:
            d = torch.cdist(pts, self.cloud[None].expand(len(pts), -1, -1)).amin(2)
        else:
            assert self.boxes is not None
            boxes = self.boxes
            if len(pts) != len(boxes):
                boxes = boxes.repeat_interleave(len(pts) // len(boxes), 0)
            d = obstacle_dist(pts, boxes, None)
        return (d - self.piper.radii).amin(1)
