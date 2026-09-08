"""Best-of-N plan selection: exact FK + box geometry scores each sampled plan
and the lowest-cost one is executed. Selection needs no gradients, so exact
geometry beats a learned proxy."""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch_volumetric import sdf as pv_sdf
from torch import Tensor

from detaug.kinematics import build_franka_chain, build_piper_chain

LINKS = [f"fr3_link{i}" for i in range(8)] + ["fr3_hand"]
FINGERS = ["fr3_leftfinger", "fr3_rightfinger"]
HAND = LINKS.index("fr3_hand")
FINGER_POINTS = [[0.0, 0.01, 0.01], [0.0, 0.01, 0.03], [0.0, 0.008, 0.045]]
FINGER_Q = 0.02
POINTS_PER_LINK = 128
POINT_RADIUS = 0.015
SELF_SUBSET = 64
SELF_STRIDE = 4
SELF_GAP = 0.01


def box_sdf(points: Tensor, center: Tensor, half_extents) -> Tensor:
    """Signed distance from points (..., d) to an axis-aligned box."""
    q = (points - center).abs() - half_extents
    outside = q.clamp(min=0.0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp(max=0.0)
    return outside + inside


def obstacle_dist(points: Tensor, box: Tensor, cloud: Tensor | None) -> Tensor:
    """Distance from points (b, ..., 3) to the obstacle: signed against boxes,
    unsigned nearest-neighbour against a cloud."""
    if cloud is None:
        if box.dim() == 2:
            box = box[:, None]
        shape = (len(box),) + (1,) * (points.dim() - 2) + (3,)
        return torch.stack(
            [
                box_sdf(points, box[:, k, :3].view(shape), box[:, k, 3:].view(shape))
                for k in range(box.shape[1])
            ]
        ).amin(dim=0)
    p = points.reshape(-1, 3)
    step = 1 << 19
    out = [
        torch.cdist(p[i : i + step], cloud).amin(dim=1) for i in range(0, len(p), step)
    ]
    return torch.cat(out).reshape(points.shape[:-1])


def sample_mesh(mesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    if len(pts) < n:
        extra, _ = trimesh.sample.sample_surface(mesh, n - len(pts), seed=0)
        pts = np.concatenate([pts, extra])
    return np.asarray(pts[:n], np.float32)


class LinkSDF:
    """Signed distance of a link's collision mesh, baked to a voxel grid in the
    link frame and read back with trilinear interpolation."""

    def __init__(self, path: str, bounds, device: str, res=0.005, pad=0.03):
        lo, hi = np.asarray(bounds[0]) - pad, np.asarray(bounds[1]) + pad
        axes = [torch.arange(lo[k], hi[k], res) for k in range(3)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        f = pv_sdf.MeshSDF(pv_sdf.MeshObjectFactory(mesh_name=path))
        d, _ = f(grid.reshape(-1, 3), compute_grad=False)
        self.grid = d.reshape(1, 1, *grid.shape[:3]).float().to(device)
        self.lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.hi = torch.tensor([a[-1] for a in axes], device=device)

    def __call__(self, p: Tensor) -> Tensor:
        g = (2.0 * (p - self.lo) / (self.hi - self.lo) - 1.0).flip(-1)
        out = F.grid_sample(
            self.grid,
            g.reshape(1, 1, 1, -1, 3),
            padding_mode="border",
            align_corners=True,
        )
        return out.reshape(p.shape[:-1])


class FrankaCollision:
    """Collision points sampled once from each link's collision mesh and FK'd
    per frame (exact wall and cube tests), plus per-link SDFs so one link's
    points can be tested inside another's volume (self-collision)."""

    def __init__(self, device: str, base_pos, cube_size: float, n=POINTS_PER_LINK):
        chain, njoints, root = build_franka_chain(device)
        self.chain = chain
        self.njoints = njoints
        self.device = device
        self.base_pos = torch.as_tensor(base_pos, dtype=torch.float32, device=device)
        self.cube_half = 0.5 * cube_size
        self.frames = LINKS + FINGERS
        self.frame_indices = chain.get_frame_indices(*self.frames)
        local, owner, self.sdfs = [], [], []
        for i, name in enumerate(LINKS):
            path = root + chain.find_frame(name).link.visuals[0].geom_param[0]
            mesh = trimesh.load(path, force="mesh")
            local.append(sample_mesh(mesh, n))
            owner += [i] * n
            self.sdfs.append(LinkSDF(path, mesh.bounds, device))
        for i, _ in enumerate(FINGERS):
            local.append(np.asarray(FINGER_POINTS, np.float32))
            owner += [len(LINKS) + i] * len(FINGER_POINTS)
        self.local = torch.as_tensor(np.concatenate(local), device=device)
        self.owner = torch.as_tensor(owner, device=device)
        self.arm_mask = self.owner < len(LINKS) - 1
        self.radius = POINT_RADIUS
        self.sub = SELF_SUBSET
        self.pairs = [
            (i, j) for i in range(len(LINKS)) for j in range(i + 2, len(LINKS))
        ]
        self.self_gap = torch.full((len(self.pairs),), SELF_GAP, device=device)

    def forward(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """q (m, 7) -> frame matrices (m, F, 4, 4) and points (m, K, 3), base frame."""
        m = q.shape[0]
        th = torch.cat([q, q.new_full((m, self.njoints - 7), FINGER_Q)], dim=1)
        tf = self.chain.forward_kinematics(th, self.frame_indices)
        mats = torch.stack([tf[n].get_matrix() for n in self.frames], dim=1)
        rot, t = mats[:, self.owner, :3, :3], mats[:, self.owner, :3, 3]
        return mats, torch.einsum("mkij,kj->mki", rot, self.local) + t

    def arm_points(self, q: Tensor) -> Tensor:
        return self.forward(q)[1]

    def into(self, pts: Tensor, mats: Tensor, j: int) -> Tensor:
        """World points (m, S, 3) expressed in link j's frame."""
        r, t = mats[:, j, :3, :3], mats[:, j, :3, 3]
        return torch.einsum("mji,msj->msi", r, pts - t[:, None])

    def self_dists(self, mats: Tensor, pts: Tensor) -> Tensor:
        """(m, P) signed distance per non-adjacent link pair: each link's sampled
        surface against the other's SDF, negative = interpenetration."""
        n, s = POINTS_PER_LINK, self.sub
        out = []
        for i, j in self.pairs:
            pi, pj = pts[:, i * n : i * n + s], pts[:, j * n : j * n + s]
            dij = self.sdfs[j](self.into(pi, mats, j)).amin(dim=1)
            dji = self.sdfs[i](self.into(pj, mats, i)).amin(dim=1)
            out.append(torch.minimum(dij, dji))
        return torch.stack(out, dim=1)

    @torch.no_grad()
    def calibrate_self(self, q: Tensor, chunk: int = 4096):
        """Pairs that sit close in valid postures get a smaller (or no) gap."""
        lo = torch.cat(
            [
                self.self_dists(*self.forward(q[i : i + chunk])).amin(dim=0)[None]
                for i in range(0, len(q), chunk)
            ]
        ).amin(dim=0)
        self.self_gap = (lo - 0.005).clamp(max=SELF_GAP)

    def cube_penetration(self, pts: Tensor, cube: Tensor) -> Tensor:
        """Arm links (not hand/fingers) inside the held cube: (m,) worst."""
        d = box_sdf(pts[:, self.arm_mask], cube[:, None], self.cube_half)
        return (self.radius - d).clamp(min=0.0).amax(dim=1)

    def body_penetration(self, q: Tensor, cube: Tensor | None) -> tuple[Tensor, Tensor]:
        """(m,) worst self + held-cube penetration per frame, and the points."""
        mats, pts = self.forward(q)
        pen = (self.self_gap - self.self_dists(mats, pts)).clamp(min=0.0).amax(dim=1)
        if cube is not None:
            pen = pen + self.cube_penetration(pts, cube)
        return pen, pts


class AnalyticSelector:
    def __init__(
        self,
        box,  # physical world [center(3), half_extents(3)]
        base_pos,
        joint_stats,  # action stats; leading 7 dims are the arm joints
        obs_stats,  # obs stats; cube pos block normalizes the plan's cube dims
        joint_start: int,
        device,
        cube_size: float,
        cube_index: int | None = 18,
        n_arm: int = 7,
        pointcloud=None,
        hold: float = 0.0,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.device = device
        self.fc = FrankaCollision(device, base_pos, cube_size)
        self.box = torch.as_tensor(box, **f32).view(-1, 6)[:, None]
        self.cloud = None if pointcloud is None else torch.as_tensor(pointcloud, **f32)
        self.jm = torch.as_tensor(joint_stats["mean"], **f32)[:n_arm]
        self.js = torch.as_tensor(joint_stats["std"], **f32)[:n_arm]
        # the plan's predicted joint STATES share the action normalization only
        # if recorded identically; use obs stats for the state block instead
        self.sm = torch.as_tensor(obs_stats["mean"], **f32)[:n_arm]
        self.ss = torch.as_tensor(obs_stats["std"], **f32)[:n_arm]
        ci = cube_index
        self.cm = self.cs = None
        if ci is not None:
            self.cm = torch.as_tensor(obs_stats["mean"], **f32)[ci : ci + 3]
            self.cs = torch.as_tensor(obs_stats["std"], **f32)[ci : ci + 3]
        self.joint_start, self.n_arm, self.cube_index = joint_start, n_arm, ci
        self.rad = self.fc.radius
        self.hold = hold
        self.gm = torch.as_tensor(joint_stats["mean"], **f32)[n_arm]
        self.gs = torch.as_tensor(joint_stats["std"], **f32)[n_arm]

    def set_boxes(self, boxes):
        boxes = np.asarray(boxes, dtype=np.float32)
        t = torch.as_tensor(boxes, device=self.device)
        self.box = t.reshape(-1, 6)[:, None] if t.dim() < 3 else t

    @torch.no_grad()
    def score(self, traj: Tensor) -> Tensor:
        """Total penetration. Ranking on clearance instead games the goal clamp
        -- the safest plan never approaches the cube at all -- so clearance buys
        nothing and every collision-free plan ties."""
        boxes = self.boxes_for(len(traj))
        out = []
        for i in range(0, len(traj), 32):
            c = traj[i : i + 32]
            pen = (-self.clearance_t(c, boxes[i : i + 32])).clamp(min=0.0).sum(dim=1)
            out.append(pen + self.body_penetration(c))
        return torch.cat(out)

    def boxes_for(self, n: int) -> Tensor:
        boxes = self.box
        if len(boxes) == 1:
            return boxes.expand(n, *boxes.shape[1:])
        return boxes.repeat_interleave(n // len(boxes), dim=0)

    def grad(self, x: Tensor) -> Tensor:
        boxes, out = self.boxes_for(len(x)), []
        with torch.enable_grad():
            for i in range(0, len(x), 32):
                c = x[i : i + 32].detach().requires_grad_(True)
                clear = self.clearance_t(c, boxes[i : i + 32], attach=True)
                pen = (-clear).clamp(min=0.0).sum()
                out.append(torch.autograd.grad(pen, c)[0])
        return torch.cat(out)

    def dist(self, points: Tensor, box: Tensor) -> Tensor:
        return obstacle_dist(points, box, self.cloud)

    def joints(self, traj: Tensor) -> tuple[Tensor, Tensor]:
        s, n = self.joint_start, self.n_arm
        return traj[..., s : s + n] * self.js + self.jm, traj[
            ..., :n
        ] * self.ss + self.sm

    def cube(self, traj: Tensor) -> Tensor | None:
        if self.cube_index is None or self.cs is None or self.cm is None:
            return None
        return traj[..., self.cube_index : self.cube_index + 3] * self.cs + self.cm

    def clearance_t(
        self, traj: Tensor, box: Tensor | None = None, attach: bool = False
    ) -> Tensor:
        """Per-timestep worst clearance (b, t): min over arm points, the cube, and
        both joint sources."""
        if box is None:
            box = self.box
        n = self.n_arm
        worst, hand = [], None
        # require both the joint targets and the plan's own predicted states to
        # clear: the two disagree by the controller's tracking lag
        for q in self.joints(traj):
            b, t = q.shape[:2]
            mats, pts = self.fc.forward(q.reshape(-1, n).float())
            pts = pts.reshape(b, t, -1, 3)
            if hand is None:
                hand = mats[:, HAND, :3, 3].reshape(b, t, 3) + self.fc.base_pos
            d = self.dist(pts + self.fc.base_pos, box)
            worst.append((d - self.rad).amin(dim=2))  # (b, t), mesh not centreline
        cube = self.cube(traj)
        if cube is not None and attach and hand is not None:
            cube = hand + (cube.float() - hand).detach()
        if cube is not None and self.hold > 0:
            grip = traj[..., self.joint_start + self.n_arm] * self.gs + self.gm
            d = self.dist(cube.float(), box) - self.hold
            worst.append(torch.where(grip < 0.02, d, torch.ones_like(d)))
        elif cube is not None:
            worst.append(self.dist(cube.float(), box) - self.fc.cube_half)
        return torch.stack(worst).amin(dim=0)

    def body_penetration(self, traj: Tensor) -> Tensor:
        """(b,) self- and arm-vs-held-cube penetration summed over strided frames."""
        q = self.joints(traj)[0][:, ::SELF_STRIDE]
        b, t = q.shape[:2]
        cube = self.cube(traj) if self.fc.cube_half > 0 else None
        if cube is not None:
            cube = (cube[:, ::SELF_STRIDE].float() - self.fc.base_pos).reshape(-1, 3)
        pen, _ = self.fc.body_penetration(q.reshape(-1, self.n_arm).float(), cube)
        return SELF_STRIDE * pen.reshape(b, t).sum(dim=1)


PIPER_SEGMENTS = (
    ("base_link", "link1"),
    ("link2", "link3"),
    ("link3", "link4"),
    ("link5", "link6"),
)
PIPER_FRAMES = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]
PIPER_TIP = 0.1358
PIPER_RADIUS = 0.045
PIPER_SEGMENT_POINTS = 8


class PiperCollision:
    """Capsule centrelines sampled between consecutive piper link origins. The
    URDF's meshes are not vendored, so the link volumes are approximated by a
    single radius about those segments. link6 is where the SDK's end pose and
    bend.py's FK stop, but the gripper reaches PIPER_TIP past it, so the last
    segment runs out to the fingertips."""

    def __init__(
        self,
        device,
        base_pos=(0.0, 0.0, 0.0),
        radius: float = PIPER_RADIUS,
        tip: float = PIPER_TIP,
        n: int = PIPER_SEGMENT_POINTS,
    ):
        self.device = device
        self.chain = build_piper_chain(device)
        self.base_pos = torch.as_tensor(base_pos, dtype=torch.float32, device=device)
        self.radius = radius
        self.tip = tip
        self.pairs = [
            (PIPER_FRAMES.index(a), PIPER_FRAMES.index(b)) for a, b in PIPER_SEGMENTS
        ]
        self.t = torch.linspace(0.0, 1.0, n, device=device)[:, None]

    def frames(self, q: Tensor) -> Tensor:
        fk = self.chain.forward_kinematics(q, end_only=False)
        return torch.stack([fk[name].get_matrix() for name in PIPER_FRAMES], dim=1)

    def tip_point(self, mats: Tensor) -> Tensor:
        m = mats[:, -1]
        return m[:, :3, 3] + self.tip * m[:, :3, 2]

    def arm_points(self, q: Tensor) -> Tensor:
        """q (m, 6) radians -> (m, K, 3) collision points in the base frame."""
        mats = self.frames(q)
        o = mats[:, :, :3, 3]
        ends = [(o[:, a], o[:, b]) for a, b in self.pairs]
        ends.append((o[:, -1], self.tip_point(mats)))
        pts = [a[:, None] + self.t * (b - a)[:, None] for a, b in ends]
        return torch.cat(pts, dim=1) + self.base_pos

    def held_point(self, q: Tensor, offset: float) -> Tensor:
        mats = self.frames(q)
        m = mats[:, -1]
        return m[:, :3, 3] + (self.tip + offset) * m[:, :3, 2] + self.base_pos


class PiperSelector:
    """AnalyticSelector's scoring for the piper arm: the recorded joints are in
    degrees and there is no object block in the observation, so the held item is
    a sphere carried past the fingertips while the gripper reads closed."""

    def __init__(
        self,
        act_stats,
        obs_stats,
        joint_start: int,
        device,
        base_pos=(0.0, 0.0, 0.0),
        n_arm: int = 6,
        hold: float = 0.0,
        hold_offset: float = 0.0,
        grip_closed: float = 1.5,
        radius: float = PIPER_RADIUS,
    ):
        f32 = {"dtype": torch.float32, "device": device}
        self.device = device
        self.fc = PiperCollision(device, base_pos, radius=radius)
        self.jm = torch.as_tensor(act_stats["mean"], **f32)[:n_arm]
        self.js = torch.as_tensor(act_stats["std"], **f32)[:n_arm]
        self.sm = torch.as_tensor(obs_stats["mean"], **f32)[:n_arm]
        self.ss = torch.as_tensor(obs_stats["std"], **f32)[:n_arm]
        self.gm = torch.as_tensor(act_stats["mean"], **f32)[n_arm]
        self.gs = torch.as_tensor(act_stats["std"], **f32)[n_arm]
        self.joint_start, self.n_arm = joint_start, n_arm
        self.hold, self.hold_offset = hold, hold_offset
        self.grip_closed = grip_closed
        self.box = torch.as_tensor([[[1e3, 1e3, 1e3, 0.01, 0.01, 0.01]]], **f32)
        self.cloud = None

    def set_boxes(self, boxes):
        t = torch.as_tensor(np.asarray(boxes, np.float32), device=self.device)
        self.box = t.reshape(-1, 6)[:, None] if t.dim() < 3 else t

    def set_cloud(self, points):
        self.cloud = (
            None
            if points is None or not len(points)
            else torch.as_tensor(np.asarray(points, np.float32), device=self.device)
        )

    def dist(self, points: Tensor, box: Tensor) -> Tensor:
        return obstacle_dist(points, box, self.cloud)

    def joints(self, traj: Tensor) -> tuple[Tensor, Tensor]:
        s, n = self.joint_start, self.n_arm
        act = traj[..., s : s + n] * self.js + self.jm
        obs = traj[..., :n] * self.ss + self.sm
        return torch.deg2rad(act), torch.deg2rad(obs)

    def clearance_t(self, traj: Tensor, box: Tensor | None = None) -> Tensor:
        if box is None:
            box = self.box
        n = self.n_arm
        worst = []
        for q in self.joints(traj):
            b, t = q.shape[:2]
            flat = q.reshape(-1, n).float()
            pts = self.fc.arm_points(flat).reshape(b, t, -1, 3)
            d = self.dist(pts, box)
            worst.append((d - self.fc.radius).amin(dim=2))
            if self.hold > 0:
                held = self.fc.held_point(flat, self.hold_offset).reshape(b, t, 3)
                grip = traj[..., self.joint_start + n] * self.gs + self.gm
                dh = self.dist(held, box) - self.hold
                worst.append(
                    torch.where(grip < self.grip_closed, dh, torch.ones_like(dh))
                )
        return torch.stack(worst).amin(dim=0)

    @torch.no_grad()
    def score(self, traj: Tensor) -> Tensor:
        boxes = self.box
        if len(boxes) == 1:
            boxes = boxes.expand(len(traj), *boxes.shape[1:])
        else:
            boxes = boxes.repeat_interleave(len(traj) // len(boxes), dim=0)
        out = []
        for i in range(0, len(traj), 8):
            c = traj[i : i + 8]
            pen = (-self.clearance_t(c, boxes[i : i + 8])).clamp(min=0.0).sum(dim=1)
            out.append(pen)
        return torch.cat(out)
