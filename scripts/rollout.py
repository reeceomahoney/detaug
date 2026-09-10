import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import cast

import numpy as np
import torch
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.robots import RobotConfig  # noqa: F401
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization

from detaug.cbf import PiperCBF
from detaug.perception.client import ObstacleClient
from detaug.selector import PiperSelector

logger = logging.getLogger(__name__)


@dataclass
class DetAugConfig:
    obstacle_url: str = ""  # empty: no perception, plain unbent rollout
    n_cond: int = 8  # bend candidates scored per replan (plus the zero label)
    bend_min: float = 0.3  # candidate arc range, fraction of the trained maximum
    bend_max: float = 1.0
    collision: str = "box"  # selector geometry: "box" or "pointcloud"
    radius: float = 0.045  # arm capsule radius
    camera_radius: float = 0.035
    margin: float = 0.0  # inflate the obstacle box by this much
    hold: float = 0.0  # >0: held-object sphere radius past the fingertips
    hold_offset: float = 0.0  # held sphere centre past the fingertips
    deviation: float = 0.0  # penalise action distance from the zero-label plan
    resample: bool = False  # fresh candidates every replan instead of latching
    wait: float = 10.0  # seconds to wait for the first obstacle before starting
    max_half: float = 0.15
    max_jump: float = 0.05
    settle: float = 1.0
    recal_timeout: float = 15.0
    recal_attempts: int = 3
    seed: int = 0
    dump: str = ""  # non-empty: write per-replan plans and executed actions here
    cbf: bool = False
    cbf_alpha: float = 10.0
    cbf_margin: float = 0.0
    cbf_arm: bool = False
    cbf_axes: str = ""
    cbf_offset: str = ""
    grip_closed: float = 1.5


@dataclass
class DetAugRolloutConfig(RolloutConfig):
    detaug: DetAugConfig = field(default_factory=DetAugConfig)


def sample_bends(n: int, stats, rng, device, lo: float, hi: float) -> torch.Tensor:
    """Candidates drawn from the labels the checkpoint actually saw, with the
    unbent label first so a clear scene ties at zero cost and argmin falls
    through to no detour.

    augment_piper proposes phi in pi * [bend_min, bend_max] but throws most of
    them away on IK error and jerk, so the realized labels are far smaller than
    the proposal range -- sampling that range conditions the policy off
    distribution and it degenerates. The checkpoint's own bend stats carry the
    surviving support, so bound the arc by those and reject anything outside."""
    low = np.asarray(stats["min"], np.float64).reshape(-1, 2)
    high = np.asarray(stats["max"], np.float64).reshape(-1, 2)
    n_arc = len(low)
    phi_max = hi * np.abs(np.stack([low, high])).max((0, 2))
    phi_min = np.minimum(lo * phi_max, phi_max)
    keep = np.zeros((0, n_arc, 2))
    for _ in range(100):
        if len(keep) >= n:
            break
        phi = rng.uniform(phi_min, phi_max, (8 * n, n_arc))
        theta = rng.uniform(0.0, np.pi, (8 * n, n_arc))
        arc = np.stack([phi * np.cos(theta), phi * np.sin(theta)], axis=2)
        inside = np.all((arc >= low) & (arc <= high), axis=(1, 2))
        keep = np.concatenate([keep, arc[inside]])
    assert len(keep) >= n, "bend stats leave no room to sample candidates"
    bends = np.concatenate([np.zeros((1, n_arc, 2)), keep[:n]]).reshape(n + 1, -1)
    return torch.tensor(bends, dtype=torch.float32, device=device)


def diagnose(
    box: np.ndarray | None,
    reference: np.ndarray | None,
    max_half: float,
    max_jump: float,
) -> str | None:
    if box is None:
        return "no obstacle box published"
    half = box[3:6]
    if float(half[:2].max()) > max_half:
        return (
            f"footprint {2 * half[0]:.3f} x {2 * half[1]:.3f} m exceeds "
            f"{2 * max_half:.3f} m, the tracker is probably masking the table"
        )
    if float(half.min()) <= 0.0:
        return f"degenerate box {box.round(3)}"
    if reference is not None:
        jump = float(np.linalg.norm(box[:3] - reference[:3]))
        if jump > max_jump:
            return (
                f"centre jumped {jump:.3f} m from {reference[:3].round(3)} to "
                f"{box[:3].round(3)}"
            )
    return None


def tracking_views(status: dict | None) -> list[str]:
    if status is None:
        return []
    cameras = status.get("cameras") or {}
    return [name for name, view in cameras.items() if view.get("tracking")]


def aligned_since(status: dict | None, stamp: float) -> bool:
    if status is None:
        return False
    cameras = status.get("cameras") or {}
    return bool(cameras) and all(
        float(view.get("aligned_at", 0.0)) >= stamp for view in cameras.values()
    )


def watch_box(client, seconds: float) -> list[np.ndarray]:
    seen: dict[float, np.ndarray] = {}
    deadline = time.monotonic() + seconds
    while True:
        payload = client.latest()
        if payload is not None and payload.get("box") is not None:
            seen[float(payload.get("stamp", 0.0))] = np.asarray(
                payload["box"], np.float32
            )
        if time.monotonic() >= deadline:
            return list(seen.values())
        time.sleep(0.02)


def settle_box(client, cfg: DetAugConfig) -> tuple[np.ndarray | None, str | None]:
    boxes = watch_box(client, cfg.settle)
    if not boxes:
        return None, "no obstacle box published"
    reason = diagnose(boxes[-1], None, cfg.max_half, cfg.max_jump)
    if reason is not None:
        return boxes[-1], reason
    centres = np.stack(boxes)[:, :3]
    spread = float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
    if spread > cfg.max_jump:
        return boxes[-1], f"centre wandered {spread:.3f} m over {cfg.settle:.1f}s"
    if not tracking_views(client.status()):
        return boxes[-1], "no view is tracking the target, the box is a leftover"
    return boxes[-1], None


def recalibrate(client, timeout: float) -> str | None:
    requested = client.recalibrate()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.status()
        phase = str((status or {}).get("phase", ""))
        if phase.startswith("Realignment failed"):
            return phase
        if aligned_since(status, requested):
            return None
        time.sleep(0.1)
    return f"tracker did not realign within {timeout:.0f}s"


class ObstacleGuard:
    def __init__(self, client, cfg: DetAugConfig):
        self.client = client
        self.cfg = cfg
        self.reference: np.ndarray | None = None
        self.requested = 0.0
        self.aligned = 0.0
        self.last_request = 0.0
        self.recalibrations = 0

    def startup(self) -> np.ndarray:
        for attempt in range(self.cfg.recal_attempts + 1):
            box, reason = settle_box(self.client, self.cfg)
            if reason is None:
                assert box is not None
                self.reference = box
                return box
            if attempt == self.cfg.recal_attempts:
                raise RuntimeError(
                    f"obstacle still looks wrong after {attempt} recalibrations "
                    f"({reason}); check the dashboard, or re-select the target "
                    "in the perception dashboard"
                )
            logger.warning("obstacle looks wrong (%s); recalibrating", reason)
            self.recalibrations += 1
            failure = recalibrate(self.client, self.cfg.recal_timeout)
            if failure is not None:
                logger.warning("recalibration %d failed: %s", attempt + 1, failure)
        raise AssertionError("unreachable")

    def request(self, reason: str) -> None:
        if time.time() - self.last_request < self.cfg.recal_timeout:
            return
        self.last_request = time.time()
        logger.warning("obstacle looks wrong (%s); requesting recalibration", reason)
        try:
            self.requested = self.client.recalibrate()
        except OSError as error:
            logger.warning("could not reach the tracker to recalibrate: %s", error)
            return
        self.aligned = 0.0
        self.recalibrations += 1

    def current(self) -> np.ndarray | None:
        box = self.client.box()
        if self.requested:
            status = self.client.status()
            phase = str((status or {}).get("phase", ""))
            if not self.aligned and aligned_since(status, self.requested):
                self.aligned = time.time()
                self.reference = None
            if not self.aligned:
                if phase.startswith("Realignment failed"):
                    logger.warning("%s; keeping the last accepted box", phase)
                    self.requested = 0.0
                elif time.time() - self.requested > self.cfg.recal_timeout:
                    logger.warning("tracker did not realign; keeping the last box")
                    self.requested = 0.0
                return None
            if time.time() - self.aligned < self.cfg.settle:
                return None
            self.requested = 0.0
        reason = diagnose(box, self.reference, self.cfg.max_half, self.cfg.max_jump)
        if reason is not None:
            self.request(reason)
            return None
        self.reference = box
        return box


def normalizer_stats(preprocessor) -> dict:
    for step in getattr(preprocessor, "steps", ()):
        if isinstance(step, NormalizerProcessorStep):
            return {
                k: {
                    n: np.asarray(
                        t.detach().cpu() if torch.is_tensor(t) else t, np.float32
                    )
                    for n, t in v.items()
                }
                for k, v in step.stats.items()
            }
    raise RuntimeError("preprocessor has no normalizer step to read stats from")


def connect_tracker(cfg: DetAugConfig) -> tuple[ObstacleClient, ObstacleGuard | None]:
    client = ObstacleClient(cfg.obstacle_url)
    client.start()
    logger.info("Waiting up to %.0fs for %s ...", cfg.wait, cfg.obstacle_url)
    if client.wait(cfg.wait) is None:
        client.stop()
        raise RuntimeError(
            f"No obstacle published at {cfg.obstacle_url}. Start the perception "
            "pipeline first: pixi run python -m detaug.perception.track_obstacle"
        )

    guard: ObstacleGuard | None = None
    if cfg.collision != "pointcloud":
        guard = ObstacleGuard(client, cfg)
        try:
            guard.startup()
        except RuntimeError:
            client.stop()
            raise
        if guard.recalibrations:
            logger.info(
                "tracker recalibrated %d times before start", guard.recalibrations
            )
    return client, guard


def current_obstacle(client, guard, cfg: DetAugConfig) -> np.ndarray | None:
    if cfg.collision == "pointcloud":
        return client.cloud()
    assert guard is not None
    box = guard.current()
    if box is None:
        return None
    box = box.copy()
    box[3:6] += cfg.margin
    return box


def attach_selector(ctx, cfg: DetAugConfig):
    policy = ctx.policy.policy
    device = policy.config.device
    rng = np.random.default_rng(cfg.seed)

    if not policy.config.cond_dim:
        raise ValueError("checkpoint has no bend conditioning; nothing to search over")

    stats = normalizer_stats(ctx.policy.preprocessor)
    if "bend" not in stats:
        raise ValueError("checkpoint carries no bend stats to bound the candidates")
    oracle = policy.config.cond_dim == 6
    sel = PiperSelector(
        stats[ACTION],
        stats[OBS_STATE],
        joint_start=policy.state_dim,
        device=device,
        hold=cfg.hold,
        hold_offset=cfg.hold_offset,
        radius=cfg.radius,
        camera_radius=cfg.camera_radius,
    )

    client, guard = connect_tracker(cfg)

    def refresh() -> bool:
        ob = current_obstacle(client, guard, cfg)
        if cfg.collision == "pointcloud":
            sel.set_cloud(ob)
            return ob is not None
        if ob is None:
            return False
        sel.set_boxes(ob)
        return True

    drawn: dict[str, torch.Tensor] = {}

    def draw():
        c = sample_bends(
            cfg.n_cond, stats["bend"], rng, device, cfg.bend_min, cfg.bend_max
        )
        drawn["c"] = c
        return c

    # the plan the selector scored and the motion the arm actually ran are the
    # two things the logs cannot tell apart, so record both
    trace: dict[str, list] = {
        "box": [],
        "labels": [],
        "costs": [],
        "chosen": [],
        "fresh": [],
        "plan": [],
        "obs": [],
        "act": [],
        "step_time": [],
    }

    def score(traj):
        fresh = refresh()
        s = sel.score(traj)
        v = s.detach().cpu().numpy()
        trace["box"].append(sel.box.reshape(-1, sel.box.shape[-1])[0].cpu().numpy())
        trace["labels"].append(drawn["c"].cpu().numpy())
        trace["costs"].append(v)
        trace["fresh"].append(fresh)
        # replicate the policy's latch so the log names the plan that actually
        # runs; the global argmin is not it once a candidate has been latched
        latched = policy.latched_idx
        run = (
            int(latched[0])
            if latched is not None and not cfg.resample
            else int(v.argmin())
        )
        logger.info(
            "replan %d: exec cand %d cost %.3f | best cand %d cost %.3f | zero %.3f%s",
            policy.replans,
            run,
            float(v[run]),
            int(v.argmin()),
            float(v.min()),
            float(v[0]),
            "" if fresh else "  [STALE obstacle]",
        )
        return s

    def box_cond():
        b = sel.box.reshape(-1, sel.box.shape[-1])[0]
        half = b[3:6]
        if b.shape[0] > 6:
            cos, sin = torch.cos(b[6]).abs(), torch.sin(b[6]).abs()
            half = torch.stack(
                [
                    half[0] * cos + half[1] * sin,
                    half[0] * sin + half[1] * cos,
                    half[2],
                ]
            )
        return torch.cat([b[:3], half])[None]

    policy.selector_fn = None if oracle else score
    policy.deviation = cfg.deviation
    if not oracle:
        policy.cond_candidates = draw if cfg.resample else draw()

    inner_chunk = policy.predict_action_chunk

    def predict_action_chunk(batch, **kwargs):
        if oracle:
            fresh = refresh()
            cond = box_cond()
            policy.cond_candidates = cond
            trace["box"].append(sel.box.reshape(-1, sel.box.shape[-1])[0].cpu().numpy())
            trace["labels"].append(cond.cpu().numpy())
            trace["fresh"].append(fresh)
        chunk = inner_chunk(batch, **kwargs)
        if oracle:
            v = sel.score(policy.plan).detach().cpu().numpy()
            trace["costs"].append(v)
            logger.info(
                "replan %d: box %s cost %.3f%s",
                len(trace["costs"]),
                np.round(cond[0].cpu().numpy(), 3),
                float(v[0]),
                "" if fresh else "  [STALE obstacle]",
            )
        if len(trace["costs"]) > len(trace["plan"]):  # a replan just landed
            trace["plan"].append(policy.plan[0].cpu().numpy())
            idx = policy.latched_idx
            trace["chosen"].append(-1 if idx is None else int(idx[0]))
        return chunk

    policy.predict_action_chunk = predict_action_chunk

    engine = ctx.policy.inference
    inner_get = engine.get_action
    obs_stats, act_stats = stats[OBS_STATE], stats[ACTION]

    def get_action(obs_frame):
        action = inner_get(obs_frame)
        if action is None or obs_frame is None:
            return action
        obs = np.asarray(obs_frame[OBS_STATE], np.float32).reshape(-1)
        act = action.detach().cpu().numpy().reshape(-1)
        trace["obs"].append((obs - obs_stats["mean"]) / (obs_stats["std"] + 1e-8))
        trace["act"].append((act - act_stats["mean"]) / (act_stats["std"] + 1e-8))
        trace["step_time"].append(time.time())
        return action

    engine.get_action = get_action

    refresh()
    logger.info(
        "detaug: %d bend candidates + zero, %s geometry, radius %.3f margin %.3f",
        cfg.n_cond,
        cfg.collision,
        cfg.radius,
        cfg.margin,
    )
    b = stats["bend"]
    logger.info(
        "bend support %s..%s, candidates:\n%s",
        np.round(b["min"], 2),
        np.round(b["max"], 2),
        "the observed box"
        if oracle
        else np.round(policy.cond_candidates.cpu().numpy(), 2)
        if torch.is_tensor(policy.cond_candidates)
        else "resampled each replan",
    )
    box = client.box()
    if box is not None:
        logger.info(
            "obstacle box (base frame): centre %s half %s yaw %s",
            box[:3].round(3),
            box[3:6].round(3),
            box[6:].round(3),
        )

    def save():
        if not cfg.dump or not trace["costs"]:
            return
        path = Path(cfg.dump)
        if path.is_dir() or not path.suffix:
            path = path / f"rollout-{time.strftime('%Y%m%d-%H%M%S')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        n = min(len(trace["costs"]), len(trace["plan"]))
        out = {
            "box": np.stack(trace["box"][:n]),
            "labels": np.stack(trace["labels"][:n]),
            "costs": np.stack(trace["costs"][:n]),
            "chosen": np.asarray(trace["chosen"][:n]),
            "fresh": np.asarray(trace["fresh"][:n]),
            "plan": np.stack(trace["plan"][:n]),
            "obs": np.stack(trace["obs"]),
            "act": np.stack(trace["act"]),
            "step_time": np.asarray(trace["step_time"]),
            "n_action_steps": np.asarray(policy.config.n_action_steps),
            "state_dim": np.asarray(policy.state_dim),
            "radius": np.asarray(cfg.radius),
            "camera_radius": np.asarray(cfg.camera_radius),
            "margin": np.asarray(cfg.margin),
        }
        for key in (OBS_STATE, ACTION):
            for stat in ("mean", "std"):
                out[f"{key}.{stat}"] = stats[key][stat]
        np.savez_compressed(path, **out)  # ty: ignore[invalid-argument-type]
        logger.info("wrote %s (%d replans, %d steps)", path, n, len(trace["act"]))

    return client, save


class ObstacleFeed:
    def __init__(self):
        self.ob: np.ndarray | None = None
        self.warned = 0.0


def attach_cbf(ctx, cfg: DetAugConfig, fps: float):
    policy = ctx.policy.policy
    device = policy.config.device
    kw: dict = {}
    if cfg.cbf_axes:
        kw["axes"] = [float(v) for v in cfg.cbf_axes.split(",")]
    if cfg.cbf_offset:
        kw["offset"] = [float(v) for v in cfg.cbf_offset.split(",")]
    cbf = PiperCBF(
        device,
        alpha=cfg.cbf_alpha,
        dt=1.0 / fps,
        arm=cfg.cbf_arm,
        radius=cfg.radius,
        camera_radius=cfg.camera_radius,
        **kw,
    )
    cbf.margin = cfg.cbf_margin
    n_arm = cbf.njoints
    held = cbf.held(cfg.hold_offset)

    client, guard = connect_tracker(cfg)
    state = ObstacleFeed()
    stop = threading.Event()

    def refresh():
        ob = current_obstacle(client, guard, cfg)
        if ob is None:
            if time.time() - state.warned > cfg.recal_timeout:
                state.warned = time.time()
                logger.warning("no fresh obstacle; filtering against the last one")
            return
        last = state.ob
        if last is not None and last.shape == ob.shape and np.array_equal(last, ob):
            return
        if cfg.collision == "pointcloud":
            cbf.set_cloud(ob)
        else:
            cbf.set_boxes([ob[None]])
        state.ob = ob

    def watch():
        while not stop.is_set():
            refresh()
            stop.wait(client.period)

    refresh()
    if state.ob is None:
        raise RuntimeError("tracker published no usable obstacle to filter against")
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()

    trace: dict[str, list] = {
        "obs": [],
        "nominal": [],
        "act": [],
        "h": [],
        "box": [],
        "step_time": [],
    }
    stats = {"steps": 0, "active": 0, "h_min": np.inf}

    engine = ctx.policy.inference
    inner_get = engine.get_action

    def get_action(obs_frame):
        action = inner_get(obs_frame)
        if action is None or obs_frame is None:
            return action
        obs = np.asarray(obs_frame[OBS_STATE], np.float32).reshape(-1)
        nominal = action.detach().cpu().numpy().reshape(-1).copy()
        q = torch.deg2rad(torch.as_tensor(obs[:n_arm], device=device))[None]
        target = torch.deg2rad(torch.as_tensor(nominal[:n_arm], device=device))[None]
        attach = None
        if cfg.hold > 0 and float(nominal[n_arm]) < cfg.grip_closed:
            attach = (held, cfg.hold)
        dq, h = cbf.filter(q, target - q, attach)
        filtered = torch.rad2deg(q + dq)[0].cpu().numpy()
        out = action.clone()
        out[:n_arm] = torch.as_tensor(filtered, dtype=out.dtype)
        moved = float(np.abs(filtered - nominal[:n_arm]).max()) > 1e-4
        stats["steps"] += 1
        stats["active"] += int(moved)
        stats["h_min"] = min(stats["h_min"], float(h[0]))
        if moved and stats["active"] % 20 == 1:
            logger.info(
                "cbf active: h %.3f, max joint edit %.2f deg",
                float(h[0]),
                float(np.abs(filtered - nominal[:n_arm]).max()),
            )
        trace["obs"].append(obs)
        trace["nominal"].append(nominal)
        trace["act"].append(out.numpy().reshape(-1).copy())
        trace["h"].append(float(h[0]))
        ob = state.ob
        trace["box"].append(
            ob if cfg.collision != "pointcloud" else np.full(7, np.nan, np.float32)
        )
        trace["step_time"].append(time.time())
        return out

    engine.get_action = get_action
    logger.info(
        "aegis cbf: alpha %.1f margin %.3f axes %s offset %s arm %s, %s geometry",
        cfg.cbf_alpha,
        cfg.cbf_margin,
        cbf.axes.cpu().numpy().round(3),
        cbf.offset.cpu().numpy().round(3),
        cfg.cbf_arm,
        cfg.collision,
    )
    ob = state.ob
    if cfg.collision != "pointcloud" and ob is not None:
        logger.info(
            "obstacle box (base frame): centre %s half %s yaw %s",
            ob[:3].round(3),
            ob[3:6].round(3),
            ob[6:].round(3),
        )

    def save():
        stop.set()
        thread.join(timeout=2.0)
        if stats["steps"]:
            logger.info(
                "cbf: %d steps, active on %d (%.0f%%), min h %.3f",
                stats["steps"],
                stats["active"],
                100.0 * stats["active"] / stats["steps"],
                stats["h_min"],
            )
        if not cfg.dump or not trace["act"]:
            return
        path = Path(cfg.dump)
        if path.is_dir() or not path.suffix:
            path = path / f"cbf-{time.strftime('%Y%m%d-%H%M%S')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            k: np.stack(v) if k != "step_time" else np.asarray(v)
            for k, v in trace.items()
        }
        out["alpha"] = np.asarray(cfg.cbf_alpha)
        out["margin"] = np.asarray(cfg.cbf_margin)
        out["axes"] = cbf.axes.cpu().numpy()
        out["offset"] = cbf.offset.cpu().numpy()
        np.savez_compressed(path, **out)  # ty: ignore[invalid-argument-type]
        logger.info("wrote %s (%d steps)", path, len(trace["act"]))

    return client, save


@parser.wrap()
def rollout(cfg: DetAugRolloutConfig):
    init_logging()

    if cfg.display_data:
        init_visualization(
            cfg.display_mode,
            session_name="rollout",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = cast(Event, signal_handler.shutdown_event)
    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = create_strategy(cfg.strategy)

    client = save = None
    try:
        if cfg.detaug.obstacle_url and cfg.detaug.cbf:
            client, save = attach_cbf(ctx, cfg.detaug, cfg.fps)
        elif cfg.detaug.obstacle_url:
            client, save = attach_selector(ctx, cfg.detaug)
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if client is not None:
            client.stop()
        if save is not None:
            save()
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization(cfg.display_mode)

    policy = ctx.policy.policy
    if policy.replans:
        logger.info(
            "replans %d, %d with no collision-free candidate",
            policy.replans,
            policy.dirty,
        )


def main():
    register_third_party_plugins()
    rollout()  # ty: ignore[missing-argument]


if __name__ == "__main__":
    main()
