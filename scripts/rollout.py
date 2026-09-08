import logging
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
    margin: float = 0.0  # inflate the obstacle box by this much
    hold: float = 0.0  # >0: held-object sphere radius past the fingertips
    hold_offset: float = 0.0  # held sphere centre past the fingertips
    deviation: float = 0.0  # penalise action distance from the zero-label plan
    resample: bool = False  # fresh candidates every replan instead of latching
    wait: float = 10.0  # seconds to wait for the first obstacle before starting
    freeze: float = 1.5  # seconds to average the box over at start; 0 = track live
    max_half: float = 0.15  # reject a box whose footprint half-extent exceeds this
    max_spread: float = 0.02  # reject a freeze window whose centre moves more
    seed: int = 0
    dump: str = ""  # non-empty: write per-replan plans and executed actions here


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


def freeze_box(client, seconds: float) -> tuple[np.ndarray, float, int]:
    """Average the published box over a window and report how far its centre
    wandered. A static obstacle needs no tracking, and a tracker that loses the
    target reports a confident box somewhere else -- scoring against that is
    indistinguishable from a clear scene, so freeze once and check the spread."""
    seen: dict[float, np.ndarray] = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        payload = client.latest()
        if payload is not None and payload.get("box") is not None:
            seen[float(payload.get("stamp", 0.0))] = np.asarray(
                payload["box"], np.float32
            )
        time.sleep(0.02)
    if not seen:
        raise RuntimeError("no obstacle published during the freeze window")
    boxes = np.stack(list(seen.values()))
    centres = boxes[:, :3]
    spread = float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
    return boxes.mean(0), spread, len(boxes)


def check_box(box: np.ndarray, max_half: float) -> None:
    half = box[3:]
    if float(half[:2].max()) > max_half:
        raise RuntimeError(
            f"obstacle footprint {2 * half[0]:.3f} x {2 * half[1]:.3f} m exceeds "
            f"{2 * max_half:.3f} m -- the tracker is probably masking the table, "
            "re-select the target with: pixi run perception --select-targets"
        )
    if float(half.min()) <= 0.0:
        raise RuntimeError(f"degenerate obstacle box {box.round(3)}")


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


def attach_selector(ctx, cfg: DetAugConfig):
    policy = ctx.policy.policy
    device = policy.config.device
    rng = np.random.default_rng(cfg.seed)

    if not policy.config.cond_dim:
        raise ValueError("checkpoint has no bend conditioning; nothing to search over")

    stats = normalizer_stats(ctx.policy.preprocessor)
    if "bend" not in stats:
        raise ValueError("checkpoint carries no bend stats to bound the candidates")
    sel = PiperSelector(
        stats[ACTION],
        stats[OBS_STATE],
        joint_start=policy.state_dim,
        device=device,
        hold=cfg.hold,
        hold_offset=cfg.hold_offset,
        radius=cfg.radius,
    )

    client = ObstacleClient(cfg.obstacle_url)
    client.start()
    logger.info("Waiting up to %.0fs for %s ...", cfg.wait, cfg.obstacle_url)
    if client.wait(cfg.wait) is None:
        client.stop()
        raise RuntimeError(
            f"No obstacle published at {cfg.obstacle_url}. Start the perception "
            "pipeline first: pixi run python -m detaug.perception.track_obstacle"
        )

    frozen: np.ndarray | None = None
    if cfg.freeze > 0 and cfg.collision != "pointcloud":
        frozen, spread, n = freeze_box(client, cfg.freeze)
        check_box(frozen, cfg.max_half)
        logger.info(
            "froze the obstacle over %.1fs: %d samples, centre spread %.3f m",
            cfg.freeze,
            n,
            spread,
        )
        if spread > cfg.max_spread:
            raise RuntimeError(
                f"obstacle centre moved {spread:.3f} m during the freeze window "
                f"(limit {cfg.max_spread:.3f}); the tracker is not holding the "
                "target, check the dashboard before running the arm"
            )
        frozen = frozen.copy()
        frozen[3:] += cfg.margin
        sel.set_boxes(frozen)

    def refresh() -> bool:
        if frozen is not None:
            return True
        if cfg.collision == "pointcloud":
            cloud = client.cloud()
            sel.set_cloud(cloud)
            return cloud is not None
        box = client.box()
        if box is None:
            return False
        box = box.copy()
        box[3:] += cfg.margin
        sel.set_boxes(box)
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
        trace["box"].append(sel.box.reshape(-1, 6)[0].cpu().numpy())
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

    policy.selector_fn = score
    policy.deviation = cfg.deviation
    policy.cond_candidates = draw if cfg.resample else draw()

    inner = policy.select_action

    def select_action(batch, **kwargs):
        action = inner(batch, **kwargs)
        if len(trace["costs"]) > len(trace["plan"]):  # a replan just landed
            trace["plan"].append(policy.plan[0].cpu().numpy())
            idx = policy.latched_idx
            trace["chosen"].append(-1 if idx is None else int(idx[0]))
        trace["obs"].append(batch[OBS_STATE].detach().cpu().numpy().reshape(-1))
        trace["act"].append(action.detach().cpu().numpy().reshape(-1))
        trace["step_time"].append(time.time())
        return action

    policy.select_action = select_action

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
        np.round(policy.cond_candidates.cpu().numpy(), 2)
        if torch.is_tensor(policy.cond_candidates)
        else "resampled each replan",
    )
    box = frozen if frozen is not None else client.box()
    if box is not None:
        logger.info(
            "obstacle box (base frame, %s): centre %s half %s",
            "frozen" if frozen is not None else "live",
            box[:3].round(3),
            box[3:].round(3),
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
            "margin": np.asarray(cfg.margin),
        }
        for key in (OBS_STATE, ACTION):
            for stat in ("mean", "std"):
                out[f"{key}.{stat}"] = stats[key][stat]
        np.savez_compressed(path, **out)  # ty: ignore[invalid-argument-type]
        logger.info("wrote %s (%d replans, %d steps)", path, n, len(trace["act"]))

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
        if cfg.detaug.obstacle_url:
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
