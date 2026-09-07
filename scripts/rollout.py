import logging
from dataclasses import dataclass, field
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
    seed: int = 0


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
    low = np.asarray(stats["min"], np.float64)
    high = np.asarray(stats["max"], np.float64)
    phi_max = hi * float(np.abs(np.stack([low, high])).max())
    phi_min = min(lo * phi_max, phi_max)
    keep = np.zeros((0, 2))
    for _ in range(100):
        if len(keep) >= n:
            break
        phi = rng.uniform(phi_min, phi_max, 8 * n)
        theta = rng.uniform(0.0, np.pi, 8 * n)
        arc = np.stack([phi * np.cos(theta), phi * np.sin(theta)], axis=1)
        inside = np.all((arc >= low) & (arc <= high), axis=1)
        keep = np.concatenate([keep, arc[inside]])
    assert len(keep) >= n, "bend stats leave no room to sample candidates"
    bends = np.concatenate([np.zeros((1, 2)), keep[:n]])
    return torch.tensor(bends, dtype=torch.float32, device=device)


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

    def refresh() -> bool:
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

    def draw():
        return sample_bends(
            cfg.n_cond, stats["bend"], rng, device, cfg.bend_min, cfg.bend_max
        )

    def score(traj):
        fresh = refresh()
        s = sel.score(traj)
        v = s.detach().cpu().numpy()
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
    box = client.box()
    if box is not None:
        logger.info("obstacle box (base frame): %s", box.round(3))
    return client


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

    client = None
    try:
        if cfg.detaug.obstacle_url:
            client = attach_selector(ctx, cfg.detaug)
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if client is not None:
            client.stop()
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
