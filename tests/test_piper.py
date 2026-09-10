import numpy as np

from detaug.bend import carry_segment, fk, solve_seq
from detaug.kinematics import build_piper_chain


def test_carry_segment_is_the_inner_closed_run():
    g = np.array([0.0] * 20 + [8.0] * 10 + [0.0] * 40 + [8.0] * 10 + [0.0] * 20)
    assert carry_segment(g, 1.5) == (30, 70)
    assert carry_segment(np.full(50, 8.0), 1.5) is None


def test_traj_seeded_ik_reproduces_the_demo():
    chain = build_piper_chain("cpu")
    t = np.linspace(0.0, 1.0, 12)[:, None]
    q = np.array([[0.3, 1.2, -0.8, 0.4, -0.6, 0.2]]) * t + np.array(
        [[-0.2, 0.4, -0.3, -0.1, -0.2, 0.5]]
    )
    ee, quat = fk(chain, q, np.zeros(3))
    out = solve_seq(chain, ee[:, None], quat[:, None], q[:, None], 4, 0.05, np.zeros(3))
    assert np.abs(out[:, 0] - q).max() < 1e-5


def test_planned_box_blocks_the_demo_and_the_bend_clears_it():
    import importlib.util

    spec = importlib.util.spec_from_file_location("aug", "scripts/augment.py")
    assert spec is not None and spec.loader is not None
    aug = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aug)

    t = np.linspace(0.0, 1.0, 120)[:, None]
    ee = np.array([[0.25, -0.3, 0.08]]) + t * np.array([[0.0, 0.5, 0.0]])
    tip = ee + np.array([0.08, 0.0, -0.05])
    pts = np.stack([ee, tip], axis=1)
    x = dict(pts=pts, ee=ee, obs_ee=ee)
    cfg = aug.Config(
        env=aug.PiperConfig(),
        bend_min=0.25,
        bend_max=0.6,
        box_xy=(0.025, 0.15),
        box_h=(0.025, 0.15),
        wall_margin=0.07,
        outside_margin=0.02,
        box_clear=0.0,
        box_tries=200,
    )
    radii = np.full(2, 0.045)
    plan = aug.plan_piper_box(
        x, radii, 10.0, np.random.default_rng(0), cfg, [(10, 110)]
    )
    assert plan is not None
    center, half, d = plan
    assert (2 * half <= 0.30 + 1e-9).all()
    assert (aug.sdf_np(pts[10:110], center, half) - radii).min() <= cfg.box_clear
    bent = np.stack([ee + d, tip + d])[:, 10:110]
    assert aug.sdf_np(bent, center, half).min() > cfg.wall_margin
    assert np.abs(d[:10]).max() == 0.0 and np.abs(d[110:]).max() == 0.0
