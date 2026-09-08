import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference.rtc import supports_rtc_inference
from lerobot.utils.constants import OBS_STATE

from detaug.policy import FlowMatchingConfig, FlowMatchingPolicy


def make_policy(rtc: bool) -> FlowMatchingPolicy:
    cfg = FlowMatchingConfig(
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (5,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        goal_dim=2,
        horizon=8,
        n_action_steps=4,
        dim_model=32,
        n_layers=1,
        device="cpu",
        rtc_config=RTCConfig(execution_horizon=4) if rtc else None,
    )
    torch.manual_seed(0)
    return FlowMatchingPolicy(cfg)


def test_policy_declares_rtc_support():
    policy = make_policy(rtc=True)
    assert supports_rtc_inference(policy)
    assert policy.rtc_processor is not None


def test_prefix_guidance_pulls_the_chunk_toward_the_leftover():
    batch = {OBS_STATE: torch.randn(1, 5)}
    prefix = torch.full((4, 2), 3.0)

    plain = make_policy(rtc=False)
    torch.manual_seed(1)
    free = plain.predict_action_chunk(batch)

    guided_policy = make_policy(rtc=True)
    torch.manual_seed(1)
    same = guided_policy.predict_action_chunk(batch, inference_delay=0)
    assert torch.allclose(same, free)
    torch.manual_seed(1)
    guided = guided_policy.predict_action_chunk(
        batch, inference_delay=1, prev_chunk_left_over=prefix
    )

    assert guided.shape == (1, 8, 2)
    assert torch.isfinite(guided).all()
    assert (guided[0, :4] - prefix).norm() < (free[0, :4] - prefix).norm()
    assert torch.allclose(guided[0, 7], free[0, 7], atol=1e-4)


def test_guidance_broadcasts_over_candidates():
    policy = make_policy(rtc=True)
    policy.n_samples = 3
    policy.selector_fn = lambda x: x[..., 3:].abs().sum((1, 2))
    batch = {OBS_STATE: torch.randn(2, 5)}
    out = policy.predict_action_chunk(
        batch, inference_delay=0, prev_chunk_left_over=torch.zeros(4, 2)
    )
    assert out.shape == (2, 8, 2)
