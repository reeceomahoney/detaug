import torch

from detaug.cape import CapeGuidance, PiperCape


def test_cape_grad_reduces_cost():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = {"mean": [0.0] * 8, "std": [1.0] * 8}
    g = CapeGuidance(dev, [0.0, 0.0, 0.0], stats, stats, joint_start=7)
    q = torch.tensor([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.8], device=dev)
    x = torch.cat([q, q]).repeat(1, 5, 1)
    ee = g.ee(x[:, :, :7])[0, 0]
    g.set_boxes(
        [
            [
                *(ee + torch.tensor([0.05, 0.0, 0.0], device=dev)).tolist(),
                0.02,
                0.02,
                0.02,
            ]
        ]
    )
    c0 = g.cost(x)
    assert c0.item() > 0
    c1 = g.cost(x - 0.05 * g.grad(x))
    assert c1.item() < c0.item()
    g.set_boxes([[1e3, 1e3, 1e3, 0.01, 0.01, 0.01]])
    assert g.cost(x).item() == 0 and g.grad(x).abs().sum().item() == 0


def test_piper_cape_grad_reduces_cost():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = {"mean": [0.0] * 7, "std": [1.0] * 7}
    g = PiperCape(dev, stats, stats, joint_start=7)
    q = torch.tensor([0.0, 60.0, -40.0, 0.0, 30.0, 0.0], device=dev)
    x = torch.cat([q, q.new_zeros(1), q, q.new_zeros(1)]).repeat(1, 5, 1)
    ee = g.ee(torch.deg2rad(x[:, :, :6]))[0, 0]
    near = (ee + torch.tensor([0.05, 0.0, 0.0], device=dev)).tolist()
    g.set_boxes([[*near, 0.02, 0.02, 0.02, 0.3]])
    c0 = g.cost(x)
    assert c0.item() > 0
    c1 = g.cost(x - 0.05 * g.grad(x))
    assert c1.item() < c0.item()
    g.set_boxes([[1e3, 1e3, 1e3, 0.01, 0.01, 0.01, 0.0]])
    assert g.cost(x).item() == 0 and g.grad(x).abs().sum().item() == 0
    g.set_cloud([near])
    assert g.cost(x).item() > 0
    g.set_cloud(None)
    assert g.cost(x).item() == 0


def test_piper_cape_grad_under_inference_mode():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = {"mean": [0.0] * 7, "std": [1.0] * 7}
    g = PiperCape(dev, stats, stats, joint_start=7)
    q = torch.tensor([0.0, 60.0, -40.0, 0.0, 30.0, 0.0], device=dev)
    x = torch.cat([q, q.new_zeros(1), q, q.new_zeros(1)]).repeat(1, 5, 1)
    ee = g.ee(torch.deg2rad(x[:, :, :6]))[0, 0]
    near = (ee + torch.tensor([0.05, 0.0, 0.0], device=dev)).tolist()
    with torch.inference_mode():
        g.set_boxes([[*near, 0.02, 0.02, 0.02, 0.0]])
        c0 = g.cost(x)
        x1 = x - 0.05 * g.grad(x)
        assert g.cost(x1).item() < c0.item()
        g.set_cloud([near])
        assert g.grad(x).abs().sum().item() > 0
