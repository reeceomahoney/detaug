import torch

from detaug.cape import CapeGuidance


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
