import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CausalForce"))

from train import htsc_loss


def test_htsc_tracks_object_ids_when_slots_swap():
    hx = torch.tensor(
        [
            [[1.0, 0.0], [0.8, 0.2]],
            [[0.6, 0.8], [1.0, 0.0]],  # ids swap slots at t=1
        ],
        requires_grad=True,
    )
    logits = torch.tensor([[3.0, 0.0], [2.0, 0.0]])
    ids = [[10, 20], [20, 10]]

    loss = htsc_loss(hx, logits, ids)

    spatial = F.cosine_similarity(hx[0, 0][None], hx[0, 1][None])[0]
    temporal_10 = F.cosine_similarity(hx[0, 0][None], hx[1, 1][None])[0]
    temporal_20 = F.cosine_similarity(hx[0, 1][None], hx[1, 0][None])[0]
    expected = (spatial - (temporal_10 - temporal_20)).pow(2)
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert hx.grad is not None
    assert torch.isfinite(hx.grad).all()


def test_htsc_empty_pair_returns_differentiable_zero():
    hx = torch.randn(2, 1, 4, requires_grad=True)
    logits = torch.randn(1, 4)

    loss = htsc_loss(hx, logits, [[7], [7]])

    assert loss.item() == 0.0
    loss.backward()
    assert hx.grad is not None
