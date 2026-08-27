import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CausalForce"))

from checkpoint_utils import load_model_checkpoint


class StageTwoToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.risk_type_head = torch.nn.Linear(2, 4)
        self.score_head = torch.nn.Linear(2, 8)


def _save(path, model, include_conformal=True):
    checkpoint = {
        "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}
    }
    if include_conformal:
        checkpoint["saocp_class"] = {"OBS": object()}
    torch.save(checkpoint, path)


def test_rejects_stage1_checkpoint_missing_score_head(tmp_path):
    source = StageTwoToyModel()
    checkpoint = {
        "state_dict": {
            f"model.{key}": value
            for key, value in source.state_dict().items()
            if not key.startswith("score_head.")
        },
        "saocp_class": {"OBS": object()},
    }
    path = tmp_path / "stage1.ckpt"
    torch.save(checkpoint, path)

    with pytest.raises(RuntimeError, match="score_head"):
        load_model_checkpoint(StageTwoToyModel(), str(path), require_conformal=True)


def test_rejects_checkpoint_without_conformal_state(tmp_path):
    path = tmp_path / "uncalibrated.ckpt"
    _save(path, StageTwoToyModel(), include_conformal=False)

    with pytest.raises(RuntimeError, match="saocp_class"):
        load_model_checkpoint(StageTwoToyModel(), str(path), require_conformal=True)


def test_loads_complete_stage2_checkpoint(tmp_path):
    source = StageTwoToyModel()
    path = tmp_path / "stage2.ckpt"
    _save(path, source)
    target = StageTwoToyModel()

    checkpoint = load_model_checkpoint(target, str(path), require_conformal=True)

    assert "saocp_class" in checkpoint
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)
