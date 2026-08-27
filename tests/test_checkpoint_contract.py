import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CausalForce"))

from checkpoint_utils import load_model_checkpoint, load_partial_checkpoint


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


class StageOneWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2, 2)
        self.model.risk_type_head = torch.nn.Linear(2, 4)
        self.model.fc_emb_1 = torch.nn.Linear(2, 2)


class StageTwoWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.backbone = torch.nn.Linear(2, 2)
        self.model.causal_risk_head = torch.nn.Module()
        self.model.causal_risk_head.type_head = torch.nn.Linear(2, 4)
        self.model.causal_risk_head.direct_head = torch.nn.Linear(2, 8)


def test_stage1_to_stage2_transfer_has_explicit_contract(tmp_path):
    source = StageOneWrapper()
    path = tmp_path / "stage1.ckpt"
    torch.save({"state_dict": source.state_dict()}, path)
    target = StageTwoWrapper()

    _, report = load_partial_checkpoint(
        target,
        str(path),
        key_transform=lambda key: key.replace(
            "risk_type_head.", "causal_risk_head.type_head."),
        allowed_missing_prefixes=("model.causal_risk_head.direct_head.",),
        allowed_unexpected_prefixes=("model.fc_emb_1.",),
        required_loaded_prefixes=(
            "model.backbone.",
            "model.causal_risk_head.type_head.",
        ),
    )

    assert report["missing"] == [
        "model.causal_risk_head.direct_head.weight",
        "model.causal_risk_head.direct_head.bias",
    ]
    torch.testing.assert_close(
        target.model.backbone.weight, source.model.backbone.weight)
    torch.testing.assert_close(
        target.model.causal_risk_head.type_head.weight,
        source.model.risk_type_head.weight,
    )


def test_stage1_transfer_rejects_missing_backbone(tmp_path):
    source = StageOneWrapper()
    state = {
        key: value for key, value in source.state_dict().items()
        if not key.startswith("model.backbone.")
    }
    path = tmp_path / "broken.ckpt"
    torch.save({"state_dict": state}, path)

    with pytest.raises(RuntimeError, match="backbone"):
        load_partial_checkpoint(
            StageTwoWrapper(),
            str(path),
            key_transform=lambda key: key.replace(
                "risk_type_head.", "causal_risk_head.type_head."),
            allowed_missing_prefixes=("model.causal_risk_head.direct_head.",),
            allowed_unexpected_prefixes=("model.fc_emb_1.",),
            required_loaded_prefixes=("model.backbone.",),
        )
