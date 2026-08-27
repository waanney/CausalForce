import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CausalForce"))

from risk_tube_metrics import (
    binary_iou,
    conformal_risk_mask,
    coverage_contains,
    evaluate_risk_tube,
    mask_interval,
)


@pytest.mark.parametrize(
    "pred, expected",
    [
        ([0, 0, 1, 1, 1, 1, 0, 0], True),
        ([0, 1, 1, 1, 1, 1, 1, 0], True),
        ([0, 0, 0, 1, 1, 1, 0, 0], False),
        ([0, 0, 1, 1, 1, 0, 0, 0], False),
    ],
)
def test_coverage_interval_cases(pred, expected):
    gt = [0, 0, 1, 1, 1, 1, 0, 0]
    assert coverage_contains(pred, gt) is expected


def test_perfect_prediction_has_perfect_metrics():
    gt = [0, 0, 1, 1, 1, 1, 0, 0]
    metrics = evaluate_risk_tube(gt, gt)
    assert metrics == pytest.approx({
        "coverage": 1.0,
        "iou": 1.0,
        "temporal_consistency": 1.0,
        "detect_alignment": 1.0,
        "release_alignment": 1.0,
        "boundary_alignment": 1.0,
        "risk_iou": 1.0,
    })


def test_all_risk_covers_nonempty_gt_with_large_volume():
    gt = [0, 0, 1, 1, 1, 1, 0, 0]
    pred = np.ones(8, dtype=np.int8)
    assert coverage_contains(pred, gt)
    assert pred.sum() == 8


def test_all_safe_does_not_cover_risk_gt():
    gt = [0, 0, 1, 1, 1, 1, 0, 0]
    assert not coverage_contains(np.zeros(8, dtype=np.int8), gt)


def test_iou_exact_and_partial_overlap():
    gt = [0, 0, 1, 1, 1, 1, 0, 0]
    assert binary_iou(gt, gt) == 1.0
    assert binary_iou([0, 0, 0, 1, 1, 1, 0, 0], gt) == pytest.approx(0.75)


def test_conformal_threshold_uses_one_minus_radius():
    scores = [0.12, 0.20, 0.56, 0.81, 0.77, 0.65, 0.28, 0.11]
    q = np.full(8, 0.5)
    pred = conformal_risk_mask(scores, q)
    assert pred.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert mask_interval(pred) == [2, 5]


def test_fragmented_mask_interval_is_inclusive_envelope_only():
    pred = [0, 1, 0, 1, 1, 0, 0, 0]
    assert mask_interval(pred) == [1, 4]
    # Evaluation itself retains both fragments; it does not fill the gap.
    assert binary_iou(pred, [0, 1, 1, 1, 1, 0, 0, 0]) == pytest.approx(0.75)
