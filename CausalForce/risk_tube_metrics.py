"""Pure, testable Risk Tube construction and metric definitions."""

import numpy as np


def _as_1d(values, name, dtype=None):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def conformal_risk_mask(scores, quantiles):
    """Include risk label 1 when ``|1-score| <= q``.

    For absolute nonconformity this is equivalent to ``score >= 1-q``.
    ``q`` is the upper conformal radius returned by ``SAOCP.predict()[1]``;
    it is not itself a probability threshold.
    """
    scores = _as_1d(scores, "scores", dtype=float)
    quantiles = _as_1d(quantiles, "quantiles", dtype=float)
    if scores.shape != quantiles.shape:
        raise ValueError(
            f"scores and quantiles must match: {scores.shape} != {quantiles.shape}")
    if not np.isfinite(scores).all() or not np.isfinite(quantiles).all():
        raise ValueError("scores and quantiles must be finite")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("risk scores must lie in [0, 1]")
    if (quantiles < 0).any():
        raise ValueError("conformal radii must be non-negative")
    return (scores >= (1.0 - quantiles)).astype(np.int8)


def mask_interval(mask):
    """Return inclusive [start, end] around all positives, or None."""
    mask = _as_1d(mask, "mask", dtype=np.int8)
    positive = np.flatnonzero(mask == 1)
    if len(positive) == 0:
        return None
    return [int(positive[0]), int(positive[-1])]


def coverage_contains(pred, gt):
    """Whether every positive GT timestep is included in prediction."""
    pred = _as_1d(pred, "pred", dtype=np.int8)
    gt = _as_1d(gt, "gt", dtype=np.int8)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must match: {pred.shape} != {gt.shape}")
    if not np.any(gt == 1):
        raise ValueError("coverage is undefined for an empty GT risk tube")
    return bool(np.all(pred[gt == 1] == 1))


def binary_iou(pred, gt):
    pred = _as_1d(pred, "pred", dtype=np.int8).astype(bool)
    gt = _as_1d(gt, "gt", dtype=np.int8).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must match: {pred.shape} != {gt.shape}")
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def temporal_consistency(pred, gt):
    """Penalize the absolute difference in binary switch counts."""
    pred = _as_1d(pred, "pred", dtype=np.int8)
    gt = _as_1d(gt, "gt", dtype=np.int8)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must match: {pred.shape} != {gt.shape}")
    if len(gt) < 2:
        return 1.0
    pred_switches = np.count_nonzero(pred[1:] != pred[:-1])
    gt_switches = np.count_nonzero(gt[1:] != gt[:-1])
    return float(max(0.0, 1.0 - abs(pred_switches - gt_switches) / (len(gt) - 1)))


def pic_star(pred, gt, tau=2.0, mode="detect"):
    """Exponentially boundary-weighted binary agreement."""
    pred = _as_1d(pred, "pred", dtype=np.int8)
    gt = _as_1d(gt, "gt", dtype=np.int8)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must match: {pred.shape} != {gt.shape}")
    positives = np.flatnonzero(gt == 1)
    if len(positives) == 0:
        return 1.0
    if mode not in {"detect", "release"}:
        raise ValueError(f"Unknown boundary mode: {mode}")
    center = positives[0] if mode == "detect" else positives[-1]
    weights = np.exp(-np.abs(np.arange(len(gt)) - center) / tau)
    agreement = 1 - np.abs(gt - pred)
    return float((weights * agreement).sum() / weights.sum())


def evaluate_risk_tube(pred, gt, tau=2.0):
    """Return per-object metrics; Risk-IoU is averaged per object by callers."""
    iou = binary_iou(pred, gt)
    tc = temporal_consistency(pred, gt)
    detect = pic_star(pred, gt, tau=tau, mode="detect")
    release = pic_star(pred, gt, tau=tau, mode="release")
    ba = 0.5 * (detect + release)
    return {
        "coverage": float(coverage_contains(pred, gt)),
        "iou": iou,
        "temporal_consistency": tc,
        "detect_alignment": detect,
        "release_alignment": release,
        "boundary_alignment": ba,
        "risk_iou": iou * (tc + ba) / 2.0,
    }
