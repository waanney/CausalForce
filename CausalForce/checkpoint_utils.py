"""Checkpoint loading helpers with explicit model-contract validation."""

from collections import OrderedDict
from typing import Mapping, Sequence

import torch


def extract_model_state_dict(checkpoint: Mapping, prefix: str = "model."):
    """Extract and normalize a Lightning or plain PyTorch state dict.

    Lightning checkpoints produced by this repository store model parameters
    under ``state_dict`` and prefix them with ``model.``.  A duplicate key
    after prefix removal is always an invalid checkpoint.
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a mapping state_dict")

    normalized = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[len(prefix):] if key.startswith(prefix) else key
        if new_key in normalized:
            raise ValueError(
                f"Checkpoint key collision after removing {prefix!r}: {new_key}")
        normalized[new_key] = value
    return normalized


def _format_keys(keys: Sequence[str]) -> str:
    return "\n".join(f"  - {key}" for key in keys) if keys else "  (none)"


def load_model_checkpoint(
    model,
    checkpoint_path: str,
    *,
    map_location="cpu",
    require_conformal: bool = False,
):
    """Load an inference checkpoint and reject any architecture mismatch.

    In particular, this prevents a Stage-1 classifier checkpoint from being
    accepted by a Stage-2 risk model with a randomly initialized score head.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Invalid checkpoint object in {checkpoint_path}")
    if require_conformal and "saocp_class" not in checkpoint:
        raise RuntimeError(
            "Checkpoint has no 'saocp_class' state. Expected a calibrated "
            "Stage-2 checkpoint, not a Stage-1 classifier checkpoint."
        )

    state_dict = extract_model_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    print(f"Loaded parameters: {len(state_dict) - len(unexpected)}", flush=True)
    print(f"Missing keys:\n{_format_keys(missing)}", flush=True)
    print(f"Unexpected keys:\n{_format_keys(unexpected)}", flush=True)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model contract mismatch. Refusing to evaluate with "
            "missing or unexpected weights.\n"
            f"Missing keys:\n{_format_keys(missing)}\n"
            f"Unexpected keys:\n{_format_keys(unexpected)}"
        )
    return checkpoint
