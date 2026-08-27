"""One-real-batch Stage-2 checkpoint, forward, loss, and gradient audit."""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from causal_model import CausalGCN_model
from causal_modules import counterfactual_loss, orthogonality_loss
from checkpoint_utils import load_model_checkpoint
from data import MultipleRisksDataset, custom_collate_fn
from train import htsc_loss


def describe(name, tensor):
    values = tensor.detach().float()
    print(f"{name}: shape={tuple(tensor.shape)} min={values.min().item():.6f} "
          f"max={values.max().item():.6f} mean={values.mean().item():.6f} "
          f"std={values.std(unbiased=False).item():.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultipleRisksDataset(args.data_root)
    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no sliding-window samples")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn)
    batch = next(iter(loader))
    print(f"Dataset sample loading: PASS ({len(dataset)} windows)", flush=True)

    model = CausalGCN_model(pretrained=False)
    checkpoint = load_model_checkpoint(
        model, args.checkpoint, map_location="cpu", require_conformal=True)
    print("Checkpoint loading: PASS", flush=True)
    for parameter in model.causal_risk_head.type_head.parameters():
        parameter.requires_grad = False
    model.to(device)
    model.train()

    front_imgs = batch["front_imgs"].to(device)
    trackers = [
        [frame.to(device) for frame in sample]
        for sample in batch["all_objs_bbs"]
    ]
    outputs = model(front_imgs, trackers)
    print("Forward pass: PASS", flush=True)
    for key in (
            "score_H8", "risk_type", "hx_seq", "causal_feat",
            "scene_feat", "direct", "indirect"):
        describe(key, outputs[key])

    valid = outputs["valid_mask"]
    if not valid.any():
        raise RuntimeError("First validation batch contains no valid objects")
    targets = torch.zeros_like(outputs["score_H8"])
    type_logits = []
    type_targets = []
    htsc_terms = []
    for batch_index in range(front_imgs.shape[0]):
        ids = batch["all_objs_id"][batch_index][-1][
            :outputs["score_H8"].shape[1]]
        risk_ids = batch["risk_id"][batch_index]
        intervals = batch["risk_interval_H8"][batch_index]
        risk_types = batch["label_risk_type"][batch_index]
        for slot, object_id in enumerate(ids):
            if object_id in risk_ids:
                gt_index = risk_ids.index(object_id)
                targets[batch_index, slot] = intervals[gt_index].to(device)
                type_logits.append(outputs["risk_type"][batch_index, slot])
                type_targets.append(risk_types[gt_index].to(device))
        htsc_terms.append(htsc_loss(
            outputs["hx_seq"][batch_index],
            outputs["risk_type"][batch_index],
            batch["all_objs_id"][batch_index]))

    score_loss = F.binary_cross_entropy(
        outputs["score_H8"][valid].float(), targets[valid].float())
    ortho_loss = orthogonality_loss(
        outputs["causal_feat"][valid], outputs["scene_feat"][valid])
    cf_loss = counterfactual_loss(
        outputs["direct"][valid], outputs["indirect"][valid], targets[valid])
    htsc = torch.stack(htsc_terms).mean()
    if type_logits:
        classification_loss = F.cross_entropy(
            torch.stack(type_logits), torch.stack(type_targets).long())
    else:
        classification_loss = outputs["risk_type"].sum() * 0.0
    total = score_loss + 10.0 * htsc + 0.1 * ortho_loss + 0.5 * cf_loss
    print(f"classification_loss={classification_loss.item():.6f} (diagnostic only)")
    print(f"risk_loss={score_loss.item():.6f}")
    print(f"HTSC_loss={htsc.item():.6f}")
    print(f"orthogonality_loss={ortho_loss.item():.6f}")
    print(f"counterfactual_loss={cf_loss.item():.6f}")
    print(f"total_training_loss={total.item():.6f}")
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite loss")
    total.backward()
    print("Backward pass: PASS", flush=True)

    groups = (
        "backbone", "object_backbone", "lstm", "causal_disentangle",
        "causal_message_passing", "out_layer", "causal_risk_head")
    for group in groups:
        parameters = [
            (name, parameter) for name, parameter in model.named_parameters()
            if name.startswith(group) and parameter.requires_grad
        ]
        with_grad = [parameter for _, parameter in parameters if parameter.grad is not None]
        grad_norm = torch.sqrt(sum(
            parameter.grad.detach().float().pow(2).sum()
            for parameter in with_grad)).item() if with_grad else 0.0
        missing = [name for name, parameter in parameters if parameter.grad is None]
        print(f"gradient {group}: norm={grad_norm:.6e} "
              f"with_grad={len(with_grad)}/{len(parameters)}", flush=True)
        for name in missing[:20]:
            print(f"  grad=None: {name}", flush=True)

    print("SAOCP state:", flush=True)
    for risk_class, cp in checkpoint["saocp_class"].items():
        q = [cp.predict(horizon=t + 1)[1] for t in range(8)]
        horizon_map = getattr(
            getattr(cp, "residuals", None), "horizon2residuals", {})
        counts = [len(horizon_map.get(t + 1, [])) for t in range(8)]
        print(f"  {risk_class} coverage={getattr(cp, 'coverage', None)} q=" +
              " ".join(f"{value:.6f}" for value in q), flush=True)
        print(f"    calibration counts={counts}", flush=True)
        if any(count == 0 for count in counts):
            raise RuntimeError(
                f"Empty SAOCP calibration bucket for {risk_class}: {counts}")


if __name__ == "__main__":
    main()
