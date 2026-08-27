"""Static dataset split, label, duplicate, and temporal-horizon audit."""

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np


CATEGORY_INDEX = {"Obs": 0, "Occ": 1, "I": 2, "C": 3}
CATEGORY_NAMES = ("OBS", "OCC", "I", "C")


def risk_type_by_id(metadata):
    result = {}
    for category, raw_ids in metadata.get("risk_id", []):
        if category not in CATEGORY_INDEX:
            continue
        object_ids = raw_ids if isinstance(raw_ids, (list, tuple)) else [raw_ids]
        for object_id in object_ids:
            result[int(object_id)] = CATEGORY_INDEX[category]
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split(name, root, hash_images=False, debug_limit=10):
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{name} root does not exist: {root}")

    scenarios = sorted(
        entry for entry in os.listdir(root)
        if os.path.isdir(os.path.join(root, entry, "rgb_front"))
    )
    windows = []
    targets = []
    image_hashes = {}
    class_count = Counter()
    gt_volume = []
    invalid = []
    temporal_examples = []
    risk_windows = 0
    risk_object_observations = 0

    for scenario in scenarios:
        scenario_root = os.path.join(root, scenario)
        image_root = os.path.join(scenario_root, "rgb_front")
        images = sorted(
            entry for entry in os.listdir(image_root)
            if os.path.isfile(os.path.join(image_root, entry))
        )
        metadata_path = os.path.join(scenario_root, "risk_id.json")
        with open(metadata_path) as stream:
            metadata = json.load(stream)
        category_by_id = risk_type_by_id(metadata)

        for index in range(max(0, len(images) - 2)):
            window = tuple(images[index:index + 3])
            windows.append((scenario,) + window)
            target_image = os.path.join(image_root, window[-1])
            frame_id = os.path.splitext(window[-1])[0]
            targets.append((scenario, frame_id, target_image))
            if hash_images and target_image not in image_hashes:
                image_hashes[target_image] = file_hash(target_image)

            interval_path = os.path.join(
                scenario_root, "risk_interval_H8_new", frame_id + ".npy")
            if not os.path.isfile(interval_path):
                invalid.append(f"missing {interval_path}")
                continue
            intervals = np.load(interval_path, allow_pickle=True).item()
            window_has_risk = False
            for object_key, raw_sequence in intervals.items():
                object_id = int(object_key)
                sequence = np.asarray(raw_sequence)
                if sequence.shape != (8,):
                    invalid.append(
                        f"{interval_path}:{object_id} shape={sequence.shape}")
                    continue
                if not np.isin(sequence, [0, 1]).all():
                    invalid.append(
                        f"{interval_path}:{object_id} is not binary")
                if object_id not in category_by_id:
                    invalid.append(
                        f"{interval_path}:{object_id} has no risk category")
                    continue
                class_count[CATEGORY_NAMES[category_by_id[object_id]]] += 1
                volume = float(sequence.sum())
                gt_volume.append(volume)
                risk_object_observations += 1
                window_has_risk = window_has_risk or volume > 0
                if len(temporal_examples) < debug_limit:
                    temporal_examples.append(
                        (scenario, frame_id, object_id, sequence.astype(int).tolist()))
            risk_windows += int(window_has_risk)

    duplicate_windows = len(windows) - len(set(windows))
    print(f"\n[{name}] root={root}")
    print(f"  scenarios={len(scenarios)}")
    print(f"  sliding_windows={len(windows)}")
    print(f"  duplicate_window_keys={duplicate_windows}")
    print(f"  risk_windows={risk_windows}")
    print(f"  risk_object_observations={risk_object_observations}")
    print("  class_count=" + ", ".join(
        f"{category}:{class_count[category]}" for category in CATEGORY_NAMES))
    if gt_volume:
        print(f"  GT volume mean={np.mean(gt_volume):.4f} "
              f"min={np.min(gt_volume):.0f} max={np.max(gt_volume):.0f}")
    print(f"  invalid_label_records={len(invalid)}")
    for error in invalid[:20]:
        print(f"    INVALID: {error}")
    print("  temporal GT examples:")
    for scenario, frame_id, object_id, sequence in temporal_examples:
        print(f"    {scenario}/{frame_id} object={object_id}: " +
              " ".join(map(str, sequence)))

    return {
        "root": root,
        "scenarios": set(scenarios),
        "targets": {(scenario, frame) for scenario, frame, _ in targets},
        "realpaths": {os.path.realpath(path) for _, _, path in targets},
        "hashes": set(image_hashes.values()),
        "invalid": invalid,
    }


def compare_splits(left_name, left, right_name, right, hash_images):
    scenario_overlap = left["scenarios"] & right["scenarios"]
    target_overlap = left["targets"] & right["targets"]
    realpath_overlap = left["realpaths"] & right["realpaths"]
    print(f"\n[{left_name} vs {right_name}]")
    print(f"  scenario_id_overlap={len(scenario_overlap)}")
    print(f"  scenario_frame_overlap={len(target_overlap)}")
    print(f"  target_realpath_overlap={len(realpath_overlap)}")
    if scenario_overlap:
        print(f"  overlapping scenarios={sorted(scenario_overlap)[:20]}")
    if hash_images:
        content_overlap = left["hashes"] & right["hashes"]
        print(f"  target_image_SHA256_overlap={len(content_overlap)}")
    if scenario_overlap or target_overlap or realpath_overlap:
        raise RuntimeError(
            f"Potential split leakage between {left_name} and {right_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--hash_images", action="store_true")
    args = parser.parse_args()

    splits = {
        "train": audit_split("train", args.train_data, args.hash_images),
        "val": audit_split("val", args.val_data, args.hash_images),
        "test": audit_split("test", args.test_data, args.hash_images),
    }
    compare_splits("train", splits["train"], "val", splits["val"], args.hash_images)
    compare_splits("train", splits["train"], "test", splits["test"], args.hash_images)
    compare_splits("val", splits["val"], "test", splits["test"], args.hash_images)
    invalid_count = sum(len(split["invalid"]) for split in splits.values())
    if invalid_count:
        raise RuntimeError(f"Dataset audit found {invalid_count} invalid label records")
    print("\nDataset audit: PASS")


if __name__ == "__main__":
    main()
