import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CausalForce"))

from data import MultipleRisksDataset


def test_risk_categories_are_looked_up_by_object_id_not_bbox_order():
    metadata = {
        "risk_id": [
            ["Obs", [7, 8]],
            ["I", 42],
            ["C", 99],
            ["Occ", [11, 12]],
        ]
    }
    by_id = MultipleRisksDataset.get_risk_type_by_id(metadata)

    # Simulate risk_interval_H8 keys in an order unrelated to metadata/bboxes.
    interval_object_ids = [99, 11, 42, 8]
    aligned_labels = [by_id[object_id] for object_id in interval_object_ids]

    assert aligned_labels == [3, 1, 2, 0]


def test_occlusion_assigns_both_participating_object_ids():
    metadata = {"risk_id": [["Occ", [101, 202]]]}
    assert MultipleRisksDataset.get_risk_type_by_id(metadata) == {
        101: 1,
        202: 1,
    }
