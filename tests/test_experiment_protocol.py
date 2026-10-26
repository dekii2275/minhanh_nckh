import copy

import pytest

from experiment_protocol import prepare_penn_splits, protocol_metadata


@pytest.fixture
def splits():
    train = [
        {"seq_id": f"train-{label}-{index}", "label": label}
        for label in range(3)
        for index in range(10)
    ]
    test = [{"seq_id": f"test-{label}", "label": label} for label in range(3)]
    return {
        "train_gt": train,
        "test_gt": copy.deepcopy(test),
        "test_raw_yolo": copy.deepcopy(test),
        "test_calib_yolo": copy.deepcopy(test),
    }


def test_split_is_disjoint_stratified_reproducible_and_preserves_official_test(splits):
    original = copy.deepcopy(splits)
    result = prepare_penn_splits(splits)

    def ids(key):
        return {sample["seq_id"] for sample in result[key]}

    assert not ids("train_gt") & ids("validation_gt")
    assert not (ids("train_gt") | ids("validation_gt")) & ids("test_gt")
    assert ids("train_gt") | ids("validation_gt") == {
        sample["seq_id"] for sample in original["train_gt"]
    }
    assert len(result["validation_gt"]) == 6
    assert {sample["label"] for sample in result["validation_gt"]} == {0, 1, 2}
    assert result == prepare_penn_splits(splits)
    assert result != prepare_penn_splits(splits, seed=7)
    assert splits == original
    for key in ("test_gt", "test_raw_yolo", "test_calib_yolo"):
        assert result[key] == original[key]


@pytest.mark.parametrize("fraction", [-0.1, 0, 1, 1.1])
def test_invalid_fraction_is_rejected(splits, fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        prepare_penn_splits(splits, fraction)


@pytest.mark.parametrize("key", ["train_gt", "test_gt", "test_raw_yolo", "test_calib_yolo"])
def test_duplicate_sequences_are_rejected(splits, key):
    splits[key].append(splits[key][0])
    with pytest.raises(ValueError, match="unique sequence IDs"):
        prepare_penn_splits(splits)


def test_official_train_test_overlap_is_rejected(splits):
    splits["train_gt"][0]["seq_id"] = splits["test_gt"][0]["seq_id"]
    with pytest.raises(ValueError, match="overlap"):
        prepare_penn_splits(splits)


def test_mismatched_modalities_are_rejected(splits):
    splits["test_raw_yolo"].pop()
    with pytest.raises(ValueError, match="match the GT test sequence IDs"):
        prepare_penn_splits(splits)


def test_mismatched_labels_are_rejected(splits):
    splits["test_calib_yolo"][0]["label"] = 9
    with pytest.raises(ValueError, match="labels do not match"):
        prepare_penn_splits(splits)


def test_singleton_class_is_rejected(splits):
    splits["train_gt"][0]["label"] = 99
    with pytest.raises(ValueError, match="at least two"):
        prepare_penn_splits(splits)


def test_metadata_records_exact_splits(splits):
    result = prepare_penn_splits(splits, 0.3, 7)
    metadata = protocol_metadata(result, 0.3, 7)
    assert metadata["seed"] == 7
    assert metadata["validation_fraction"] == 0.3
    assert metadata["checkpoint_selection"] == "validation_gt_top1"
    for key, ids in metadata["sequence_ids"].items():
        assert ids == sorted(sample["seq_id"] for sample in result[key])
