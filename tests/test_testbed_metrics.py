import numpy as np
import pandas as pd
import pytest

from scientisttwo.testbed.metrics import (aligned_frames, dice, map_at_20, mean_rmse,
                                          rle_decode, score_frames, smape_plus_one)


def test_identifier_join_is_order_independent_and_exact():
    labels = pd.DataFrame({"id": ["001", "002"], "target": [3.0, 8.0]})
    predictions = pd.DataFrame({"id": ["002", "001"], "target": [7.0, 4.0]})
    assert score_frames("mae", labels, predictions, "id", ["target"]) == 1
    for bad in (predictions.iloc[:1], pd.concat([predictions, predictions.iloc[:1]]),
                predictions.assign(id=["002", "wrong"])):
        with pytest.raises(ValueError):
            aligned_frames(labels, bad, "id", ["target"])


def test_shifted_smape_matches_pinned_formula_and_missing_truth_mask():
    expected = 100 * np.mean([1 / 1.5, 2 / 4])
    assert smape_plus_one([0, 2, np.nan], [1, 4, 999]) == pytest.approx(expected)
    assert smape_plus_one([0], [0]) == 0
    for truth, predictions in (([1], [-1]), ([np.nan], [1]), ([1], [np.nan])):
        with pytest.raises(ValueError):
            smape_plus_one(truth, predictions)


def test_feedback_metric_is_mean_per_target_rmse_not_pooled_rmse():
    assert mean_rmse([[0, 0], [0, 0]], [[0, 2], [0, 2]]) == 1
    with pytest.raises(ValueError):
        mean_rmse([[1, 2]], [[np.inf, 2]])


def test_fathomnet_map_deduplicates_predictions_and_uses_real_labels():
    assert map_at_20(["a b", "c"], ["a a b", "c"]) == pytest.approx(((1 + 2 / 3) / 2 + 1) / 2)
    assert map_at_20(["a"], ["b"]) == 0
    with pytest.raises(ValueError):
        map_at_20(["a"], [" ".join(map(str, range(21)))])


def test_rle_dice_shapes_and_invalid_runs():
    mask = rle_decode("1 2 5 1", (2, 3))
    np.testing.assert_array_equal(mask, [[1, 0, 1], [1, 0, 0]])
    assert dice(mask, mask) == 1
    assert dice(np.zeros((2, 3)), np.zeros((2, 3))) == 1
    for encoding in ("0 2", "1 0", "1 3 3 2", "1", "9 1", "x y"):
        with pytest.raises(ValueError):
            rle_decode(encoding, (2, 3))


def test_metrics_reject_nonfinite_predictions():
    labels = pd.DataFrame({"id": ["a"], "x": [3.0]})
    with pytest.raises(ValueError):
        score_frames("mae", labels, labels.assign(x=np.nan), "id", ["x"])
