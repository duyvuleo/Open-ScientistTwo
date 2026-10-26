"""Independent finite-valued scoring; never import submitted training code."""

import numpy as np
import pandas as pd


def aligned_frames(labels, predictions, id_column, targets):
    for name, frame in (("labels", labels), ("predictions", predictions)):
        if not {id_column, *targets}.issubset(frame.columns):
            raise ValueError(f"{name} missing ID or target columns")
        ids = frame[id_column].astype(str)
        if frame[id_column].isna().any() or ids.duplicated().any():
            raise ValueError(f"{name} has missing or duplicate IDs")
    expected = labels[id_column].astype(str)
    actual = predictions[id_column].astype(str)
    if set(expected) != set(actual):
        raise ValueError("Prediction IDs do not exactly match protected labels")
    right = predictions.copy()
    right[id_column] = actual
    right = right.set_index(id_column).loc[expected]
    return labels[targets].to_numpy(), right[targets].to_numpy()


def _finite(values):
    result = np.asarray(values, dtype=np.float64)
    if not result.size or not np.isfinite(result).all():
        raise ValueError("Metric inputs must be nonempty and finite")
    return result


def smape_plus_one(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("SMAPE arrays have incompatible shapes")
    # Upstream intentionally excludes unavailable clinical targets.
    mask = np.isfinite(truth)
    truth, prediction = _finite(truth[mask]), _finite(prediction[mask])
    if (truth < 0).any() or (prediction < 0).any():
        raise ValueError("Shifted SMAPE requires nonnegative targets/predictions")
    return float(100 * np.mean(np.abs(prediction - truth) / ((truth + prediction) / 2 + 1)))


def mean_rmse(truth, prediction):
    truth, prediction = _finite(truth), _finite(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Mean RMSE requires equally shaped multi-target matrices")
    return float(np.mean(np.sqrt(np.mean((truth - prediction) ** 2, axis=0))))


def map_at_20(truth, prediction):
    if len(truth) != len(prediction) or not len(truth):
        raise ValueError("MAP inputs must have the same nonzero number of rows")
    scores = []
    for expected, ranked in zip(truth, prediction):
        relevant = set(str(expected).split())
        items = str(ranked).split()
        if not relevant or not items or len(items) > 20:
            raise ValueError("Category labels and one to twenty ranked predictions are required")
        hits, score, seen = 0, 0.0, set()
        for rank, item in enumerate(items, 1):
            if item in relevant and item not in seen:
                hits += 1
                score += hits / rank
            seen.add(item)
        scores.append(score / min(len(relevant), 20))
    return float(np.mean(scores))


def rle_decode(value, shape=(256, 256)):
    if not isinstance(value, str):
        if value is None or pd.isna(value):
            value = ""
        else:
            raise ValueError("RLE must be a string of integer start/length pairs")
    parts = value.split()
    if len(parts) % 2:
        raise ValueError("RLE has an unmatched start")
    try:
        values = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError("RLE entries must be integers") from error
    total = int(np.prod(shape))
    flat = np.zeros(total, dtype=np.uint8)
    previous_end = 0
    for start, length in zip(values[::2], values[1::2]):
        start -= 1
        if length <= 0 or start < previous_end or start + length > total:
            raise ValueError("RLE contains invalid, overlapping or out-of-bounds runs")
        flat[start:start + length] = 1
        previous_end = start + length
    return flat.reshape(shape, order="F")


def dice(truth, prediction):
    truth, prediction = _finite(truth), _finite(prediction)
    if truth.shape != prediction.shape or not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Dice requires matched binary masks")
    return float((2 * np.sum(truth * prediction) + 1e-6) / (truth.sum() + prediction.sum() + 1e-6))


def score_frames(metric, labels, predictions, id_column, targets):
    truth, prediction = aligned_frames(labels, predictions, id_column, targets)
    if metric == "smape_plus_one":
        return smape_plus_one(truth, prediction)
    if metric == "mean_rmse":
        return mean_rmse(truth, prediction)
    if metric == "map_at_20":
        return map_at_20(truth.ravel(), prediction.ravel())
    if metric == "accuracy":
        if pd.isna(truth).any() or pd.isna(prediction).any():
            raise ValueError("Classification labels must be present")
        return float(np.mean(truth.astype(str) == prediction.astype(str)))
    truth, prediction = _finite(truth), _finite(prediction)
    if metric == "mae":
        return float(np.mean(np.abs(truth - prediction)))
    raise ValueError(f"Unsupported frame metric: {metric}")
