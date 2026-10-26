"""Offline datasets and generated problem families; no remote downloads or model calls."""

from functools import lru_cache
import numpy as np
from sklearn import datasets

CLASSIFICATION = (
    "iris",
    "wine",
    "digits",
    "linear",
    "overlap",
    "imbalance",
    "redundant",
    "multiclass",
    "moons",
    "circles",
)
REGRESSION = (
    "linear",
    "sparse",
    "correlated",
    "heteroscedastic",
    "quadratic",
    "cubic",
    "sine",
    "friedman",
    "piecewise",
    "outliers",
)
OPTIMIZATION = {
    "box-isotropic": ("box", 4, 1),
    "box-mild": ("box", 8, 5),
    "box-conditioned": ("box", 16, 30),
    "box-hard": ("box", 24, 100),
    "box-wide": ("box", 48, 20),
    "simplex-isotropic": ("simplex", 4, 1),
    "simplex-mild": ("simplex", 8, 5),
    "simplex-conditioned": ("simplex", 16, 30),
    "simplex-hard": ("simplex", 24, 100),
    "simplex-wide": ("simplex", 48, 20),
}


def frozen(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x.setflags(write=False)
    y.setflags(write=False)
    return x, y


@lru_cache(maxsize=128)
def classification_data(name, data_seed):
    if name not in CLASSIFICATION:
        raise ValueError(f"Unknown classification dataset: {name}")
    if name in {"iris", "wine", "digits"}:
        return frozen(*getattr(datasets, f"load_{name}")(return_X_y=True))
    if name == "moons":
        return frozen(*datasets.make_moons(n_samples=600, noise=0.2, random_state=data_seed))
    if name == "circles":
        return frozen(
            *datasets.make_circles(n_samples=600, noise=0.1, factor=0.4, random_state=data_seed)
        )
    options = {"n_features": 12, "n_informative": 8, "n_redundant": 2, "class_sep": 1.5}
    if name == "overlap":
        options.update(class_sep=0.5, flip_y=0.12)
    if name == "imbalance":
        options.update(weights=[0.85, 0.15], class_sep=1)
    if name == "redundant":
        options.update(n_features=20, n_informative=4, n_redundant=12, class_sep=1)
    if name == "multiclass":
        options.update(n_classes=3, class_sep=1)
    return frozen(*datasets.make_classification(n_samples=600, random_state=data_seed, **options))


@lru_cache(maxsize=128)
def regression_data(name, data_seed):
    if name not in REGRESSION:
        raise ValueError(f"Unknown regression dataset: {name}")
    rng = np.random.default_rng(data_seed)
    x = rng.normal(size=(500, 6))
    noise = rng.normal(size=500)
    if name == "linear":
        y = x @ np.array([2, -1, 0.5, 0, 1, -0.5]) + 0.1 * noise
    elif name == "sparse":
        y = 3 * x[:, 0] - 2 * x[:, 1] + 0.3 * noise
    elif name == "correlated":
        x[:, 2:] += x[:, :1] * 0.9
        y = x[:, 0] + 2 * x[:, 2] - x[:, 4] + 0.2 * noise
    elif name == "heteroscedastic":
        y = x[:, 0] + x[:, 1] + (0.1 + np.abs(x[:, 0])) * noise
    elif name == "quadratic":
        y = x[:, 0] ** 2 + 2 * x[:, 1] * x[:, 2] - x[:, 3] + 0.1 * noise
    elif name == "cubic":
        y = 0.3 * x[:, 0] ** 3 + x[:, 1] ** 2 - 2 * x[:, 2] + 0.2 * noise
    elif name == "sine":
        x = rng.uniform(-3, 3, size=(500, 6))
        y = 2 * np.sin(x[:, 0]) + np.cos(2 * x[:, 1]) + 0.1 * noise
    elif name == "friedman":
        return frozen(
            *datasets.make_friedman1(n_samples=500, n_features=6, noise=0.5, random_state=data_seed)
        )
    elif name == "piecewise":
        y = np.where(x[:, 0] > 0, 2 * x[:, 1], -2 * x[:, 1]) + 0.2 * noise
    else:
        y = 2 * x[:, 0] - x[:, 1] + 0.1 * noise
        y[rng.choice(500, 25, replace=False)] += rng.normal(0, 10, 25)
    return frozen(x, y)


def quadratic_optimum(q, center, constraint):
    if constraint == "box":
        return np.clip(center, 0, 1)
    lo = np.min(q * (center - 1), axis=1) - 1
    hi = np.max(q * center, axis=1) + 1
    for _ in range(100):
        mid = (lo + hi) / 2
        sums = np.maximum(center - mid[:, None] / q, 0).sum(axis=1)
        lo = np.where(sums > 1, mid, lo)
        hi = np.where(sums > 1, hi, mid)
    return np.maximum(center - ((lo + hi) / 2)[:, None] / q, 0)


@lru_cache(maxsize=128)
def optimization_data(name, data_seed):
    if name not in OPTIMIZATION:
        raise ValueError(f"Unknown optimization family: {name}")
    constraint, dimension, condition = OPTIMIZATION[name]
    rng = np.random.default_rng(data_seed)
    q = np.exp(rng.uniform(0, np.log(condition), (240, dimension)))
    center = rng.normal(0.3, 0.7, (240, dimension))
    mode = np.full((240, 1), 0 if constraint == "box" else 1)
    x = np.concatenate([mode, q, center], axis=1)
    return frozen(x, quadratic_optimum(q, center, constraint))
