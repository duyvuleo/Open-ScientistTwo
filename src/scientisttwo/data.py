"""Trusted classification data preparation with frozen task-level partitions."""

from .catalog import classification_data
from .models import EvalSpec
from .splits import partition, prepare_arrays


def split_indices(y, seed: int):
    """Compatibility helper: seed here is a partition seed, never a training seed."""
    return partition(y, seed, stratified=True)


def prepare(spec: EvalSpec):
    """Evaluation labels remain on the trusted host; membership is fixed across all model seeds."""
    yield from prepare_arrays(spec, classification_data, stratified=True)
