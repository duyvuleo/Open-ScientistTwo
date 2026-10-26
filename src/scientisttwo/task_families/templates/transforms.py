"""Candidate-editable batch transform; protected labels are never passed here."""


def transform_batch(task, inputs, *, training, config):
    return inputs
