"""Frozen, declarative augmentation. Inputs use the prepared [-1, 1] scale."""


def transform_batch(task, inputs, *, training, config):
    if task != "cifar10" or inputs.ndim != 4 or tuple(inputs.shape[1:]) != (3, 32, 32):
        raise ValueError("CIFAR augmentation requires NCHW 32-by-32 RGB inputs")
    if not training:
        return inputs
    import torch
    import torch.nn.functional as functional

    padding = config["crop_padding"]
    if padding:
        # -1 is black in the converter's uint8 / 127.5 - 1 convention.
        padded = functional.pad(inputs, (padding, padding, padding, padding), value=-1)
        offsets = torch.randint(0, 2 * padding + 1, (len(inputs), 2), device=inputs.device)
        rows = torch.arange(len(inputs), device=inputs.device)[:, None, None, None]
        channels = torch.arange(3, device=inputs.device)[None, :, None, None]
        y = offsets[:, 0, None, None, None] + torch.arange(32, device=inputs.device)[None, None, :, None]
        x = offsets[:, 1, None, None, None] + torch.arange(32, device=inputs.device)[None, None, None, :]
        inputs = padded[rows, channels, y, x]
    probability = config["horizontal_flip_probability"]
    if probability:
        flip = torch.rand(len(inputs), device=inputs.device) < probability
        inputs = torch.where(flip[:, None, None, None], inputs.flip(-1), inputs)
    return inputs
