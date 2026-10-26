"""Fixed CIFAR ResNet18. This file is not candidate editable."""


def build_model(task, config, assets):
    if task != "cifar10":
        raise ValueError("This fixed architecture is for CIFAR-10 only")
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
