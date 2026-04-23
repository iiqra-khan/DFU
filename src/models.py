from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torchvision
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
)
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
)


def build_segmentation_model(model_name: str) -> nn.Module:
    if model_name == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT
        )
        model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        return model

    if model_name == "fcn_resnet50":
        model = torchvision.models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)
        return model

    raise ValueError(f"Unsupported segmentation model: {model_name}")


def build_classifier(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported classification model: {model_name}")


def forward_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output: Any = model(inputs)
    if isinstance(output, dict):
        return output["out"]
    return output
