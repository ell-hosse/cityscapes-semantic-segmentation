import torch.nn as nn
import torchvision.models.segmentation as seg_models

from utils import NUM_CLASSES


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = seg_models.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        self.model = seg_models.deeplabv3_resnet101(weights=weights)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]