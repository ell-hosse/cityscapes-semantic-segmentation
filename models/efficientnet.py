import torch.nn as nn
import segmentation_models_pytorch as smp

from utils import NUM_CLASSES


class EfficientNet(nn.Module):
    """
    UNet with EfficientNet-B4 encoder pretrained on ImageNet.
    forward() returns (B, num_classes, H, W) logits — compatible with
    CrossEntropyLoss and DiceLoss used in the training scripts.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)