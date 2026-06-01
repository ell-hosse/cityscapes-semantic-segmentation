import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from utils import NUM_CLASSES

# Cityscapes fine-tuned checkpoints (ordered by size):
#   nvidia/segformer-b0-finetuned-cityscapes-1024-1024   (~4M params)
#   nvidia/segformer-b1-finetuned-cityscapes-1024-1024
#   nvidia/segformer-b2-finetuned-cityscapes-1024-1024   (best accuracy/speed trade-off)
#   nvidia/segformer-b3-finetuned-cityscapes-1024-1024
#   nvidia/segformer-b4-finetuned-cityscapes-1024-1024
#   nvidia/segformer-b5-finetuned-cityscapes-1024-1024   (~85M params)
_DEFAULT_CKPT = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"


class SegFormer(nn.Module):
    """
    SegFormer wrapper for Cityscapes fine-tuning.

    Loads a checkpoint already fine-tuned on Cityscapes (19 classes), so all
    weights including the decode head are preserved — ideal starting point.

    forward() returns (B, num_classes, H, W) logits upsampled to input resolution.
    """

    def __init__(self, num_classes=NUM_CLASSES, model_name=_DEFAULT_CKPT):
        super().__init__()
        config = SegformerConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=(num_classes != 19),
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=(num_classes != 19),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        logits = self.model(pixel_values=x).logits  # (B, num_classes, H/4, W/4)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)