import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation

from utils import NUM_CLASSES

# Available Cityscapes semantic variants (ordered by size):
#   facebook/mask2former-swin-tiny-cityscapes-semantic   (~47M params)
#   facebook/mask2former-swin-small-cityscapes-semantic
#   facebook/mask2former-swin-base-IN21k-cityscapes-semantic
#   facebook/mask2former-swin-large-IN21k-cityscapes-semantic
_DEFAULT_CKPT = "facebook/mask2former-swin-tiny-cityscapes-semantic"


class Mask2Former(nn.Module):
    """
    Mask2Former wrapper for Cityscapes fine-tuning.

    The pretrained Cityscapes checkpoint uses the same 19 classes as the target
    dataset, so all weights (including the classification head) are loaded as-is,
    giving the best possible starting point.

    forward() returns (B, C, H, W) raw logits (unnormalized scores), compatible with
    CrossEntropyLoss, softmax-based DiceLoss, and FocalFastCPLoss(from_logits=True).
    """

    def __init__(self, num_classes=NUM_CLASSES, model_name=_DEFAULT_CKPT):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            use_safetensors=True,
            ignore_mismatched_sizes=(num_classes != 19),
        )
        if num_classes != self.model.config.num_labels:
            hidden_dim = self.model.config.hidden_dim
            self.model.class_predictor = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        outputs = self.model(pixel_values=x)

        # class_queries_logits: (B, Q, num_classes + 1) — last dim is "no object"
        class_probs = outputs.class_queries_logits.softmax(-1)[..., :-1]  # (B, Q, C)

        # masks_queries_logits: (B, Q, H/4, W/4) → upsample → sigmoid
        mask_probs = F.interpolate(
            outputs.masks_queries_logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()  # (B, Q, H, W)

        # Weighted sum over queries → per-pixel unnormalized scores (B, C, H, W)
        semantic = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        return semantic