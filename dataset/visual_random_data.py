"""
Cityscapes data loaders (train / val) using the colored ground-truth labels
(`*_gtFine_color.png`), plus a quick visualization of a few training samples.
"""

import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# -------- Paths --------
DATA_ROOT = r"D:\cityscapes"
IMG_ROOT = os.path.join(DATA_ROOT, "leftImg8bit_trainvaltest", "leftImg8bit")
LBL_ROOT = os.path.join(DATA_ROOT, "gtFine_trainvaltest", "gtFine")


# -------- Dataset --------
class CityscapesColorDataset(Dataset):
    """
    Returns (image, color_label) pairs.
    `color_label` is the official Cityscapes RGB visualization
    (`*_gtFine_color.png`) — same spatial size as the image.
    """

    def __init__(self, img_root, lbl_root, split="train", img_size=(512, 1024)):
        assert split in {"train", "val", "test"}
        self.split = split
        self.img_size = img_size  # (H, W)

        img_dir = os.path.join(img_root, split)
        lbl_dir = os.path.join(lbl_root, split)

        self.img_paths = sorted(
            glob(os.path.join(img_dir, "*", "*_leftImg8bit.png"))
        )
        self.lbl_paths = []
        for p in self.img_paths:
            city = os.path.basename(os.path.dirname(p))
            base = os.path.basename(p).replace("_leftImg8bit.png", "_gtFine_color.png")
            self.lbl_paths.append(os.path.join(lbl_dir, city, base))

        # ImageNet normalization (common for pretrained backbones)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.img_paths)

    def _joint_transform(self, image, label):
        # Resize both — bilinear for image, nearest for label (preserve colors)
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        # Horizontal flip augmentation for train only
        if self.split == "train" and random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        return image, label

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        label = Image.open(self.lbl_paths[idx]).convert("RGB")  # colored label

        image, label = self._joint_transform(image, label)

        image_t = self.normalize(TF.to_tensor(image))          # (3, H, W) normalized
        label_t = torch.from_numpy(np.array(label)).permute(2, 0, 1)  # (3, H, W) uint8

        return image_t, label_t


# -------- DataLoader factory --------
def get_dataloaders(batch_size=4, num_workers=2, img_size=(512, 1024)):
    train_set = CityscapesColorDataset(IMG_ROOT, LBL_ROOT, "train", img_size)
    val_set = CityscapesColorDataset(IMG_ROOT, LBL_ROOT, "val", img_size)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# -------- Visualization --------
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


def visualize_batch(loader, n_samples=4, save_path=None):
    images, labels = next(iter(loader))
    n_samples = min(n_samples, images.size(0))

    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for i in range(n_samples):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        lbl = labels[i].permute(1, 2, 0).numpy().astype(np.uint8)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(lbl)
        axes[i, 1].set_title(f"Colored Label {i}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4, num_workers=0)
    print(f"Train samples: {len(train_loader.dataset)} | batches: {len(train_loader)}")
    print(f"Val   samples: {len(val_loader.dataset)} | batches: {len(val_loader)}")

    visualize_batch(train_loader, n_samples=4, save_path="train_samples.png")
