import numpy as np
import torch
import torch.nn.functional as F

IGNORE_INDEX = 255

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

NUM_CLASSES = len(CITYSCAPES_CLASSES)  # 19

# Official Cityscapes RGB palette (train IDs 0–18)
CITYSCAPES_COLORS = [
    (128,  64, 128),  # 0  road
    (244,  35, 232),  # 1  sidewalk
    ( 70,  70,  70),  # 2  building
    (102, 102, 156),  # 3  wall
    (190, 153, 153),  # 4  fence
    (153, 153, 153),  # 5  pole
    (250, 170,  30),  # 6  traffic light
    (220, 220,   0),  # 7  traffic sign
    (107, 142,  35),  # 8  vegetation
    (152, 251, 152),  # 9  terrain
    ( 70, 130, 180),  # 10 sky
    (220,  20,  60),  # 11 person
    (255,   0,   0),  # 12 rider
    (  0,   0, 142),  # 13 car
    (  0,   0,  70),  # 14 truck
    (  0,  60, 100),  # 15 bus
    (  0,  80, 100),  # 16 train
    (  0,   0, 230),  # 17 motorcycle
    (119,  11,  32),  # 18 bicycle
]
PALETTE = np.array(CITYSCAPES_COLORS, dtype=np.uint8)

# Approximate pixel-frequency (%) across Cityscapes fine training set
CLASS_APPEARANCE_PERCENTAGE = [
    36.0, 5.5, 18.6, 0.6, 0.6, 0.6, 0.2, 0.6,
    14.5, 1.3, 6.3, 1.4, 0.2, 6.8, 0.4, 0.3, 0.1, 0.2, 0.4,
]

# Median-frequency balancing weights (median_freq / class_freq)
CLASS_WEIGHTS = torch.tensor([
    0.017, 0.109, 0.032, 1.000, 1.000, 1.000, 3.000, 1.000,
    0.041, 0.462, 0.095, 0.429, 3.000, 0.088, 1.500, 2.000, 6.000, 3.000,
    1.500,
])

DATASET_DESCRIPTION = (
    "Cityscapes fine-annotation semantic segmentation: 5,000 urban driving images "
    "(2048×1024) with pixel-precise labels across 19 training classes, captured across 50 European cities. "
    "The dataset focuses on dense urban scenes with frequent pedestrians, cyclists, trams, and "
    "complex street infrastructure."
)

# Maps original Cityscapes label IDs (gtFine_labelIds.png) → train IDs 0–18 (255 = ignore)
# Use gtFine_labelTrainIds.png directly if available — this table is for raw label ID files.
LABEL_ID_TO_TRAIN_ID = {
     7: 0,   # road
     8: 1,   # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}
