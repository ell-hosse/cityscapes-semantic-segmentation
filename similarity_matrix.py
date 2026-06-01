import torch
from sentence_transformers import SentenceTransformer
from utils import *

def llm_based(classes):
    prompt = f"""I am working on a semantic segmentation task using the Cityscapes dataset.

Dataset description:
{DATASET_DESCRIPTION}

Target classes (in order):
{classes}

Task:
Generate a {len(classes)} × {len(classes)} asymmetric penalty-tolerance matrix S for an urban driving segmentation model.

Definition of S[i][j]:
  S[i][j] = how ACCEPTABLE (i.e., low-risk / low-penalty) it is when the model predicts class j but the TRUE label is class i.
  - Row index i = the true class (ground truth).
  - Column index j = the predicted class (model output).
  - S[i][i] = 1.0 for all i (correct prediction, zero penalty).
  - S[i][j] close to 1 → confusing true-i with pred-j is relatively safe or understandable.
  - S[i][j] close to 0 → confusing true-i with pred-j is dangerous or catastrophic.

IMPORTANT — the matrix must be ASYMMETRIC because the direction of confusion matters:
  Example 1: True=person, Pred=road    → S[person][road]    ≈ x_1
             (The car may drive through the person. Catastrophic.)
  Example 2: True=road,   Pred=person  → S[road][person]    ≈ x_2
             (The car brakes unnecessarily. Annoying but safe.)
  Example 3: True=person, Pred=car     → S[person][car]     ≈ x_3
             (A pedestrian misread as a vehicle — very risky.)
  Example 4: True=car,    Pred=truck   → S[car][truck]      ≈ x_4
             (Both are large vehicles; minor planning impact.)
  Example 5: True=rider,  Pred=bicycle → S[rider][bicycle]  ≈ x_5
             (Losing the human part of a rider is moderately risky.)
  Example 6: True=train,  Pred=bus     → S[train][bus]      ≈ x_6
             (Cityscapes contains trams/trains on fixed tracks — confusing with bus is less
              dangerous than misclassifying as a non-vehicle class.)

Guiding principles:
  - Misclassifying a vulnerable road user (person, rider) as background/road/building is always near 0.
  - Misclassifying static infrastructure (road, building) as another static class is less dangerous than
    misclassifying it as a dynamic object, and vice-versa.
  - False positives for safety-critical classes (person, rider) are tolerable; false negatives are not.
  - Cityscapes is a dense European urban dataset: pedestrians and cyclists are very frequent — weight
    those confusion directions accordingly.
  - Do NOT make the matrix symmetric. Carefully reason about each direction separately.

Requirements:
  - Output ONLY the raw {len(classes)} × {len(classes)} 2D array. No text, no comments, no variable names.
  - Values in [0.0, 1.0]. Diagonal must be 1.0.
  - Preserve the exact class ordering given above.

Output format:
[
  [1.0, <S[0][1]>, ...],
  [<S[1][0]>, 1.0, ...],
  ...
]
"""

    print(prompt)
    return sm_selected


if __name__ == "__main__":
    sm = llm_based(CITYSCAPES_CLASSES)
    print()