# Prompt — Class Penalty Matrix for CPLoss on Cityscapes

You are designing the penalty-tolerance matrix `S` consumed by the
Contextual Penalty Loss (CPLoss) for a semantic segmentation model
trained on the **Cityscapes** driving dataset.

## How the matrix is consumed (do not ignore this)

For every pixel with ground-truth class `i`, the model outputs a softmax
distribution `p` over the 19 classes. CPLoss computes:

    D = 1 - S
    loss_pixel = sum_j  p_j * D[i, j]
              = sum_j  p_j * (1 - S[i, j])

Therefore:
- `S[i, j]` directly controls how much we tolerate the model placing
  probability mass on class `j` when the true class is `i`.
- `S[i, j] = 1.0` → zero penalty (the model is allowed to predict `j`
  instead of `i` for free).
- `S[i, j] = 0.0` → maximum penalty (this confusion is treated as bad
  as predicting the most unrelated class possible).
- Row `i` = TRUE class. Column `j` = PREDICTED class.
- The matrix MUST be asymmetric in general: the cost of confusing
  `person → road` is not the cost of confusing `road → person`.

## Domain

Cityscapes is a fine-annotation semantic segmentation dataset of **5,000
urban driving scenes** captured across **50 European cities** at
2048 × 1024 px. Key characteristics that affect the penalty matrix:

- **Pedestrian- and cyclist-dense**: sidewalks, crossings, and cycling
  lanes are heavily populated. Missing a person or rider is
  disproportionately dangerous compared to other driving datasets.
- **Dense infrastructure**: narrow streets, frequent poles, traffic
  lights, and signs appear close together and can be visually ambiguous.
- **Trams and trains on fixed tracks**: the `train` class includes
  trams. Their trajectory is constrained to rails, which changes the
  safety calculus of confusing `train` with other vehicles.
- **High vegetation coverage**: trees and bushes often border the road
  and can occlude other classes; vegetation ↔ terrain confusion is common.
- **Rare classes with high safety impact**: `rider`, `motorcycle`, and
  `train` are infrequent (< 0.2 % of pixels each) but safety-critical.

The **19 target classes** (in this exact order, indices 0–18) are:

    0  road
    1  sidewalk
    2  building
    3  wall
    4  fence
    5  pole
    6  traffic light
    7  traffic sign
    8  vegetation
    9  terrain
    10 sky
    11 person
    12 rider
    13 car
    14 truck
    15 bus
    16 train
    17 motorcycle
    18 bicycle

## What S[i, j] must encode

A single scalar in [0, 1] that blends THREE factors. Reason about all
three before writing each cell:

1. **Visual / appearance similarity** — could a vision model plausibly
   confuse `i` with `j` based on shape, texture, color, scale, or
   typical pose in a European urban scene?
   (road ↔ sidewalk, car ↔ truck, person ↔ rider, wall ↔ building: high.
    train ↔ bus: moderate. sky ↔ anything else: very low.)

2. **Semantic / functional kinship** — do `i` and `j` belong to the same
   high-level group (drivable surface, static structure, vulnerable road
   user, large vehicle, two-wheeler, rail vehicle, etc.)?

3. **Driving-safety consequence of this specific confusion direction** —
   if the planner trusts the prediction `j` while the world is actually
   `i`, what happens?
   - "Vehicle drives into / over the real class `i`" → `S[i, j]` must be
     near 0 even if appearance similarity is moderate.
     Example: `S[person, road] ≈ 0.02` — catastrophic.
   - "Vehicle brakes / slows unnecessarily" → tolerable.
     Example: `S[road, person] ≈ 0.35` — annoying but safe.
   - "Both classes trigger the same downstream behavior" → tolerable
     even if they look different.
   - "Tram/train on tracks predicted as bus" → less dangerous than
     predicting a free-roaming vehicle, because trams cannot leave
     their tracks.

Safety dominates when factors (1) and (3) disagree.

## Hard constraints

- Output a 19 × 19 matrix.
- `S[i, i] = 1.0` for every `i`.
- All values in `[0.0, 1.0]`, two decimals.
- **Do NOT symmetrize.** Reason about each direction independently.
- Preserve the class order above (row 0 = road, …, row 18 = bicycle).
- No row, no column, and no off-diagonal cell may be identically 0 or 1.
- Use the full calibration scale below; avoid clustering everything in
  the 0.1–0.3 band.

## Calibration scale (use these anchors)

| Value     | Meaning                                                                  |
|-----------|--------------------------------------------------------------------------|
| 1.00      | Same class (diagonal only).                                              |
| 0.70–0.85 | Near-interchangeable for driving: car↔truck, person↔rider, motorcycle↔bicycle, traffic light↔traffic sign, wall↔fence, road↔sidewalk (false-brake direction). |
| 0.45–0.65 | Same high-level group, different role: truck↔bus, vegetation↔terrain, building↔wall, train↔bus (fixed-track mitigation). |
| 0.20–0.40 | Weak relation, safe-side confusion: static ↔ static across groups; predicting a VRU where there is only road. |
| 0.05–0.15 | Cross-group, mildly unsafe: predicting infrastructure where there is a vehicle. |
| 0.00–0.05 | Safety-critical failure: predicting drivable / empty where the truth is person, rider, or any free-moving vehicle in the ego path. |

## Worked anchors (you must respect these)

**Vulnerable road users — Cityscapes pedestrians/cyclists are very frequent:**
- `S[person, road]       ≈ 0.02`  pedestrian erased from scene → catastrophic
- `S[road,   person]     ≈ 0.35`  phantom pedestrian → false brake, tolerable
- `S[person, rider]      ≈ 0.70`  both are humans present in the scene
- `S[rider,  person]     ≈ 0.65`  loses the vehicle context, still a human
- `S[rider,  bicycle]    ≈ 0.65`  loses the rider, keeps the form factor
- `S[rider,  motorcycle] ≈ 0.55`  different two-wheeler dynamics
- `S[person, sidewalk]   ≈ 0.05`  pedestrian erased into background → dangerous

**Vehicles:**
- `S[car,   truck]       ≈ 0.80`  minor planning impact; size underestimated
- `S[truck, car]         ≈ 0.65`  underestimating large vehicle size is worse
- `S[motorcycle, bicycle] ≈ 0.75` similar two-wheeler dynamics
- `S[bicycle, motorcycle] ≈ 0.70`
- `S[bus,   truck]       ≈ 0.65`  both large; bus stops are predictable
- `S[truck, bus]         ≈ 0.65`

**Rail vehicles (Cityscapes trams / trains on fixed tracks):**
- `S[train, bus]         ≈ 0.55`  tram on rails, trajectory constrained → less dangerous than a free vehicle
- `S[bus,   train]       ≈ 0.45`  bus misread as fixed-track vehicle → could miss its lateral movement
- `S[train, car]         ≈ 0.30`  significantly different size and behavior
- `S[car,   train]       ≈ 0.20`  small object misread as large rail vehicle

**Drivable surfaces:**
- `S[road,     sidewalk] ≈ 0.55`  drivable misread as not-drivable → over-cautious
- `S[sidewalk, road]     ≈ 0.20`  pedestrian zone misread as drivable → dangerous
- `S[road,     terrain]  ≈ 0.30`  off-road surface treated as drivable
- `S[terrain,  road]     ≈ 0.22`  dirt / grass treated as drivable road

**Static structures:**
- `S[wall,  fence]       ≈ 0.75`  same physical role
- `S[fence, wall]        ≈ 0.75`
- `S[building, wall]     ≈ 0.60`  same construction group
- `S[wall, building]     ≈ 0.55`

**Objects / signals:**
- `S[traffic light, traffic sign] ≈ 0.75`  both are mounted roadside signals
- `S[traffic sign,  traffic light] ≈ 0.75`
- `S[pole, traffic light] ≈ 0.40`  structural similarity but different function
- `S[pole, traffic sign]  ≈ 0.40`

**Nature / sky:**
- `S[vegetation, terrain] ≈ 0.65`  adjacent natural classes
- `S[terrain, vegetation] ≈ 0.60`
- `S[sky, *non-sky*]      ≤ 0.10`  sky confusions are almost always wrong

Calibrate every other cell against these anchors.

## Output format

Return ONLY a valid Python 2D list named `S`, 19 rows × 19 columns,
two-decimal floats, no comments, no prose, no markdown fences:

    S = [
      [1.00, ..., ...],
      [..., 1.00, ...],
      ...
      [..., ..., 1.00],
    ]