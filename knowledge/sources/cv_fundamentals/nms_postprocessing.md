# Detection Post-Processing: NMS, Anchors, and Box Regression

## Overview

Object detection models produce thousands of raw predictions per image. Post-processing transforms these raw outputs into a clean set of final detections. This document covers Non-Maximum Suppression and its variants, anchor-based vs anchor-free detection paradigms, detection head designs, and box regression methods used across YOLO versions.

---

## 1. Non-Maximum Suppression (NMS)

### Purpose

A single object typically triggers high-confidence predictions from multiple nearby grid cells or anchor boxes. NMS removes redundant detections, keeping only the best one per object.

### Algorithm Step by Step

```
Input: List of detections D, each with (box, score, class)
       IoU threshold T (typically 0.45-0.65)

1. Sort all detections by confidence score (descending)
2. Initialize empty list of final detections F

3. While D is not empty:
   a. Take the detection d with the highest score from D
   b. Move d to F (this is a final detection)
   c. Compute IoU between d and all remaining detections in D
   d. Remove from D all detections with IoU(d, detection) > T
   e. Repeat

4. Return F
```

### Properties

- **Greedy algorithm**: Always selects the highest-confidence detection first
- **Class-wise**: NMS is applied independently per class in most implementations
- **Time complexity**: O(N^2) in the worst case, where N is the number of detections (but typically fast because N is small after confidence thresholding)

### Hyperparameters

- **IoU threshold T**: Higher T keeps more overlapping boxes (lenient); lower T suppresses more aggressively. Typical range: 0.45-0.65.
- **Confidence threshold**: Applied before NMS to discard low-confidence detections (typically 0.25 for inference, 0.001 for mAP evaluation).
- **Max detections**: Cap on the number of final detections per image (typically 100-300 for COCO evaluation).

---

## 2. IoU Calculation for NMS

### Intersection over Union (IoU)

```
IoU(A, B) = Area(A intersect B) / Area(A union B)
         = Area(A intersect B) / (Area(A) + Area(B) - Area(A intersect B))
```

### Axis-Aligned Box Intersection

For boxes defined as `(x1, y1, x2, y2)`:

```
inter_x1 = max(A.x1, B.x1)
inter_y1 = max(A.y1, B.y1)
inter_x2 = min(A.x2, B.x2)
inter_y2 = min(A.y2, B.y2)

inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
union_area = A.area + B.area - inter_area

IoU = inter_area / union_area
```

### IoU Properties

- Range: [0, 1]
- IoU = 0: No overlap
- IoU = 1: Perfect overlap
- IoU > 0.5: Generally considered a "match" (COCO AP@0.5)
- Scale-invariant: Same IoU regardless of absolute box size

---

## 3. Soft-NMS

### Problem with Standard NMS

Standard NMS uses a hard threshold: any detection overlapping above T is completely removed. This causes problems in crowded scenes where objects genuinely overlap (e.g., pedestrians in a crowd).

### Soft-NMS Algorithm

Instead of removing overlapping detections, Soft-NMS reduces their confidence scores based on overlap:

#### Linear Decay

```
s_i = s_i * (1 - IoU(M, b_i))    if IoU(M, b_i) >= T
s_i = s_i                          otherwise
```

#### Gaussian Decay (Preferred)

```
s_i = s_i * exp(-IoU(M, b_i)^2 / sigma)
```

Where `sigma` controls the decay rate (typically 0.5). Higher overlap leads to stronger score suppression, but never complete removal.

### Benefits

- Detections for nearby objects are suppressed less aggressively
- Consistently improves AP by 1-2% on datasets with crowded scenes
- No additional training required; drop-in replacement for standard NMS

### Drawback

- Slightly slower than standard NMS (marginal difference in practice)
- The sigma hyperparameter needs tuning

---

## 4. DIoU-NMS

### Motivation

Standard NMS uses only IoU (overlap) to decide suppression. Two boxes might have the same IoU but very different spatial relationships. DIoU-NMS considers the distance between box centers.

### Algorithm

Replace the IoU criterion with:

```
Score = IoU - (d^2 / c^2)
```

Where:
- `d` is the Euclidean distance between box centers
- `c` is the diagonal of the smallest enclosing box

Suppression criterion:
```
Remove b_i if IoU(M, b_i) - (d(M, b_i)^2 / c(M, b_i)^2) > T
```

### Benefit

Boxes that are far apart (even if they happen to overlap) are less likely to be suppressed. This helps in scenarios with elongated objects or objects of very different aspect ratios that partially overlap.

---

## 5. Weighted NMS

### Algorithm

Instead of keeping only the highest-confidence box and discarding others, Weighted NMS computes a weighted average of all overlapping boxes:

```
For a cluster of overlapping detections {b_1, ..., b_k} with scores {s_1, ..., s_k}:
    b_final = sum(s_i * b_i) / sum(s_i)
    s_final = max(s_1, ..., s_k)
```

### Benefit

- Produces more accurate box coordinates by leveraging information from multiple detections
- Particularly helpful when a single object triggers many slightly different bounding boxes
- Can improve AP at higher IoU thresholds (AP@0.75, AP@0.95)

---

## 6. Confidence Thresholding

### Pre-NMS Filtering

Before NMS, detections below a confidence threshold are removed:

```
filtered = [d for d in detections if d.score > conf_threshold]
```

### Typical Thresholds

| Scenario | Confidence Threshold | Rationale |
|---|---|---|
| Training/Validation (mAP) | 0.001 | Keep almost everything for complete precision-recall curve |
| Inference (real-time) | 0.25 - 0.5 | Reduce false positives for practical use |
| High-precision application | 0.5 - 0.7 | Only very confident detections |
| Safety-critical (recall) | 0.1 - 0.2 | Prefer false positives over missed detections |

### Impact on Speed

Pre-NMS filtering is critical for inference speed. Going from 0.001 to 0.25 can reduce the number of boxes entering NMS from 10,000+ to a few hundred, dramatically speeding up post-processing.

---

## 7. NMS-Free Approaches

### DETR (Detection Transformer)

DETR uses a set-based prediction approach:
- Fixed number of learned object queries (typically 100-300)
- Hungarian matching between predictions and ground truth during training
- Each query learns to detect at most one object
- **No NMS needed**: The model directly outputs non-redundant detections
- Drawback: Slow convergence (500 epochs), struggles with small objects

### YOLOv10: NMS-Free YOLO

YOLOv10 introduces consistent dual assignments to eliminate NMS:
- **Training**: Uses both one-to-many assignment (for rich supervision) and one-to-one assignment (for NMS-free inference)
- **Inference**: Uses only the one-to-one branch, which assigns exactly one prediction per object
- Removes NMS latency (which can be 2-5ms per image)
- Slight accuracy trade-off compared to NMS-based approaches

### RT-DETR

Real-Time DETR combines:
- Efficient hybrid encoder for multi-scale feature fusion
- Uncertainty-minimal query selection
- No NMS required (inherits DETR's set-based detection)
- Achieves real-time speed competitive with YOLO models

---

## 8. Anchor-Based Detection

### What Are Anchor Boxes?

Anchor boxes (also called prior boxes or default boxes) are predefined bounding boxes of various sizes and aspect ratios placed at each feature map position. The model predicts **offsets** relative to these anchors rather than absolute box coordinates.

### How Anchors Work

At each feature map cell, K anchor boxes are defined:
```
For a feature map of size (H x W) with K anchors per cell:
    Total anchors = H * W * K
```

For each anchor, the model predicts:
- **Offsets** (dx, dy, dw, dh) to adjust the anchor to fit the object
- **Objectness score**: Probability that the anchor contains an object
- **Class probabilities**: Distribution over object classes

### Anchor Specifications

#### YOLOv3 Anchors (9 anchors, 3 per scale)
```
P3 (small):  (10,13),  (16,30),   (33,23)
P4 (medium): (30,61),  (62,45),   (59,119)
P5 (large):  (116,90), (156,198), (373,326)
```

These are typically determined by **K-means clustering** on the training dataset bounding box dimensions.

### Anchor Matching

During training, anchors are matched to ground truth objects:
- **Positive**: IoU with any ground truth > positive threshold (e.g., 0.5)
- **Negative**: IoU with all ground truths < negative threshold (e.g., 0.4)
- **Ignored**: IoU between thresholds (not used for loss computation)

### Limitations of Anchor-Based Detection

- Anchors are dataset-specific (need recomputing for new datasets)
- Many anchor hyperparameters to tune (sizes, aspect ratios, thresholds)
- Most anchors are negative (class imbalance problem)
- Fixed anchor shapes limit detection of unusual aspect ratios

---

## 9. Anchor-Free Detection

### Center-Based (FCOS, CenterNet)

**FCOS (Fully Convolutional One-Stage)**: Predicts objects from feature map points directly.
- At each feature map location, predict: (l, t, r, b) distances to the four sides of the bounding box
- Additionally predict a "centerness" score to down-weight predictions far from the object center
- No anchor boxes, no IoU-based matching

**CenterNet**: Models objects as center points.
- Predict a heatmap of object centers
- At each center, regress width and height
- Peak detection (local maxima) replaces NMS

### Keypoint-Based

Some methods detect objects by predicting corner points or extreme points:
- **CornerNet**: Detects top-left and bottom-right corners, then groups them
- **ExtremeNet**: Detects extreme points (top, bottom, left, right) and center

### Anchor-Free in YOLO

Starting with YOLOv6 and YOLOv8, YOLO moved to anchor-free detection:
- Each feature map cell directly predicts box coordinates (not offsets from anchors)
- Predictions are typically (cx, cy, w, h) where cx, cy are offsets within the cell
- Simpler design, fewer hyperparameters
- YOLOv8 uses **DFL (Distribution Focal Loss)** for box regression, predicting a probability distribution over possible offset values

---

## 10. YOLO Head Types: Coupled vs Decoupled

### Coupled Head (YOLOv3, YOLOv5)

A single convolutional branch predicts all outputs together:
```
Feature Map -> Conv layers -> [box (4), objectness (1), classes (C)] per anchor
```

- Shared features for all prediction tasks
- Simpler architecture
- Potential conflict: box regression and classification may benefit from different features

### Decoupled Head (YOLOX, YOLOv6, YOLOv8)

Separate branches for different prediction tasks:
```
Feature Map -> Split
    -> Branch 1 (Classification): Conv -> Conv -> class predictions (C)
    -> Branch 2 (Box Regression): Conv -> Conv -> box predictions (4) + DFL
    -> (Optional) Branch 3 (Objectness): Conv -> objectness (1)
```

### Benefits of Decoupled Heads

- **Faster convergence**: Each branch can optimize for its specific task
- **Better accuracy**: Classification and localization features can specialize
- **More flexible**: Easy to add new output branches
- Typically 1-2% mAP improvement over coupled heads
- Standard since YOLOX (2021)

---

## 11. Task-Aligned Learning (TAL) in YOLOv8

### The Assignment Problem

How to decide which feature map predictions are responsible for each ground truth object? Traditional approaches use IoU-based matching, but this doesn't account for model prediction quality.

### TAL Formulation

TAL uses a task-aligned metric that considers both classification and localization quality:

```
t = s^alpha * u^beta
```

Where:
- `s` is the classification score (predicted class probability)
- `u` is the IoU between the predicted box and the ground truth box
- `alpha` and `beta` control the relative importance (typically alpha=0.5, beta=6.0)

### Assignment Process

1. For each ground truth, compute the task-aligned metric `t` for all predictions
2. Select the top-K predictions with the highest `t` as positive samples
3. Predictions not selected are negative samples

### Why TAL Works

- Aligns the assignment with what the model actually needs: predictions that are both confident AND well-localized
- Dynamically adapts during training (unlike fixed anchor matching)
- Avoids the need for handcrafted anchor matching rules
- Used in YOLOv8 and later versions

---

## 12. Box Regression Losses

### Traditional Losses

#### L1 Loss (MAE)

```
L_L1 = |x_pred - x_gt| + |y_pred - y_gt| + |w_pred - w_gt| + |h_pred - h_gt|
```
- Scale-dependent: Same pixel error matters more for small objects
- Does not directly optimize IoU

#### Smooth L1 Loss (Huber Loss)

```
SmoothL1(x) = 0.5 * x^2          if |x| < 1
            = |x| - 0.5           otherwise
```
- Less sensitive to outliers than L1
- Used in Faster R-CNN family

### IoU-Based Losses

#### IoU Loss

```
L_IoU = 1 - IoU(pred, gt)
```
- Directly optimizes the detection metric
- Scale-invariant
- Problem: Zero gradient when boxes don't overlap (IoU = 0)

#### GIoU (Generalized IoU) Loss

```
GIoU = IoU - (|C - (A union B)| / |C|)
L_GIoU = 1 - GIoU
```
Where C is the smallest enclosing box. Range: [-1, 1].
- Provides gradient even when boxes don't overlap
- The penalty term measures how much of the enclosing box is wasted
- Used in YOLOv5 early versions

#### DIoU (Distance IoU) Loss

```
DIoU = IoU - (d^2 / c^2)
L_DIoU = 1 - DIoU
```
Where d = center distance, c = diagonal of enclosing box.
- Directly minimizes center point distance
- Faster convergence than GIoU
- Better for cases where boxes have same IoU but different center alignment

#### CIoU (Complete IoU) Loss

```
CIoU = IoU - (d^2 / c^2) - alpha * v

where v = (4/pi^2) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))^2
      alpha = v / (1 - IoU + v)
```
- Considers three geometric factors: overlap, center distance, aspect ratio
- Most comprehensive IoU-based loss
- Used in YOLOv5, and as a component in later versions

#### SIoU (SCYLLA IoU) Loss

```
SIoU considers four components:
1. Angle cost: Penalizes misalignment between the line connecting centers and axes
2. Distance cost: Center point distance (direction-aware)
3. Shape cost: Aspect ratio similarity
4. IoU cost: Standard IoU
```
- Redefines the distance cost to be direction-aware
- Helps the model learn the correct movement direction during training

### Which Loss to Use

| Loss | Pros | Cons | Used In |
|---|---|---|---|
| GIoU | Gradient when no overlap | Slow convergence | Early YOLOv5 |
| DIoU | Fast center convergence | Ignores aspect ratio | Some custom models |
| CIoU | Most comprehensive | Complex computation | YOLOv5, v7 |
| DFL + CIoU | Distribution-aware regression | More parameters | YOLOv8, v9, v10, v11 |
| Inner-IoU | Focuses on inner region | Specialized | Custom variants |

---

## NMS in Practice: YOLO Inference Pipeline

### Complete Post-Processing Flow

```
1. Model outputs raw predictions for all 8400 cells (at 640x640)
   Each prediction: [x, y, w, h, class_1, class_2, ..., class_C]

2. Confidence thresholding:
   - Compute max class score for each prediction
   - Remove predictions below conf_threshold (e.g., 0.25)
   - Typically reduces 8400 predictions to ~100-500

3. Box format conversion:
   - Convert (cx, cy, w, h) to (x1, y1, x2, y2)

4. Class-wise NMS:
   - For each class independently:
     - Sort by class confidence
     - Apply greedy NMS with IoU threshold (e.g., 0.45)

5. Max detections limit:
   - Keep top-K detections (e.g., 300 for COCO)

6. Output: Final detections [(x1, y1, x2, y2, confidence, class_id), ...]
```

### Batched NMS Optimization

For GPU-efficient NMS across multiple classes simultaneously:
```
Offset each class's boxes by (class_id * max_coordinate)
Run single NMS on all offset boxes
This ensures boxes from different classes never suppress each other
```

This is the standard implementation in torchvision and Ultralytics.

---

## Summary

Post-processing is often overlooked but significantly impacts detection quality and speed. Standard NMS remains the default in most YOLO versions, while NMS-free approaches (YOLOv10, DETR family) represent the future direction. The shift from anchor-based to anchor-free detection has simplified YOLO architectures, and innovations like TAL and DFL have improved assignment and regression quality. Understanding these components is critical for tuning detection performance in production systems.
