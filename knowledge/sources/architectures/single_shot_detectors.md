# Single-Shot Detectors: One-Stage Object Detection Architectures

## Overview

Single-shot detectors perform object detection in a single forward pass through the network, without a separate region proposal stage. This makes them significantly faster than two-stage detectors while achieving competitive accuracy.

## SSD: Single Shot MultiBox Detector (2016)

SSD was one of the first competitive single-shot detectors, introduced by Wei Liu et al.

### Core Architecture
SSD uses a VGG-16 backbone followed by progressively smaller convolutional feature maps. Detection is performed at multiple scales by attaching prediction layers to several feature maps of different resolutions.

### Multi-Scale Feature Maps
- **Early feature maps** (large spatial resolution): Detect small objects.
- **Late feature maps** (small spatial resolution): Detect large objects.
- Each feature map cell predicts offsets and class scores for a set of default (anchor) boxes.

### Default Boxes
At each feature map cell, SSD places default boxes of various aspect ratios (1:1, 2:1, 1:2, 3:1, 1:3). Each default box predicts:
- 4 bounding box offsets (cx, cy, w, h relative to the default box)
- C+1 class scores (C classes + background)

### Training
- **Matching strategy**: Default boxes with IoU > 0.5 with any ground truth are positive.
- **Hard negative mining**: Sorts negative examples by confidence loss and selects the top ones to maintain a 3:1 negative-to-positive ratio.
- **Loss**: Weighted sum of localization loss (smooth L1) and confidence loss (cross-entropy).

### Performance and Legacy
- **74.3 mAP** on VOC2007 at 59 fps (300x300 input) — comparable to Faster R-CNN at 7 fps.
- Demonstrated that multi-scale feature maps are critical for detecting objects at different sizes.
- The multi-scale prediction concept influenced all subsequent detectors, including YOLO and RetinaNet.

### Limitations
- Small object detection was weaker because early feature maps have limited semantic information.
- No feature fusion between scales (each scale operates independently).
- VGG-16 backbone is heavy by modern standards.

## RetinaNet (2017)

RetinaNet by Tsung-Yi Lin et al. achieved a breakthrough by solving the class imbalance problem that plagued all dense single-shot detectors.

### The Class Imbalance Problem
In a dense detector, the vast majority of anchor locations correspond to background (easy negatives). With standard cross-entropy loss, the total loss is dominated by these easy examples, overwhelming the gradients from the rare hard positives. This was the primary reason single-shot detectors lagged behind two-stage detectors in accuracy.

### Key Innovation: Focal Loss
Focal loss is a modified cross-entropy that down-weights the contribution of easy examples:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `p_t`: Model's predicted probability for the correct class.
- `gamma` (focusing parameter): Controls how much easy examples are down-weighted. Default gamma=2.0.
- `alpha_t` (balancing factor): Weights for positive vs negative examples.

When gamma=0, focal loss equals standard cross-entropy. As gamma increases, the loss for well-classified examples (high p_t) diminishes rapidly, focusing training on hard, misclassified examples.

### Architecture: Feature Pyramid Network + Anchors
RetinaNet uses a ResNet backbone with a Feature Pyramid Network (FPN) that creates a top-down pathway with lateral connections, producing semantically rich features at all scales.

- **FPN levels**: P3 through P7 (corresponding to strides 8, 16, 32, 64, 128).
- **Anchors**: 9 anchors per location (3 scales x 3 aspect ratios) at each FPN level.
- **Two subnetworks**: A classification subnet and a box regression subnet, both applied to each FPN level.

### Impact
- **First single-shot detector to surpass two-stage detectors** in accuracy on COCO.
- Focal loss became a standard technique adopted by many subsequent architectures.
- FPN became the de facto standard for multi-scale feature extraction.

## FCOS: Fully Convolutional One-Stage Detection (2019)

FCOS by Zhi Tian et al. eliminated anchor boxes entirely, demonstrating that anchor-free detection could match or exceed anchor-based methods.

### Anchor-Free Design
Instead of predicting offsets relative to predefined anchor boxes, FCOS directly predicts:
- **4 distances**: From each foreground pixel to the four sides of the bounding box (left, top, right, bottom).
- **Class score**: Object class probability at each location.
- **Center-ness score**: A value between 0 and 1 indicating how close the pixel is to the object center.

### Center-ness Branch
The center-ness score suppresses low-quality predictions from pixels far from object centers. Without it, pixels near object boundaries produce inaccurate bounding boxes.

```
center-ness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
```

### Multi-Scale Assignment
Objects of different sizes are assigned to different FPN levels based on the maximum regression distance:
- P3: Objects with max regression distance <= 64
- P4: 64-128
- P5: 128-256
- P6: 256-512
- P7: > 512

### Significance
- Proved that anchor boxes are not necessary for high-accuracy detection.
- Simplified the detection pipeline by removing anchor-related hyperparameters.
- Influenced modern YOLO designs, which have adopted anchor-free approaches (YOLOv8+).

## CenterNet: Keypoint-Based Detection (2019)

CenterNet by Xingyi Zhou et al. formulated detection as keypoint estimation, representing each object by its center point.

### Core Idea
- **Object center**: The model predicts a heatmap of object center points (one per class).
- **Object size**: At each center point, the model regresses the width and height.
- **Offset**: A sub-pixel offset corrects for discretization error from the output stride.

### Architecture
Any fully convolutional backbone (ResNet, DLA, Hourglass) produces a feature map at 1/4 input resolution. Three parallel heads predict:
1. Center heatmap: H/4 x W/4 x C (C classes)
2. Size regression: H/4 x W/4 x 2 (width, height)
3. Offset: H/4 x W/4 x 2 (x_offset, y_offset)

### Key Advantage: No NMS Required
Since the model produces at most one detection per center point, and peak detection on the heatmap naturally selects local maxima, NMS is not needed in principle. In practice, a simple 3x3 max pooling replaces NMS.

### Extensions
CenterNet naturally extends to other tasks by adding more regression heads:
- **3D detection**: Predict depth, 3D dimensions, orientation.
- **Pose estimation**: Predict keypoint offsets from center.
- **Tracking**: Predict offset to previous frame center.

## EfficientDet (2020)

EfficientDet by Mingxing Tan et al. applied the compound scaling philosophy from EfficientNet to object detection.

### BiFPN: Bidirectional Feature Pyramid Network
Standard FPN only has a top-down pathway. BiFPN adds bottom-up connections and uses weighted feature fusion:

- **Bidirectional**: Features flow both top-down and bottom-up.
- **Weighted fusion**: Learned weights determine how much each input feature contributes.
- **Repeated blocks**: BiFPN layers are stacked for progressive feature refinement.

### Compound Scaling
EfficientDet scales three dimensions simultaneously using a compound coefficient phi:
- **Backbone**: EfficientNet-B0 to B7.
- **BiFPN depth and width**: More layers and channels.
- **Image resolution**: 512 to 1536.

This produces a family of models (D0 through D7) spanning a wide efficiency-accuracy spectrum.

### EfficientDet Family

| Model | Input Size | Params | COCO mAP | FPS (V100) |
|-------|-----------|--------|----------|------------|
| D0    | 512       | 3.9M   | 34.6     | 98         |
| D1    | 640       | 6.6M   | 40.5     | 74         |
| D2    | 768       | 8.1M   | 43.0     | 56         |
| D4    | 1024      | 21M    | 49.7     | 23         |
| D7    | 1536      | 52M    | 55.1     | 3          |

### Legacy
EfficientDet demonstrated that systematic scaling across all model dimensions yields better efficiency than ad hoc scaling. This principle influenced YOLO scaling strategies.

## Speed vs Accuracy Comparison

Historical approximate COCO mAP vs inference speed:

| Architecture | mAP (approx) | Speed (fps, GPU) | Year |
|-------------|-------------|-------------------|------|
| SSD300      | 25.1        | 59                | 2016 |
| SSD512      | 28.8        | 22                | 2016 |
| RetinaNet-50| 40.4        | 14                | 2017 |
| FCOS        | 44.7        | 21                | 2019 |
| CenterNet   | 45.1        | 28                | 2019 |
| EfficientDet-D2 | 43.0   | 56                | 2020 |
| YOLOv5m     | 45.4        | 100+              | 2020 |
| YOLO11m     | 51.5        | 150+              | 2024 |

Note: Numbers are approximate and depend heavily on hardware, input resolution, and evaluation conditions. Modern YOLO models dominate both speed and accuracy.

## Historical Progression

### Phase 1: Multi-Scale Dense Prediction (2016)
SSD showed that predicting at multiple feature map scales captures objects of different sizes. This was a foundational insight.

### Phase 2: Solving Class Imbalance (2017)
RetinaNet's focal loss eliminated the accuracy gap between one-stage and two-stage detectors. This was the turning point that made single-shot detectors viable for accuracy-critical applications.

### Phase 3: Anchor-Free Revolution (2019)
FCOS and CenterNet demonstrated that anchor boxes are not necessary. This simplified architectures, reduced hyperparameters, and opened new design possibilities.

### Phase 4: Efficient Scaling (2020)
EfficientDet's compound scaling showed how to systematically trade computation for accuracy, creating families of models for different deployment targets.

### Phase 5: Modern Integration (2020-present)
Modern single-shot detectors (YOLO11, RT-DETR) incorporate the best ideas from all previous generations: multi-scale features (SSD), focal loss or equivalent (RetinaNet), anchor-free prediction (FCOS/CenterNet), efficient scaling (EfficientDet), and advanced augmentation and training strategies.

## Key Concepts That Persist in Modern Detectors

1. **Feature Pyramid Networks**: Multi-scale feature extraction is used in virtually all modern detectors. The specific design has evolved (FPN, PAN, BiFPN, hybrid encoders), but the principle remains.

2. **Anchor-Free Prediction**: Modern YOLO (v8+) and RT-DETR use anchor-free approaches, directly predicting box parameters at each grid cell.

3. **Focal Loss or Variants**: Class imbalance handling remains critical. Modern detectors use task-aligned loss functions, varifocal loss, or quality-aware losses that build on focal loss concepts.

4. **Decoupled Heads**: Separate classification and regression heads (first used explicitly in FCOS) are now standard in modern YOLO architectures.

5. **NMS-Free Detection**: CenterNet's NMS-free approach foreshadowed DETR-style detectors that use set prediction to eliminate NMS entirely.

## Practical Relevance Today

While these architectures are largely superseded by modern YOLO and DETR variants for production use, understanding them is valuable because:

- They explain why modern architectures make specific design choices.
- Concepts like FPN, focal loss, and anchor-free detection remain foundational.
- For academic and custom research, these architectures serve as important baselines.
- Some specialized deployments still use SSD or EfficientDet for specific hardware targets.
