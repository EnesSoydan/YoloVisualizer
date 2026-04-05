# YOLO Evaluation Metrics — Complete Reference

## Overview

Evaluation metrics are essential for understanding how well an object detection model performs. This document covers every metric used in YOLO evaluation, from fundamental concepts to advanced analysis techniques.

---

## IoU (Intersection over Union)

### Definition
IoU measures the overlap between a predicted bounding box and a ground truth bounding box.

### Formula
```
IoU = Area of Intersection / Area of Union
     = Area(Pred ∩ GT) / Area(Pred ∪ GT)
     = Area(Pred ∩ GT) / (Area(Pred) + Area(GT) - Area(Pred ∩ GT))
```

### Visual Explanation
```
+-------------------+
|    Ground Truth    |
|          +---------+--------+
|          | Inter-  |        |
|          | section |        |
+----------+---------+        |
           |     Prediction   |
           +------------------+

IoU = Intersection Area / (GT Area + Pred Area - Intersection Area)
```

### IoU Thresholds
- **IoU = 0.50:** Standard threshold (PASCAL VOC style). A prediction with IoU >= 0.50 is considered a True Positive.
- **IoU = 0.75:** Strict threshold. Requires much more precise localization.
- **IoU = 0.50:0.95:** COCO evaluation standard. Averages over 10 IoU thresholds (0.50, 0.55, 0.60, ..., 0.95).

### IoU Variants Used in YOLO Training
| Variant | Formula | When Used |
|---------|---------|-----------|
| **IoU** | Standard intersection/union | Basic evaluation |
| **GIoU** | IoU - (Area of enclosing box - Union) / Area of enclosing box | Accounts for non-overlapping boxes |
| **DIoU** | IoU - (distance between centers)^2 / (diagonal of enclosing box)^2 | Faster convergence, considers center distance |
| **CIoU** | DIoU - alpha * consistency_term | Considers center distance, aspect ratio, and overlap; default in YOLOv5/v8 |

---

## Precision and Recall

### Precision
**"Of all detections the model made, how many were correct?"**

```
Precision = True Positives / (True Positives + False Positives)
          = TP / (TP + FP)
          = TP / All Detections
```

- High precision = few false alarms.
- A model that only predicts when it is very confident will have high precision but may miss objects (low recall).

### Recall
**"Of all objects that exist, how many did the model find?"**

```
Recall = True Positives / (True Positives + False Negatives)
       = TP / (TP + FN)
       = TP / All Ground Truths
```

- High recall = few missed objects.
- A model that predicts many boxes will have high recall but many false positives (low precision).

### The Precision-Recall Trade-off
- Increasing the confidence threshold: Precision goes up, Recall goes down (fewer but more confident predictions).
- Decreasing the confidence threshold: Recall goes up, Precision goes down (more predictions, including uncertain ones).
- The Precision-Recall (PR) curve visualizes this trade-off across all confidence thresholds.

### Classification of Detections
A detection is classified based on IoU with ground truth boxes:

| Scenario | Classification | Description |
|----------|---------------|-------------|
| Predicted box matches a GT box with IoU >= threshold and correct class | **True Positive (TP)** | Correct detection |
| Predicted box has IoU < threshold with all GT boxes (or wrong class) | **False Positive (FP)** | False alarm |
| GT box has no matching predicted box with IoU >= threshold | **False Negative (FN)** | Missed object |
| No GT, no prediction in a region | **True Negative (TN)** | Not typically used in object detection |

---

## F1 Score

The harmonic mean of Precision and Recall, providing a single metric that balances both.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Interpretation
- **F1 = 1.0:** Perfect precision and recall.
- **F1 = 0.0:** Either precision or recall is zero.
- F1 is maximized when precision and recall are balanced.
- The **F1-confidence curve** shows F1 at each confidence threshold. The peak indicates the optimal confidence threshold for deployment.

### When to Use F1
- When you need a single number to compare models and both precision and recall matter equally.
- **Caution:** F1 treats precision and recall equally. If one is more important in your application (e.g., medical: recall > precision; autonomous driving: precision > recall for some classes), use the PR curve or adjust confidence threshold accordingly.

---

## AP (Average Precision)

### Definition
AP summarizes the Precision-Recall curve into a single value. It is the area under the PR curve for a single class.

### Calculation Methods

#### 11-Point Interpolation (PASCAL VOC 2007)
1. Compute precision at 11 equally spaced recall values: {0, 0.1, 0.2, ..., 1.0}.
2. At each recall value, use the **maximum precision at that recall or higher**.
3. AP = (1/11) * sum of interpolated precisions.

```
AP = (1/11) * Σ(r ∈ {0, 0.1, ..., 1.0}) max(p(r̃) for r̃ >= r)
```

#### All-Point Interpolation (PASCAL VOC 2010+, COCO)
1. Compute precision at every unique recall value where precision changes.
2. Interpolate: at each recall value, precision is the maximum precision at any recall >= current.
3. AP = area under this interpolated PR curve (computed as sum of rectangular areas).

```
AP = Σ(n) (R_n - R_{n-1}) * P_interp(R_n)
where P_interp(R_n) = max(P(R̃) for R̃ >= R_n)
```

**The all-point method is the standard used in COCO and Ultralytics.**

### Step-by-Step Example
1. Sort all detections by confidence (descending).
2. For each detection, determine if it's TP or FP.
3. Compute cumulative precision and recall.
4. Plot the PR curve.
5. Compute the area under the curve = AP.

---

## mAP@50 (Mean Average Precision at IoU 0.50)

### Definition
The mean of AP values across all classes, using IoU threshold of 0.50.

```
mAP@50 = (1/N_classes) * Σ(c=1 to N_classes) AP_c@50
```

### Interpretation
- **mAP@50 = 0.90:** The model correctly detects 90% of objects (on average across classes) with at least 50% overlap.
- A lenient metric — a box that overlaps 50% with the ground truth is considered correct.
- **Good for:** Getting a general sense of detection capability. Common in PASCAL VOC benchmarks.
- **Typical values for COCO:** YOLOv8-N ~37.3, YOLOv8-L ~52.9 (mAP@50:95), which corresponds to much higher mAP@50 values.

---

## mAP@50:95 (mAP at IoU 0.50 to 0.95)

### Definition
The primary COCO evaluation metric. Averages mAP over 10 IoU thresholds from 0.50 to 0.95 in steps of 0.05.

```
mAP@50:95 = (1/10) * Σ(t ∈ {0.50, 0.55, ..., 0.95}) mAP@t
```

### Interpretation
- **Much stricter than mAP@50.** Rewards precise localization.
- At IoU=0.95, the predicted box must almost perfectly align with the ground truth.
- **Typical values:** Even state-of-the-art models achieve only 50-55% mAP@50:95 on COCO.
- **mAP@50:95 is roughly 60-70% of mAP@50** for most models, because high IoU thresholds are very demanding.

### Why mAP@50:95 Matters
- A model with high mAP@50 but low mAP@50:95 produces boxes that roughly cover objects but aren't precisely localized.
- For applications requiring precise localization (e.g., robotic grasping, medical imaging), mAP@50:95 is more informative.

---

## Confusion Matrix for Object Detection

### Structure
Unlike classification confusion matrices, detection confusion matrices include a "Background" class (FP/FN).

```
                  Predicted
              Cat    Dog    Background(FN)
Actual Cat  [ 80     5       15  ]     -> 80 TP, 5 misclassified, 15 missed
Actual Dog  [  3    90        7  ]     -> 90 TP, 3 misclassified, 7 missed
Background  [ 10     8        -  ]     -> FP: 10 false cats, 8 false dogs
(FP)
```

### How to Read It
- **Diagonal:** Correct detections (TP). Higher is better.
- **Off-diagonal (within classes):** Misclassifications. Cat detected but labeled as Dog.
- **Last column (Background/FN):** Missed objects. Objects that were not detected.
- **Last row (Background/FP):** False positives. Detections where no object exists.

### Common Patterns
- **High FN column:** Model is missing many objects. Lower confidence threshold or train longer.
- **High FP row:** Too many false detections. Raise confidence threshold or add negative examples.
- **Specific class confusion:** Two classes frequently confused. Check if they look similar, need more training data, or labels are inconsistent.

---

## FPS and Latency Metrics

### Definitions
- **FPS (Frames Per Second):** Number of images processed per second. FPS = 1000 / latency_ms.
- **Latency:** Total time to process one image (ms). Includes preprocessing + inference + NMS/postprocessing.

### Components of Latency
| Component | Typical Share | Description |
|-----------|--------------|-------------|
| Preprocessing | 5-15% | Resize, normalize, letterbox padding |
| Inference | 70-85% | Forward pass through the neural network |
| NMS/Postprocessing | 5-20% | Non-Maximum Suppression, coordinate scaling |

### Measuring Correctly
- **Warm up:** Run 10-50 inference passes before timing to ensure GPU is warmed up and CUDA kernels are cached.
- **Batch size:** Report whether FPS is measured at batch=1 (latency-critical) or batch=32 (throughput-critical).
- **Precision:** Specify FP32, FP16, or INT8. FP16 is roughly 2x faster than FP32 on modern GPUs.
- **Hardware:** Always report GPU model. A model that runs at 100 FPS on A100 may run at 20 FPS on RTX 3060.
- **Framework:** PyTorch vs TensorRT vs ONNX Runtime can have 2-5x speed differences.

### Latency Targets by Application
| Application | Max Latency | Min FPS |
|------------|-------------|---------|
| Real-time video (30fps) | 33 ms | 30 |
| Real-time video (60fps) | 16 ms | 60 |
| Autonomous driving | 50-100 ms | 10-20 |
| Industrial inspection | 100-500 ms | 2-10 |
| Batch processing | No limit | Throughput matters |

---

## GFLOPs (Giga Floating-Point Operations)

### Definition
GFLOPs measures the computational complexity of a model — the total number of floating-point operations (in billions) required for one forward pass.

### Interpretation
- **Not directly equal to speed.** A model with higher GFLOPs may be faster if it uses more parallelizable operations.
- **Useful for:** Comparing model sizes within the same architecture family (e.g., YOLOv8-N vs YOLOv8-L).
- **Typical values:**
  - YOLOv8-N: 8.7 GFLOPs
  - YOLOv8-S: 28.6 GFLOPs
  - YOLOv8-M: 78.9 GFLOPs
  - YOLOv8-L: 165.2 GFLOPs
  - YOLOv8-X: 257.8 GFLOPs

### Relationship to Other Metrics
- GFLOPs correlates with inference time but is hardware-independent.
- **Parameters (M)** measure model size (storage/memory), while **GFLOPs** measures computation.
- A model can have few parameters but high GFLOPs (if layers are wide) or vice versa.

---

## Per-Class Analysis Techniques

### Why Per-Class Analysis Matters
Overall mAP can mask poor performance on specific classes. A model with 80% mAP might have 95% AP on "car" but 20% AP on "bicycle".

### Techniques

#### 1. Per-Class AP Breakdown
```
Class      | AP@50 | AP@50:95 | Instances
-----------|-------|----------|----------
car        | 0.95  | 0.72     | 5000
person     | 0.91  | 0.65     | 8000
bicycle    | 0.45  | 0.28     | 200
traffic_light| 0.38 | 0.22   | 150
```

#### 2. Confidence Distribution per Class
- Plot histogram of prediction confidences for each class.
- Classes with low average confidence may need more training data or are inherently harder.

#### 3. Size-Based Analysis (COCO-style)
- **AP_small:** Objects with area < 32x32 pixels.
- **AP_medium:** Objects with area 32x32 to 96x96 pixels.
- **AP_large:** Objects with area > 96x96 pixels.
- Most models have significantly lower AP_small, revealing difficulty with small objects.

#### 4. Error Analysis (TIDE framework)
TIDE breaks errors into types:
- **Classification Error:** Correct localization, wrong class.
- **Localization Error:** Correct class, imprecise box.
- **Both:** Wrong class and imprecise box.
- **Duplicate Detection:** Multiple detections for same object.
- **Background Error:** Detection on background region.
- **Missed Error:** Object not detected at all.

---

## When mAP is Misleading

### 1. Class Imbalance
- mAP is a mean across classes. If you have 10 classes but 90% of instances are "car", a model that only detects cars well will still get decent mAP.
- **Solution:** Report per-class AP. Use weighted mAP or focus on underrepresented classes.

### 2. Small Objects
- Small objects often have very low AP that gets averaged away in the overall mAP.
- **Solution:** Report AP_small separately. Use COCO's size-stratified metrics.

### 3. Easy vs Hard Datasets
- mAP@50 can be 90%+ on easy datasets (large objects, clean backgrounds) even with a weak model.
- **Solution:** Use mAP@50:95 which is more discriminating. Compare against a baseline model.

### 4. High Recall at Low Precision
- A model that predicts thousands of boxes per image will have high recall but terrible precision. mAP accounts for this via the PR curve, but the overall number can still be misleading.
- **Solution:** Always check the PR curve shape. A healthy curve has a large area with high precision maintained across recall values.

### 5. Crowd/Dense Scenes
- In dense scenes, NMS may remove valid detections. mAP penalizes for both missed objects (FN) and duplicate detections (FP).
- **Solution:** Use softer NMS or evaluate with crowd-specific metrics.

### 6. Temporal Consistency (Video)
- mAP evaluates per-frame. A model that flickers (detects an object in frame 1, misses in frame 2, detects in frame 3) may have decent mAP but terrible real-world performance.
- **Solution:** Use tracking metrics (MOTA, MOTP, HOTA) for video applications.

---

## Metric Summary Table

| Metric | Range | Higher = Better? | Measures |
|--------|-------|-------------------|----------|
| IoU | 0-1 | Yes | Box overlap quality |
| Precision | 0-1 | Yes | Detection reliability (few false alarms) |
| Recall | 0-1 | Yes | Detection completeness (few misses) |
| F1 Score | 0-1 | Yes | Balance of precision and recall |
| AP@50 | 0-1 | Yes | Per-class detection quality (lenient) |
| AP@50:95 | 0-1 | Yes | Per-class detection quality (strict) |
| mAP@50 | 0-1 | Yes | Overall detection quality (lenient) |
| mAP@50:95 | 0-1 | Yes | Overall detection quality (strict) |
| FPS | 0-inf | Yes (higher = faster) | Inference speed |
| Latency (ms) | 0-inf | No (lower = faster) | Inference time per image |
| GFLOPs | 0-inf | No (lower = simpler) | Computational complexity |
| Parameters (M) | 0-inf | No (lower = smaller) | Model size |
