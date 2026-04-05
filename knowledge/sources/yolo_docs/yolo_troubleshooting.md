# YOLO Troubleshooting — Common Problems and Solutions

## Overview

This guide covers the most common issues encountered when training YOLO models, their root causes, diagnostic techniques, and proven solutions.

---

## 1. Overfitting

### Symptoms
- Training loss keeps decreasing but validation loss increases (divergence after a certain epoch).
- High mAP on training set but significantly lower mAP on validation set (gap > 10%).
- Model memorizes training data — performs poorly on new/unseen images.
- Validation metrics plateau or degrade while training metrics improve.

### Causes
- Dataset too small relative to model complexity.
- Insufficient augmentation.
- Model too large (e.g., YOLOv8-X on 500 images).
- Training for too many epochs without early stopping.
- Train/val split not representative (data leakage or biased split).

### Solutions
1. **Increase augmentation intensity:**
   - Enable `mosaic=1.0`, `mixup=0.15`, increase `hsv_h/s/v` values.
   - Add `degrees=10`, `translate=0.2`, `scale=0.9`, `shear=2.0`.
2. **Use a smaller model:** Switch from YOLOv8-L to YOLOv8-S or YOLOv8-N.
3. **Add regularization:** Increase `weight_decay` to 0.001-0.005.
4. **Enable early stopping:** Set `patience=50` (stop training if no improvement for 50 epochs).
5. **Add more data:** Collect more training images, use synthetic data generation, or apply offline augmentation.
6. **Freeze backbone layers:** Set `freeze=10` to prevent backbone overfitting.
7. **Use dropout:** Some custom architectures support dropout, though standard YOLO uses weight decay instead.
8. **Verify train/val split:** Ensure no duplicate or near-duplicate images across splits. Use stratified splitting.

---

## 2. Underfitting

### Symptoms
- Both training and validation loss remain high.
- Low mAP on both training and validation sets.
- Model produces very few or no predictions.
- Predictions are random-looking — poor localization and classification.

### Causes
- Model too small for the task complexity.
- Learning rate too low or too high (no convergence).
- Insufficient training epochs.
- Augmentation too aggressive (model can't learn from heavily distorted images).
- Label errors (wrong classes, poor bounding boxes).
- Image resolution too low for the objects.

### Solutions
1. **Increase model size:** Move from YOLOv8-N to YOLOv8-S or YOLOv8-M.
2. **Increase epochs:** Try 200-300 epochs instead of 100.
3. **Adjust learning rate:** Increase `lr0` by 2-5x. Try `lr0=0.02` for SGD.
4. **Reduce augmentation:** Set `mosaic=0.5`, reduce geometric augmentations. Let the model learn from cleaner data first.
5. **Use pretrained weights:** Start from COCO-pretrained weights instead of training from scratch.
6. **Increase image size:** Move from `imgsz=640` to `imgsz=1280` if objects are small.
7. **Check labels thoroughly:** Use `yolo detect val` on training data to visualize predictions vs labels.
8. **Verify data.yaml:** Make sure class names, number of classes, and paths are correct.

---

## 3. Loss Not Decreasing

### Diagnostic Steps
1. Check if loss is completely flat (initialization problem) or oscillating (LR problem).
2. Verify labels load correctly (check console for warnings about missing labels).
3. Run a sanity check: overfit on a small subset (10-50 images) to confirm the model can learn.

### Causes and Fixes

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| Learning rate too high | Loss oscillates wildly | Reduce `lr0` by 5-10x |
| Learning rate too low | Loss decreases extremely slowly | Increase `lr0` by 2-5x |
| Label format error | Loss stays at initial value | Verify YOLO format: `class x_center y_center width height` (all normalized 0-1) |
| Wrong number of classes | Loss is abnormally high | Ensure `nc` in data.yaml matches actual number of classes in labels |
| Corrupted images | Random errors during training | Run validation on dataset to find corrupt files |
| Cache stale | Old cached labels used | Delete `.cache` files in dataset directory and retrain |
| Extremely imbalanced data | Loss dominated by majority class | Use focal loss, class weights, or oversample minority classes |

---

## 4. Low mAP Despite Low Loss

### Explanation
Loss and mAP measure different things. Loss is computed during training with augmentation. mAP is computed on clean validation images with different evaluation criteria.

### Common Causes
1. **Confidence threshold mismatch:** Model predictions are correct but confidence is too low. Adjust `conf` threshold during evaluation.
2. **NMS threshold too aggressive:** IoU threshold for NMS is too low, removing valid predictions. Try `iou=0.7` instead of default `0.45`.
3. **Augmentation mismatch:** Training with heavy augmentation but evaluating on very different-looking validation images.
4. **Class imbalance:** Model learns dominant classes well (low loss) but fails on rare classes (low mAP because mAP averages across classes).
5. **Localization vs classification:** Box positions are correct but classes are wrong, or vice versa. Check per-component losses.
6. **Small validation set:** Too few images in validation set lead to unstable mAP scores.

### Solutions
- Lower `conf` threshold during validation to see if recall improves dramatically.
- Check per-class AP to identify weak classes.
- Evaluate with different IoU thresholds to see if it's a localization or classification issue.
- Increase validation set size if possible.

---

## 5. NaN Loss

### Immediate Actions
1. Stop training immediately.
2. Check the last saved checkpoint — do not resume from a NaN-corrupted state.

### Causes and Fixes

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| Learning rate too high | NaN appears in first few epochs | Reduce `lr0` to 0.001 or lower |
| AMP (Mixed Precision) instability | NaN appears sporadically | Set `amp=False` to disable mixed precision |
| Corrupted/extreme labels | NaN appears when specific batches are processed | Validate all labels: values must be in [0, 1] for YOLO format. Check for NaN/Inf in label files |
| Division by zero in loss | Consistent NaN at specific point | Check for empty label files (images with no objects). Add at least one negative example with an empty label file |
| Extremely small objects | NaN in box loss | Filter out objects smaller than 2x2 pixels. These cause numerical instability |
| Gradient explosion | Loss spikes then goes to NaN | Reduce `lr0`, increase `warmup_epochs`, use gradient clipping (built-in in Ultralytics) |
| Bad pretrained weights | NaN from epoch 0 | Download fresh pretrained weights. Verify file integrity |

### Prevention
- Always use `amp=True` with AMP auto-fallback (Ultralytics handles this automatically).
- Set `warmup_epochs=3-5` to prevent early gradient explosion.
- Validate dataset before training: `yolo detect val data=data.yaml model=yolov8n.pt` to check for issues.

---

## 6. Class Imbalance Handling

### Problem
Some classes have significantly more instances than others. The model learns to predict common classes well but ignores rare classes.

### Detection
- Check class distribution: count instances per class in your labels.
- Imbalance ratio > 10:1 between most and least common class requires intervention.

### Solutions

#### A. Weighted Loss (cls_pw)
- **What:** Assign higher loss weight to underrepresented classes.
- **How:** In YOLO, `cls_pw=1.0` is the class positive weight. Increase for rare classes.
- **Limitation:** Ultralytics YOLO uses a single `cls_pw` for all classes. For per-class weights, custom code modifications are needed.

#### B. Oversampling
- Duplicate images containing rare classes in your training set.
- Or use augmentation tools (Roboflow, Albumentations) to create variations of rare-class images.

#### C. Focal Loss
- Default in Ultralytics: `fl_gamma=0.0` (disabled). Set `fl_gamma=1.5-2.0` to enable.
- Focal loss down-weights easy/well-classified examples and focuses on hard examples.
- Formula: `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)`
- **When to use:** Moderate imbalance (5:1 to 50:1 ratio).

#### D. Undersampling
- Remove some instances of the majority class.
- **Risk:** May lose valuable training data. Use only when the majority class is extremely dominant (100:1+).

#### E. Synthetic Data
- Generate synthetic images for rare classes using cut-paste augmentation or generative AI.
- Copy-paste augmentation (`copy_paste=0.5`) in YOLO can help if you have segmentation masks.

---

## 7. Small Object Detection Improvements

### Why Small Objects Are Hard
- After multiple downsampling operations, small objects (< 32x32 px) may occupy only 1-2 pixels in the feature map.
- Standard anchors/grid cells are optimized for medium-to-large objects.
- Mosaic augmentation can shrink already-small objects further.

### Solutions

#### A. Higher Input Resolution
- Increase `imgsz` from 640 to 1280 or 1536.
- **Impact:** Small objects get 4x more pixels. Training time increases ~4x.
- **Memory trick:** Use a smaller model (YOLOv8-S) with higher resolution instead of a larger model (YOLOv8-L) with lower resolution.

#### B. Image Tiling (SAHI - Slicing Aided Hyper Inference)
- Split large images into overlapping tiles, run detection on each tile, then merge results.
- **Library:** `sahi` package works directly with Ultralytics YOLO.
- **Parameters:** Tile size (e.g., 640x640), overlap ratio (e.g., 0.2), confidence merge threshold.
- **Best for:** Very high-resolution images (4K+) with small objects.

#### C. Feature Pyramid Network (FPN) Tuning
- Add an extra detection head for smaller objects (P2 layer, stride 4).
- Requires modifying the model YAML configuration.
- **Caution:** Adds significant computation. Use only when small objects are critical.

#### D. Anchor Optimization
- For anchor-based models (YOLOv5/v7), re-run anchor clustering on your dataset.
- Anchor-free models (YOLOv8+) don't need this.

#### E. Reduce Mosaic Impact
- Set `mosaic=0.5` or `close_mosaic=20` to reduce the epochs where mosaic shrinks objects.
- Consider disabling `scale` augmentation or reducing its range.

#### F. Dataset Strategies
- Crop regions with small objects and add them as additional training samples.
- Ensure sufficient small-object examples in training data.

---

## 8. Slow Training Optimization

### Profiling Bottlenecks
1. **GPU utilization < 80%:** Dataloader bottleneck. Increase `workers`.
2. **GPU utilization ~100% but slow:** Model/image size too large for GPU. Reduce batch/imgsz or use a faster GPU.
3. **Disk I/O spikes:** Storage bottleneck. Use SSD or cache to RAM.

### Optimization Strategies

| Strategy | Speedup | How |
|----------|---------|-----|
| Cache to RAM | 2-5x | `cache=ram` (requires dataset fits in RAM) |
| Cache to disk | 1.5-2x | `cache=disk` (creates .npy files) |
| Enable AMP | 1.5-2x | `amp=True` (default in recent versions) |
| Increase workers | 1.2-3x | `workers=8` or `workers=16` (match CPU cores) |
| Reduce imgsz | Linear | Halving imgsz ~4x faster training |
| Use smaller model | Varies | YOLOv8-N is ~10x faster to train than YOLOv8-X |
| Multi-GPU (DDP) | Near-linear | `device=0,1,2,3` for 4-GPU training |
| Rectangular training | 1.1-1.3x | `rect=True` reduces padding waste |
| Reduce augmentation | 1.1-1.5x | Disable heavy augmentations during prototyping |

---

## 9. Memory Issues (OOM) and Solutions

### Symptoms
- `CUDA out of memory` error.
- Training crashes after a few epochs (memory leak) or immediately.
- System becomes unresponsive (CPU OOM).

### GPU Memory Solutions (in order of preference)

1. **Reduce batch size:** Most direct solution. Try halving the batch size.
2. **Use auto-batch:** `batch=-1` lets Ultralytics find the optimal batch size.
3. **Reduce image size:** Drop from `imgsz=1280` to `imgsz=640`.
4. **Use a smaller model:** Switch from YOLOv8-L to YOLOv8-S.
5. **Enable AMP:** `amp=True` uses FP16, roughly halving memory usage.
6. **Disable cache:** `cache=False` if `cache=ram` was causing OOM.
7. **Reduce workers:** High `workers` count increases CPU memory usage and can cause OOM on systems with limited RAM.
8. **Gradient checkpointing:** Not built-in for YOLO, but can be manually added to trade compute for memory.
9. **Use gradient accumulation:** Smaller batch size with accumulation simulates larger effective batch.

### CPU/System Memory Solutions
- Reduce `workers` count (each worker loads data into RAM).
- Don't use `cache=ram` on datasets larger than available RAM.
- Close other applications.
- Use a swap file for temporary relief (will be slow).

### Memory Leak Diagnosis
- If training starts fine but OOM occurs after many epochs, there may be a memory leak.
- Monitor GPU memory with `nvidia-smi -l 1` during training.
- Common cause: Older PyTorch versions or custom callbacks that accumulate tensors without detaching.

---

## 10. Label Errors and Quality Control

### Common Label Errors
| Error | Impact | Detection Method |
|-------|--------|-----------------|
| Wrong class ID | Model learns incorrect associations | Manual review, train a model and check confusion matrix |
| Bounding box too large | Model learns imprecise localization | Visualization scripts, statistical outlier detection |
| Bounding box too small | Missed object parts, poor recall | Compare box area distribution per class |
| Missing labels | Model treats unlabeled objects as background — learns false negatives | Examine FP patterns — if model detects objects not in labels, they may be missing annotations |
| Duplicate labels | Double-counting in metrics | Script to find overlapping boxes with IoU > 0.95 |
| Values outside [0,1] | Training errors or garbage predictions | Parse label files and check ranges |
| Wrong label format | Silent failure, poor training | Verify format: `class_id x_center y_center width height` |

### Quality Control Workflow
1. **Automated checks:** Script to validate label format, value ranges, class ID ranges.
2. **Visual inspection:** Use `yolo detect val` to overlay predictions/labels. Spot-check 50-100 random images.
3. **Statistical analysis:** Plot distribution of box sizes, aspect ratios, class frequencies. Outliers may indicate errors.
4. **Cross-validation:** Train on a subset, predict on the rest. High-confidence false positives may be missed labels.
5. **Consensus labeling:** Have multiple annotators label the same images. Use consensus to identify ambiguous cases.

---

## 11. Domain Adaptation Strategies

### Problem
Model trained on one domain (e.g., daylight urban scenes) performs poorly on another domain (e.g., nighttime, foggy, industrial).

### Strategies

#### A. Fine-Tuning on Target Domain
- Collect a small dataset (100-500 images) from the target domain.
- Fine-tune the COCO-pretrained model with a low LR (`lr0=0.001`).
- Freeze backbone if target domain data is very small.

#### B. Progressive Domain Adaptation
1. Train on source domain (large dataset).
2. Fine-tune on mixed source + target domain.
3. Final fine-tune on target domain only.

#### C. Style Transfer Augmentation
- Use Neural Style Transfer or CycleGAN to generate target-domain-style images from source data.
- Add generated images to training set.

#### D. Domain-Specific Augmentation
- For night scenes: Reduce brightness in HSV augmentation.
- For foggy conditions: Add gaussian blur and haze augmentation.
- For different camera: Match contrast and color profile of target camera.

#### E. Test-Time Augmentation (TTA)
- At inference, apply augmentations (flips, scales) and merge predictions.
- `yolo predict augment=True` enables TTA in Ultralytics.
- Improves robustness but 2-3x slower inference.

---

## 12. Confidence Threshold Tuning

### Why It Matters
The confidence threshold (`conf`) determines which predictions are kept. It directly trades off precision and recall.

### Finding the Optimal Threshold
1. **F1-Confidence Curve:** Ultralytics generates this automatically. The peak of the curve gives the optimal threshold balancing precision and recall.
2. **Application-Specific Tuning:**
   - **Security/surveillance (miss nothing):** Low threshold (0.15-0.25). Accept more false positives.
   - **Quality control (high precision):** High threshold (0.5-0.7). Only keep very confident detections.
   - **General purpose:** Use F1-optimal threshold (typically 0.25-0.45).

### Per-Class Thresholds
- Different classes may need different confidence thresholds.
- Easy classes (large, clear objects) naturally have higher confidence.
- Hard classes (small, occluded) may need lower thresholds.
- Implement per-class thresholds in post-processing for production deployments.

### IoU Threshold for NMS (`iou`)
- **Lower NMS IoU (0.3-0.45):** More aggressive suppression. Fewer duplicate boxes but may remove valid overlapping detections.
- **Higher NMS IoU (0.5-0.7):** Less suppression. Better for crowded scenes but more duplicate detections.
- Default: `iou=0.7` for training, `iou=0.45` for inference in Ultralytics.

---

## Quick Diagnostic Flowchart

```
Training not working?
├── Loss not decreasing?
│   ├── Check labels (format, paths, nc)
│   ├── Reduce lr0
│   └── Overfit on 10 images as sanity check
├── NaN loss?
│   ├── Disable AMP
│   ├── Check for extreme label values
│   └── Reduce lr0, increase warmup
├── Low mAP?
│   ├── Check per-class AP
│   ├── Reduce conf threshold
│   ├── Increase epochs
│   └── Add more data or augmentation
├── Overfitting?
│   ├── Add augmentation
│   ├── Use smaller model
│   ├── Early stopping
│   └── Add more training data
├── OOM?
│   ├── Reduce batch size
│   ├── Reduce imgsz
│   ├── Use smaller model
│   └── Enable AMP
└── Slow training?
    ├── Cache data to RAM
    ├── Increase workers
    ├── Enable AMP
    └── Use smaller model/imgsz
```
