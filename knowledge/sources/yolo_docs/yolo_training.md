# YOLO Training — Complete Best Practices Guide

## Overview

Training a YOLO model effectively requires careful configuration of hyperparameters, augmentation strategies, hardware utilization, and training workflows. This document covers everything from basic setup to advanced distributed training.

---

## Core Hyperparameters

### Learning Rate

| Parameter | Default (YOLOv8) | Description | Tuning Advice |
|-----------|-------------------|-------------|---------------|
| `lr0` | 0.01 | Initial learning rate | Lower (0.001-0.005) for fine-tuning pretrained models. Higher (0.01-0.02) for training from scratch. |
| `lrf` | 0.01 | Final learning rate as a fraction of `lr0` | The final LR = `lr0 * lrf`. Default 0.01 means LR decays to 1% of initial value. |

- **SGD:** Typical `lr0=0.01` with momentum. More stable convergence.
- **Adam/AdamW:** Typical `lr0=0.001`. Faster initial convergence but may generalize worse. Use `weight_decay=0.05` with AdamW.
- **Learning Rate Scheduler:** Cosine annealing is the default. Linearly warms up during `warmup_epochs`, then follows cosine decay to `lr0 * lrf`.

### Momentum and Weight Decay

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `momentum` | 0.937 | SGD momentum / Adam beta1 | 0.9-0.98 for SGD. Higher values smooth updates but may overshoot. |
| `weight_decay` | 0.0005 | L2 regularization coefficient | Increase (0.001-0.01) if overfitting. Decrease if underfitting. Applied to conv layers, not BN. |

### Warmup Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_epochs` | 3.0 | Number of warmup epochs (can be fractional) |
| `warmup_momentum` | 0.8 | Initial momentum during warmup, linearly increases to `momentum` |
| `warmup_bias_lr` | 0.1 | Bias learning rate during warmup |

**Why warmup?** Prevents early training instability. The model starts with random/pretrained weights — large initial gradients can cause divergence. Warmup gradually increases LR from near-zero to `lr0` over `warmup_epochs`.

---

## Batch Size Selection Guide

Batch size directly impacts training stability, GPU memory, and convergence speed.

### GPU Memory Guidelines

| GPU | VRAM | YOLOv8-N (640) | YOLOv8-S (640) | YOLOv8-M (640) | YOLOv8-L (640) | YOLOv8-X (640) |
|-----|------|----------------|----------------|----------------|----------------|----------------|
| RTX 3060 | 12 GB | batch=32 | batch=24 | batch=12 | batch=8 | batch=4 |
| RTX 3080 | 10 GB | batch=24 | batch=16 | batch=10 | batch=6 | batch=4 |
| RTX 3090 | 24 GB | batch=64 | batch=48 | batch=24 | batch=16 | batch=8 |
| RTX 4090 | 24 GB | batch=80 | batch=56 | batch=28 | batch=18 | batch=10 |
| A100 | 40 GB | batch=128 | batch=96 | batch=48 | batch=32 | batch=16 |
| A100 | 80 GB | batch=256 | batch=192 | batch=96 | batch=64 | batch=32 |

*Values are approximate. Actual batch sizes depend on input resolution, augmentation, and other factors.*

### Auto Batch Size
YOLOv8/v11 supports `batch=-1` which automatically determines the largest batch size that fits in GPU memory (with a safety margin of ~60% VRAM utilization).

### Rules of Thumb
- **Minimum batch size:** 8 (below this, BN statistics become unreliable).
- **Optimal range:** 16-64 for most use cases.
- **Larger batches** (64+): More stable gradients, may need higher LR. Use with `accumulate` gradient steps if GPU is limited.
- **Gradient Accumulation:** `accumulate = max(round(64 / batch_size), 1)`. Simulates a larger effective batch size by accumulating gradients over multiple forward passes before updating weights.

---

## Image Size (imgsz) Considerations

| Scenario | Recommended imgsz | Rationale |
|----------|-------------------|-----------|
| General detection | 640 | Default, good balance of speed and accuracy |
| Small object detection | 1280 | Small objects get more pixels, improving detection |
| Very high accuracy needed | 1280-1536 | More spatial detail preserved |
| Edge deployment | 320-416 | Faster inference, lower memory |
| Very large objects only | 416-640 | Sufficient detail, faster training |

**Key considerations:**
- `imgsz` must be a multiple of 32 (due to max stride in the model).
- Doubling `imgsz` roughly quadruples memory usage and training time.
- Images are letterboxed (padded with gray) to maintain aspect ratio.
- For rectangular training (`rect=True`), images are padded to minimize wasted computation.

---

## Augmentation Pipeline

YOLO uses an aggressive online augmentation pipeline. Understanding each parameter is critical for optimal training.

### Mosaic Augmentation (`mosaic=1.0`)
- Combines 4 random training images into a single mosaic image.
- **Benefits:** Increases effective batch diversity, reduces need for large batch sizes, helps model see objects at various scales and in different contexts.
- **Probability:** 1.0 means always applied. Typically disabled in the last 10 epochs (`close_mosaic=10`) to stabilize training.
- **Tip:** Mosaic is one of the most impactful augmentations. Only disable it if your dataset has very specific spatial relationships that mosaic would break.

### MixUp (`mixup=0.0`)
- Blends two images together with a random alpha ratio.
- **When to enable:** Large datasets (10k+ images), when model is overfitting.
- **Typical value:** 0.1-0.3.
- **Note:** Less commonly used than mosaic. Can confuse the model on small datasets.

### Copy-Paste Augmentation (`copy_paste=0.0`)
- Copies object instances from one image and pastes them into another.
- **Requires:** Instance segmentation labels (polygon masks).
- **When to enable:** When you have segmentation labels and need more object diversity. Good for rare classes.
- **Typical value:** 0.1-0.5.

### HSV Augmentation
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hsv_h` | 0.015 | 0.0-1.0 | Hue shift fraction of the full hue wheel (0.015 = ~5.4 degrees) |
| `hsv_s` | 0.7 | 0.0-1.0 | Saturation shift fraction |
| `hsv_v` | 0.4 | 0.0-1.0 | Value (brightness) shift fraction |

**Tip:** Increase HSV augmentation if your deployment conditions have variable lighting. Decrease if color is a critical discriminative feature (e.g., traffic light detection).

### Geometric Augmentations
| Parameter | Default | Description |
|-----------|---------|-------------|
| `degrees` | 0.0 | Random rotation range in degrees (-degrees to +degrees) |
| `translate` | 0.1 | Random translation fraction (0.1 = 10% of image size) |
| `scale` | 0.5 | Random scaling gain (0.5 means scale from 0.5x to 1.5x) |
| `shear` | 0.0 | Random shear angle in degrees |
| `perspective` | 0.0 | Random perspective transformation coefficient (0.0-0.001) |
| `flipud` | 0.0 | Vertical flip probability |
| `fliplr` | 0.5 | Horizontal flip probability |

**Tuning advice:**
- `flipud`: Enable (0.5) for aerial/satellite imagery. Keep 0.0 for natural scenes where objects have consistent up-down orientation.
- `fliplr`: 0.5 is good default. Set to 0.0 if left-right distinction matters (e.g., text detection).
- `degrees`: Use 5-15 for general detection. Higher values only if objects appear at various rotations in your domain.
- `perspective`: Use sparingly (0.0001-0.001). Can cause extreme distortions.

### Erasing Augmentation (`erasing=0.4`)
- Randomly erases rectangular patches of the image during classification training.
- **Purpose:** Forces model to learn from partial information, reduces overfitting.

---

## Multi-Scale Training

Multi-scale training randomly resizes input images during training, typically within a range of `imgsz * 0.5` to `imgsz * 1.5`.

- **Benefits:** Model becomes robust to objects at various scales. Effectively provides scale augmentation without explicit multi-scale labels.
- **Implementation:** In Ultralytics, set `rect=False` and the dataloader will vary image sizes. The `scale` augmentation parameter also provides multi-scale behavior within mosaic.
- **Cost:** ~30-50% slower training due to variable tensor sizes and inability to fully optimize memory allocation.

---

## Transfer Learning Strategies

### Strategy 1: Full Fine-Tuning (Default)
```
yolo detect train model=yolov8n.pt data=custom.yaml epochs=100
```
- Loads COCO-pretrained weights, fine-tunes all layers.
- **Best for:** Datasets with >1000 images that are different from COCO.

### Strategy 2: Freeze Backbone
```
yolo detect train model=yolov8n.pt data=custom.yaml epochs=100 freeze=10
```
- Freezes the first `N` layers (backbone). Only trains neck and head.
- **Best for:** Small datasets (<500 images) or domains similar to COCO.
- **Typical freeze values:** 10 (backbone only) or varies by architecture.
- **Note:** Freezing too many layers limits the model's ability to adapt to your domain. Freezing too few doesn't prevent overfitting.

### Strategy 3: Staged Fine-Tuning
1. Train with frozen backbone for 20-30 epochs (head warmup).
2. Unfreeze all layers and train with a lower LR (lr0=0.001) for remaining epochs.
- **Best for:** Very small datasets or significant domain shift.

### Strategy 4: Train from Scratch
```
yolo detect train model=yolov8n.yaml data=custom.yaml epochs=300
```
- Uses `.yaml` config (no `.pt` weights). Initializes randomly.
- **Best for:** Very large datasets (100k+ images) or domains completely different from COCO (medical, satellite, microscopy).
- **Note:** Requires 3-5x more epochs than fine-tuning.

---

## Resume Training

If training is interrupted, resume from the last checkpoint:
```
yolo detect train resume model=path/to/last.pt
```
- This restores: model weights, optimizer state, epoch number, learning rate schedule, and best fitness.
- The `last.pt` file is saved every epoch in `runs/detect/train/weights/`.
- `best.pt` is saved whenever validation mAP improves.

---

## Distributed Training (DDP)

For multi-GPU training using PyTorch DistributedDataParallel:

```bash
yolo detect train model=yolov8n.pt data=coco.yaml epochs=100 device=0,1,2,3
```

### Key Points:
- **Linear LR Scaling:** LR is automatically scaled by the number of GPUs. Effective LR = `lr0 * num_gpus * batch_size / 64`.
- **Batch Size:** Per-GPU batch size. Total batch = `batch_size * num_gpus`.
- **SyncBN:** Batch normalization is synchronized across GPUs by default.
- **Communication Backend:** NCCL (NVIDIA Collective Communications Library) for GPU-to-GPU communication.
- **Node Training:** For multi-node training, use PyTorch's `torchrun`:
  ```bash
  # Node 0 (master)
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=29500 yolo detect train ...

  # Node 1
  torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=MASTER_IP --master_port=29500 yolo detect train ...
  ```

### DDP Best Practices:
- Keep batch size per GPU >= 8 for stable BN statistics.
- Use `nccl` backend (default on Linux) for best GPU-to-GPU speed.
- Ensure all GPUs are the same model for even workload distribution.
- Monitor per-GPU memory utilization to ensure balanced loading.

---

## Mixed Precision Training (AMP)

Automatic Mixed Precision uses FP16 for most operations and FP32 for operations that need higher precision (loss computation, normalization).

```
yolo detect train model=yolov8n.pt data=coco.yaml amp=True
```

### Benefits:
- **~2x memory reduction:** FP16 tensors use half the memory.
- **~1.5-2x speed improvement:** Modern GPUs (Volta/Turing/Ampere) have dedicated FP16 tensor cores.
- **No accuracy loss:** GradScaler prevents underflow in FP16 gradients.

### When AMP Fails:
- Very deep networks or unusual architectures may see NaN losses with AMP.
- Solution: Disable AMP (`amp=False`) and train in FP32, or use BF16 if hardware supports it (Ampere+ GPUs).
- YOLOv8 automatically falls back to FP32 if AMP causes NaN losses.

---

## Common Training Issues and Solutions

### Issue 1: Training Loss Oscillates Wildly
- **Cause:** Learning rate too high.
- **Solution:** Reduce `lr0` by 2-5x. Increase `warmup_epochs` to 5-10.

### Issue 2: Validation mAP Plateaus Early
- **Cause:** Model converging to local minimum or dataset too small.
- **Solution:** Use cosine LR scheduler (default), increase epochs, add augmentation, try AdamW optimizer.

### Issue 3: Training is Extremely Slow
- **Causes:** Large `imgsz`, small GPU, heavy augmentation, slow disk I/O.
- **Solutions:** Reduce `imgsz`, use `workers=8` (or more), cache images to RAM (`cache=ram`) or disk (`cache=disk`), use SSD storage, enable AMP.

### Issue 4: GPU Utilization is Low
- **Causes:** Dataloader bottleneck, small batch size, CPU preprocessing bottleneck.
- **Solutions:** Increase `workers` (dataloader workers), increase `batch` size, use `cache=ram`, use faster storage.

### Issue 5: Labels Not Found
- **Cause:** Incorrect `data.yaml` paths or directory structure.
- **Solution:** Verify `data.yaml` has correct absolute/relative paths. Ensure `images/` and `labels/` directories mirror each other exactly. Every image file should have a corresponding `.txt` label file.

### Issue 6: Model Predicts Only One Class
- **Cause:** Severe class imbalance or label error.
- **Solution:** Check class distribution in dataset. Use `cls_pw` (class positive weight) or focal loss. Verify label files have correct class indices.

---

## Training Configuration Checklist

1. **Dataset:** Verify images and labels are correctly paired. Run `yolo detect val` on training data first.
2. **Model size:** Start small (n or s) for prototyping, scale up (m, l, x) for production.
3. **Epochs:** Minimum 100 for fine-tuning, 300 for training from scratch. Use early stopping (`patience=50`).
4. **Batch size:** Use `batch=-1` for auto-detection, or set manually based on GPU table above.
5. **Image size:** 640 default. Use 1280 for small objects.
6. **Augmentation:** Keep defaults for initial run. Tune based on validation results.
7. **Optimizer:** SGD (default, more stable) or AdamW (faster convergence).
8. **Pretrained:** Always use pretrained weights unless domain is radically different.
9. **Patience:** Set `patience=50` for early stopping to prevent wasted computation.
10. **Save period:** Set `save_period=10` to save checkpoints every 10 epochs for long training runs.
