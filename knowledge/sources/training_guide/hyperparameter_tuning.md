# Hyperparameter Tuning for Object Detection

## Learning Rate

The learning rate (LR) is the single most impactful hyperparameter in training. It controls the step size for weight updates during gradient descent.

### Initial Learning Rate Selection
- **SGD**: Typical starting LR is 0.01 for YOLO models.
- **Adam/AdamW**: Typical starting LR is 0.001 (10x lower than SGD).
- Rule of thumb: if training diverges (loss explodes), reduce LR by 10x. If training is too slow to converge, increase LR by 2-3x.

### Learning Rate Warmup
Warmup gradually increases the LR from a small value to the initial LR over the first few epochs. This prevents early instability when the model weights are far from optimal.

- **Warmup epochs** (`warmup_epochs`): Typically 3 epochs in Ultralytics YOLO.
- **Warmup momentum** (`warmup_momentum`): Starts at 0.8 and ramps to target momentum.
- **Warmup bias LR** (`warmup_bias_lr`): Separate warmup for bias parameters, typically 0.1.

### Learning Rate Schedulers

**Step Decay**: Reduces LR by a factor at predefined epoch milestones. Simple but requires knowing when to decay.

**Cosine Annealing**: Smoothly reduces LR following a cosine curve from initial LR to `lrf * lr0`. Default `lrf=0.01` in Ultralytics means the final LR is 1% of the initial LR.

**Linear Decay**: Linearly reduces LR from initial to final value. Less commonly used in modern training.

**One Cycle (1cycle)**: Increases LR from low to high, then decreases below starting value. Can achieve faster convergence. Used by some YOLO implementations.

Cosine annealing is the default and recommended scheduler for most YOLO training runs.

## Batch Size

Batch size determines how many images are processed before updating model weights.

### Relationship with Learning Rate
The linear scaling rule: when batch size increases by factor k, scale LR by factor k. For example, if base LR is 0.01 at batch size 16, use LR 0.02 at batch size 32.

This rule is approximate. For very large batch sizes (>64), use square root scaling or LARS/LAMB optimizers.

### Memory Constraints
- Batch size is often limited by GPU VRAM.
- Each doubling of image size roughly quadruples memory usage.
- Ultralytics auto-batch feature (`batch=-1`) finds the optimal batch size for available memory.

### Gradient Noise and Generalization
- **Smaller batches** (8-16): More gradient noise, can act as regularization, often better generalization.
- **Larger batches** (32-64): Smoother gradients, faster training per epoch, may need additional regularization.
- **Very small batches** (<4): Unstable training, batch norm statistics become unreliable.

Recommended range for YOLO: 8-64, depending on GPU memory and image size. Default is 16.

## Weight Decay

Weight decay (L2 regularization) penalizes large weight values to prevent overfitting.

- **Purpose**: Keeps weights small, improving generalization. Acts as a prior favoring simpler models.
- **Typical value for YOLO**: 0.0005 (`weight_decay=0.0005`).
- **With Adam/AdamW**: Use decoupled weight decay (AdamW) rather than L2 regularization in Adam. AdamW applies weight decay directly to weights, not through the gradient.
- **Scaling**: Higher weight decay for smaller datasets. Lower weight decay (or 0) if the model is underfitting.

## Momentum

### SGD Momentum
- Standard value: 0.937 in Ultralytics YOLO.
- Higher momentum (0.9-0.99): Smoother updates, faster convergence on consistent gradients.
- Lower momentum (0.8-0.9): More responsive to recent gradients, useful for noisy loss landscapes.

### Adam Beta Parameters
- **beta1** (momentum analog): Default 0.9. Controls the exponential decay of the first moment (mean of gradients).
- **beta2** (RMSProp analog): Default 0.999. Controls the exponential decay of the second moment (variance of gradients).
- For object detection, beta1=0.9 and beta2=0.999 are nearly universal defaults.

## Optimizer Selection

### SGD with Momentum
- **Pros**: Well-understood, robust, often achieves highest final accuracy, better generalization.
- **Cons**: Requires careful LR scheduling, slower convergence.
- **When to use**: Large datasets, final production training, when maximum accuracy matters.

### Adam
- **Pros**: Adaptive per-parameter learning rates, fast convergence, less sensitive to LR choice.
- **Cons**: Can generalize worse than SGD, higher memory footprint (stores first and second moments).
- **When to use**: Quick prototyping, small datasets, unstable training.

### AdamW
- **Pros**: Proper decoupled weight decay (fixes Adam's weight decay issue), state-of-the-art default.
- **Cons**: Same memory overhead as Adam.
- **When to use**: Modern default choice. Recommended for most training runs in recent YOLO versions.

Ultralytics default optimizer: SGD. AdamW is increasingly preferred for transformer-based models (RT-DETR).

## Image Size

The input image resolution directly affects detection quality, speed, and memory usage.

### Trade-offs
| Image Size | Accuracy     | Speed       | Memory  |
|------------|-------------|-------------|---------|
| 320        | Lower       | Fastest     | Lowest  |
| 640        | Good        | Balanced    | Medium  |
| 1280       | Highest     | Slowest     | Highest |

### Guidelines
- **640x640** is the default and works well for most tasks.
- Use **1280** when detecting small objects (e.g., distant vehicles, defects in high-res images).
- Use **320-416** for edge deployment where speed is critical.
- Image size must be a multiple of the model's stride (typically 32).
- During training, multi-scale training randomly varies image size by +/- 50% around the base size.

## Confidence and IoU Thresholds for Inference

### Confidence Threshold (`conf`)
- Filters out predictions below this confidence score.
- **Default**: 0.25 for inference, 0.001 for validation (to compute full mAP curve).
- **High conf (0.5+)**: Fewer false positives, may miss some objects.
- **Low conf (0.1-0.2)**: More detections, more false positives. Useful when recall is critical.

### IoU Threshold for NMS (`iou`)
- Controls how much overlap is allowed between predictions before suppressing duplicates.
- **Default**: 0.7 in Ultralytics YOLO.
- **High iou (0.7-0.9)**: Allows more overlapping boxes. Better for crowded scenes.
- **Low iou (0.3-0.5)**: Aggressive suppression. Better when objects rarely overlap.

## NMS Parameters

Non-Maximum Suppression removes duplicate detections after inference.

- **iou_threshold**: IoU above which a lower-confidence box is suppressed (default 0.7).
- **conf_threshold**: Minimum confidence to consider a detection (default 0.25).
- **max_det**: Maximum number of detections per image (default 300). Increase for dense scenes.
- **agnostic_nms**: When True, NMS is applied across all classes (treats all boxes as same class). Useful when different classes overlap in the same spatial region.
- **classes**: Filter to only keep specific class indices.

## Epochs: How Many Are Enough

### General Guidelines
- **Small dataset (<1,000 images)**: 200-500 epochs.
- **Medium dataset (1,000-10,000)**: 100-300 epochs.
- **Large dataset (10,000+)**: 50-150 epochs.
- Default in Ultralytics: 100 epochs.

### Early Stopping
Monitors validation metrics and stops training when no improvement is observed for a patience period.

- **patience** parameter: Number of epochs without improvement before stopping. Default: 100 in Ultralytics.
- Set patience to 20-50 for faster experimentation, 50-100 for production training.
- Monitors mAP@0.5 by default.
- Prevents wasting compute on diminishing returns.

### Signs of Needing More Epochs
- Validation mAP is still increasing at the end of training.
- Training loss is still decreasing.
- Gap between training and validation metrics is small (not overfitting yet).

### Signs of Training Too Long
- Validation mAP plateaus or decreases while training loss continues falling.
- Gap between training and validation metrics is widening.

## Ultralytics YOLO Hyperparameter Evolution

Ultralytics provides a built-in genetic algorithm for hyperparameter search.

### How It Works
1. Start with a base set of hyperparameters.
2. Mutate hyperparameters using a Gaussian distribution.
3. Train the model with mutated parameters.
4. Evaluate fitness (weighted combination of mAP metrics).
5. Select the best-performing hyperparameters as parents for the next generation.
6. Repeat for N iterations.

### Usage
```
model = YOLO("yolo11n.pt")
model.tune(data="dataset.yaml", epochs=30, iterations=300, optimizer="AdamW")
```

### Key Considerations
- Each iteration trains a full model, so this is extremely compute-intensive.
- Use fewer epochs per iteration (e.g., 30) to speed up the search.
- Use a smaller model variant (nano or small) for the search, then apply found hyperparameters to larger variants.
- Results are saved to `evolve.csv` for analysis.

## Ray Tune Integration

For more sophisticated hyperparameter search, Ultralytics integrates with Ray Tune.

### Benefits over Built-in Evolution
- Supports advanced search algorithms: Bayesian optimization, HyperBand, ASHA.
- Distributed search across multiple GPUs or machines.
- Better early stopping of bad trials.
- Dashboard for monitoring search progress.

### Usage
```
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
result = model.tune(
    data="dataset.yaml",
    epochs=50,
    iterations=100,
    use_ray=True
)
```

### Search Space Definition
Ray Tune allows defining custom search spaces with different sampling strategies:
- Uniform sampling for continuous parameters (LR, weight decay)
- Log-uniform sampling for parameters spanning orders of magnitude
- Choice sampling for categorical parameters (optimizer type)

## Summary: Priority Order for Tuning

When time is limited, tune hyperparameters in this order of impact:

1. **Learning rate** and **scheduler** — largest impact on convergence
2. **Image size** — determines what the model can see
3. **Batch size** — constrained by hardware, affects LR scaling
4. **Augmentation strategy** — especially mosaic and close_mosaic
5. **Weight decay** — regularization strength
6. **Optimizer** — SGD vs AdamW
7. **Confidence and NMS thresholds** — tuned at inference time, no retraining needed
