# Transfer Learning for Object Detection

## What Is Transfer Learning and Why It Works

Transfer learning is the practice of taking a model trained on one task (source domain) and adapting it to a different but related task (target domain). Instead of training from randomly initialized weights, the model starts with weights that already encode useful visual features.

### Why It Works
Neural networks learn hierarchical features. Early layers learn universal low-level features (edges, textures, color gradients) that are useful across nearly all visual tasks. Middle layers learn mid-level patterns (shapes, parts, textures) that transfer well between similar domains. Later layers learn task-specific features (object classes, spatial relationships) that may need retraining.

### Key Benefits
- **Faster convergence**: Pretrained models reach good accuracy in far fewer epochs.
- **Better performance with limited data**: The pretrained features compensate for small dataset size.
- **Reduced compute cost**: Fewer training epochs mean less GPU time and energy.
- **Improved generalization**: Features learned from millions of diverse images transfer robustness.

## COCO Pretrained Weights

Most YOLO models are pretrained on the Microsoft COCO dataset, which contains 330,000 images across 80 object categories.

### What the Model Already Knows
- **Low-level features**: Edge detection, texture recognition, color patterns, gradient orientations.
- **Mid-level features**: Object parts, shapes (circles, rectangles), common textures (fur, metal, fabric).
- **High-level features**: 80 COCO object categories including person, car, dog, chair, etc.
- **Spatial understanding**: Multi-scale feature extraction, aspect ratio patterns, common object sizes.

### When COCO Pretraining Helps Most
- Target domain contains objects visually similar to COCO categories.
- Target dataset is small (fewer than a few thousand images).
- Target objects have recognizable shapes and textures.

### When COCO Pretraining Helps Less
- Target domain is radically different from natural images (e.g., X-ray, radar, spectrogram).
- Target objects are abstract or do not share visual features with common objects.
- Dataset is extremely large (100,000+ images) and domain-specific.

## Fine-Tuning Strategies

### Strategy 1: Full Model Fine-Tuning
Train all layers of the pretrained model on the new dataset.

- **When to use**: Medium-to-large datasets (1,000+ images), target domain differs from COCO.
- **Learning rate**: Use 1/10th of the from-scratch LR (e.g., 0.001 instead of 0.01 for SGD).
- **Risk**: Can overfit on small datasets since all parameters are free to change.

### Strategy 2: Freeze Backbone, Train Head
Freeze the backbone (feature extractor) and only train the detection head (neck + prediction layers).

- **When to use**: Small datasets (<500 images), target domain similar to COCO.
- **Benefit**: Drastically reduces the number of trainable parameters. Prevents overfitting.
- **Limitation**: The backbone cannot adapt its feature extraction to the new domain.

### Strategy 3: Freeze Specific Layers
Freeze the first N layers of the backbone, train the rest.

- **When to use**: Moderate datasets where some backbone adaptation is desired.
- **Strategy**: Freeze early layers (universal features) and fine-tune later layers (domain-specific features).
- **Typical approach**: Freeze the first 10-15 layers of a YOLO backbone.

### Strategy 4: Progressive Unfreezing
Start with the backbone frozen, train for a few epochs, then unfreeze layers progressively from top to bottom.

1. Freeze backbone, train head for 5-10 epochs.
2. Unfreeze top backbone layers, train with lower LR for 5-10 more epochs.
3. Unfreeze all layers, train with very low LR for final epochs.

This approach gets the best of both worlds: stable initial training and full adaptation.

## Layer Freezing in Ultralytics YOLO

The `freeze` parameter in Ultralytics YOLO controls which layers are frozen during training.

### Usage
```
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Freeze the first 10 layers (backbone)
model.train(data="dataset.yaml", epochs=100, freeze=10)

# Freeze all backbone layers (varies by model)
model.train(data="dataset.yaml", epochs=100, freeze=22)
```

### How It Works
- `freeze=N` freezes the first N layers of the model.
- Frozen layers have `requires_grad=False` and their weights do not update.
- Reduces memory usage because gradients are not computed for frozen layers.
- Layer numbering starts at 0 from the input end of the network.

### YOLO Model Layer Counts (approximate)
- Backbone typically comprises layers 0-9 (for YOLOv8/YOLO11 nano/small).
- Neck (FPN/PAN) layers follow the backbone.
- Detection head layers are at the end.

## Learning Rate for Fine-Tuning

The learning rate is the most critical hyperparameter when fine-tuning. Using a standard training LR will catastrophically overwrite the pretrained features.

### Guidelines
- **Fine-tuning LR**: 1/10th to 1/100th of the from-scratch LR.
- **SGD fine-tuning**: lr0=0.001 to 0.005 (vs 0.01 for from-scratch).
- **AdamW fine-tuning**: lr0=0.0001 to 0.0005 (vs 0.001 for from-scratch).
- **Discriminative learning rates**: Use lower LR for early layers, higher LR for later layers. Not natively supported in Ultralytics but can be implemented with custom training loops.

### Warmup for Fine-Tuning
- Still use warmup (3 epochs), but from an even lower starting LR.
- Warmup prevents sudden large updates that could damage pretrained features.

## When to Train From Scratch vs Fine-Tune

### Fine-Tune (Recommended in Most Cases)
- Target dataset has fewer than 50,000 images.
- Target domain shares visual similarity with natural images.
- Compute budget is limited.
- Faster experimentation cycles are needed.

### Train From Scratch
- Target dataset is very large (100,000+ images) and domain-specific.
- Source and target domains are fundamentally different (e.g., medical imaging, synthetic data, non-optical sensors).
- Custom architecture modifications that change the backbone structure.
- Research exploring novel architectures or training procedures.

### Empirical Test
If uncertain, try both approaches for 50 epochs and compare validation mAP. Fine-tuning almost always wins for small-to-medium datasets.

## Domain Adaptation

Domain adaptation addresses the gap between the source domain (where pretrained weights come from) and the target domain (where the model will be deployed).

### Common Domain Gaps
- **Appearance gap**: Different lighting, weather, camera quality.
- **Layout gap**: Different viewpoints, object arrangements.
- **Scale gap**: Different object sizes relative to the image.
- **Style gap**: Synthetic vs real, different image processing pipelines.

### Techniques
- **Gradual fine-tuning**: Start with source domain data, progressively increase target domain ratio.
- **Domain randomization**: Apply heavy augmentation to bridge the gap.
- **Style transfer**: Transform source images to look like target domain.
- **Intermediate domain**: Find or create a dataset that bridges source and target.

## Few-Shot Object Detection

Few-shot detection aims to detect new object categories with very few examples (1-30 images per class).

### Approaches
- **Fine-tune only the last layer**: Freeze everything except the classification head. Train on few-shot examples.
- **Meta-learning**: Train the model to learn how to learn from few examples. Architectures like Meta-RCNN and FSCE.
- **Prototype networks**: Represent each class by a prototype feature vector. Classify new detections by nearest prototype.
- **Prompt-based**: Use text or visual prompts to guide detection without retraining. Open-vocabulary detection models.

### Practical Tips for Few-Shot with YOLO
- Use the largest pretrained model available (more pretrained knowledge).
- Freeze the backbone completely.
- Use aggressive augmentation on the few available samples.
- Train for many epochs (200+) with a very low learning rate.
- Consider oversampling the rare class images.

## Progressive Resizing Strategy

Progressive resizing trains the model at increasing image resolutions over the course of training.

### How It Works
1. **Phase 1**: Train at low resolution (e.g., 320x320) for the first 30% of epochs. Fast iteration, model learns coarse features.
2. **Phase 2**: Increase to medium resolution (e.g., 480x480) for the next 30%. Model refines features.
3. **Phase 3**: Train at target resolution (e.g., 640x640) for the final 40%. Model learns fine details.

### Benefits
- **Faster early training**: Low resolution means faster forward/backward passes and larger effective batch sizes.
- **Regularization effect**: Resolution changes act as an augmentation, reducing overfitting.
- **Better feature learning**: Coarse-to-fine learning follows a natural curriculum.

### Implementation
Not natively supported in Ultralytics as a single training command, but can be achieved by running sequential training sessions with increasing `imgsz` and loading weights from the previous stage.

## Knowledge Distillation Basics

Knowledge distillation transfers knowledge from a large, accurate teacher model to a smaller, faster student model.

### How It Works
1. Train a large teacher model (e.g., YOLO11x) to high accuracy.
2. Train a smaller student model (e.g., YOLO11n) to mimic the teacher's predictions.
3. The student learns from both the ground truth labels and the teacher's soft predictions.

### Loss Function
The total loss is a weighted combination:
```
loss = alpha * hard_loss(student, ground_truth) + (1 - alpha) * soft_loss(student, teacher)
```

### Benefits for Detection
- Student model can approach teacher accuracy while maintaining fast inference.
- Soft predictions from the teacher contain richer information than hard labels (e.g., class similarity, confidence distribution).
- Particularly useful for edge deployment where model size is constrained.

### Practical Considerations
- The teacher model must be significantly more capable than the student.
- Temperature scaling is applied to soften the teacher's predictions.
- Feature-level distillation (matching intermediate representations) often works better than output-level distillation for detection.
- Not natively supported in Ultralytics CLI, but can be implemented with custom training scripts.

## Summary: Transfer Learning Decision Guide

| Scenario | Strategy | Freeze | LR | Epochs |
|----------|----------|--------|----|--------|
| Small data, similar domain | Freeze backbone | 10+ layers | 0.001 | 100-200 |
| Small data, different domain | Freeze + progressive unfreeze | Start 10, reduce | 0.001-0.005 | 200-300 |
| Medium data, similar domain | Full fine-tune | 0 | 0.005 | 100-150 |
| Medium data, different domain | Full fine-tune | 0 | 0.005-0.01 | 150-200 |
| Large data, any domain | Train from scratch or light fine-tune | 0 | 0.01 | 50-100 |
| Few-shot (<30 per class) | Freeze backbone, heavy augment | All backbone | 0.0005 | 300+ |
