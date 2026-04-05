# YOLO Dataset Management — Complete Guide

## Overview

A well-structured, high-quality dataset is the most critical factor in YOLO model performance. This guide covers everything from label format to advanced dataset management strategies.

---

## YOLO Label Format

### Standard Format
Each image has a corresponding `.txt` label file with the same name. Each line in the label file represents one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

### Field Definitions
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `class_id` | integer | 0 to (nc-1) | Zero-indexed class index |
| `x_center` | float | 0.0 to 1.0 | X coordinate of box center, normalized by image width |
| `y_center` | float | 0.0 to 1.0 | Y coordinate of box center, normalized by image height |
| `width` | float | 0.0 to 1.0 | Box width, normalized by image width |
| `height` | float | 0.0 to 1.0 | Box height, normalized by image height |

### Example
For an image of size 1920x1080 with a bounding box at pixel coordinates:
- Top-left: (500, 200), Bottom-right: (900, 600)
- Box center: ((500+900)/2, (200+600)/2) = (700, 400)
- Box size: (900-500, 600-200) = (400, 400)
- Normalized: x_center=700/1920=0.3646, y_center=400/1080=0.3704, width=400/1920=0.2083, height=400/1080=0.3704

```
0 0.3646 0.3704 0.2083 0.3704
```

### Conversion Formulas
```
# From pixel coordinates (x1, y1, x2, y2) to YOLO format:
x_center = ((x1 + x2) / 2) / image_width
y_center = ((y1 + y2) / 2) / image_height
width = (x2 - x1) / image_width
height = (y2 - y1) / image_height

# From YOLO format to pixel coordinates:
x1 = (x_center - width/2) * image_width
y1 = (y_center - height/2) * image_height
x2 = (x_center + width/2) * image_width
y2 = (y_center + height/2) * image_height
```

### Other YOLO Task Formats

#### Instance Segmentation
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
Each point is a polygon vertex, normalized by image dimensions.

#### Oriented Bounding Box (OBB)
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```
Four corner points of the rotated box, normalized.

#### Pose Estimation
```
<class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_visible> ... <kpN_x> <kpN_y> <kpN_visible>
```
Standard box followed by keypoint coordinates and visibility flags.

#### Classification
No label files needed. Directory structure determines class:
```
dataset/
  train/
    class_a/
      img1.jpg
    class_b/
      img2.jpg
```

---

## Dataset Directory Structure

### Standard YOLO Structure
```
dataset/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── img_101.jpg
│   │   └── ...
│   └── labels/
│       ├── img_101.txt
│       └── ...
└── test/
    ├── images/
    │   ├── img_201.jpg
    │   └── ...
    └── labels/
        ├── img_201.txt
        └── ...
```

### Alternative Structure (Also Supported)
```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── img_001.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
```

### Important Rules
- **Image-label pairing:** Every image must have a corresponding label file with the exact same filename (different extension). `img_001.jpg` -> `img_001.txt`.
- **Empty labels:** Images with no objects should have an empty `.txt` file (zero bytes). This tells YOLO the image is a valid negative example.
- **Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.
- **No spaces in paths:** Avoid spaces in directory and file names. Use underscores instead.
- **Case sensitivity:** On Linux, `Image.JPG` and `image.jpg` are different files. Standardize to lowercase.

---

## data.yaml Configuration

### Minimal Configuration
```yaml
# dataset paths
train: path/to/dataset/train/images
val: path/to/dataset/val/images
test: path/to/dataset/test/images  # optional

# number of classes
nc: 3

# class names
names:
  0: cat
  1: dog
  2: bird
```

### Full Configuration
```yaml
# Dataset root path (absolute or relative to YOLO working directory)
path: /home/user/datasets/my_dataset

# Relative paths to splits (relative to 'path' above)
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 5

# Class names (must match nc count, zero-indexed)
names:
  0: person
  1: car
  2: bicycle
  3: motorcycle
  4: bus

# Optional: download script or URL
download: https://example.com/dataset.zip
```

### Path Resolution
- **Absolute paths:** Used as-is.
- **Relative paths:** If `path` is specified, relative to `path`. Otherwise, relative to the current working directory.
- **Best practice:** Use absolute paths or set `path` to avoid confusion.

### Common data.yaml Mistakes
| Mistake | Symptom | Fix |
|---------|---------|-----|
| `nc` doesn't match actual classes in labels | Incorrect predictions, training instability | Count unique class IDs in all label files |
| Paths point to labels instead of images | "No images found" error | Paths must point to image directories |
| Class names don't match label indices | Wrong class names in predictions | Ensure names dict keys (0,1,2...) match label class IDs |
| Trailing spaces in names | Class name mismatch issues | Trim whitespace from class names |
| Windows backslashes | Path resolution failure on Linux | Use forward slashes: `train: C:/data/train` |

---

## Labeling Tools Comparison

### LabelImg
- **Type:** Desktop application (Python/Qt).
- **Format:** PASCAL VOC (XML), YOLO (TXT), CreateML (JSON).
- **Pros:** Free, offline, simple, supports YOLO format natively.
- **Cons:** No team collaboration, no auto-labeling, slow for large datasets. Project is archived/unmaintained.
- **Best for:** Small personal projects, quick labeling tasks.

### Roboflow
- **Type:** Web platform (SaaS).
- **Format:** Exports to 30+ formats including YOLO, COCO, VOC.
- **Pros:** Auto-labeling with foundation models (SAM, DINO), dataset versioning, augmentation pipeline, team collaboration, dataset health check, active learning.
- **Cons:** Paid for large datasets (free tier: 10k images). Data stored in cloud.
- **Best for:** Teams, production projects, when you need dataset management pipeline.

### CVAT (Computer Vision Annotation Tool)
- **Type:** Web application (self-hosted or cloud).
- **Format:** COCO, PASCAL VOC, YOLO, and many others.
- **Pros:** Free and open-source, self-hostable (data privacy), supports video annotation, interpolation for tracking, semi-automatic annotation, team management.
- **Cons:** Requires setup for self-hosting, steeper learning curve, heavier resource usage.
- **Best for:** Medium-to-large teams needing data privacy, video annotation, complex workflows.

### Label Studio
- **Type:** Web application (self-hosted or cloud).
- **Format:** Highly flexible, exports to COCO, VOC, YOLO via converters.
- **Pros:** Free and open-source (Community Edition), extremely flexible labeling UI (custom templates), supports all data types (image, text, audio, video, time-series), ML backend integration for pre-annotation.
- **Cons:** YOLO export requires additional conversion step, more general-purpose (not specialized for CV).
- **Best for:** Multi-modal projects, custom labeling workflows, integration with ML pipelines.

### Quick Comparison Table

| Feature | LabelImg | Roboflow | CVAT | Label Studio |
|---------|----------|----------|------|--------------|
| Cost | Free | Freemium | Free/Enterprise | Free/Enterprise |
| YOLO export | Native | Native | Native | Via converter |
| Auto-labeling | No | Yes (AI) | Semi-auto | ML backend |
| Team support | No | Yes | Yes | Yes |
| Self-hosted | Desktop | Cloud only | Yes | Yes |
| Video support | No | Limited | Yes | Yes |
| Dataset versioning | No | Yes | Limited | Limited |
| Active learning | No | Yes | No | Yes (custom) |

---

## Annotation Best Practices

### Bounding Box Quality Rules

1. **Tight boxes:** The bounding box should tightly encompass the visible portion of the object. Minimize background pixels inside the box.
2. **Include all visible parts:** If an object is partially occluded, draw the box around the visible portion, not the imagined full extent.
3. **Handle truncation:** If an object extends beyond the image edge, draw the box to the image boundary.
4. **Consistent class definitions:** Create a clear annotation guide defining what each class includes/excludes. Share with all annotators.

### Edge Cases

| Scenario | Recommendation |
|----------|---------------|
| Heavily occluded (>70% hidden) | Skip annotation — too ambiguous for the model |
| Partially occluded (30-70%) | Annotate the visible portion |
| Object groups/clusters | Annotate each individual object if distinguishable |
| Very small objects (< 10px) | Skip unless small objects are your primary target |
| Ambiguous class | Define clear rules in annotation guide. When in doubt, use a "uncertain" tag for review |
| Object at image edge | Annotate the visible portion to the image boundary |
| Reflections/shadows | Do not annotate unless they are detection targets |
| Object on screen/poster | Depends on use case. Usually do not annotate depicted objects |

### Annotation Quality Metrics
- **Inter-annotator agreement:** Have 2+ annotators label the same subset. Compare IoU of their boxes. Target > 0.85 IoU.
- **Annotation speed vs quality:** Typical rates: 200-500 boxes/hour for simple detection, 50-100 boxes/hour for complex scenes.
- **Review cycle:** 10-20% of all annotations should be reviewed by a senior annotator.

---

## Class Balance Strategies

### Diagnosing Imbalance
```python
# Count instances per class
from collections import Counter
import os

label_dir = "dataset/train/labels"
class_counts = Counter()

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file)) as f:
        for line in f:
            class_id = int(line.strip().split()[0])
            class_counts[class_id] += 1

for cls_id, count in sorted(class_counts.items()):
    print(f"Class {cls_id}: {count} instances")
```

### Balancing Approaches

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Oversampling** | Duplicate images with rare classes | Imbalance ratio 5:1 to 20:1 |
| **Undersampling** | Remove some majority class images | Only when majority class is extremely dominant (100:1+) |
| **Class weights** | Increase loss weight for rare classes | Moderate imbalance (3:1 to 10:1) |
| **Focal loss** | Down-weight easy examples automatically | Moderate to severe imbalance |
| **Data collection** | Collect more images of rare classes | Always the best option if feasible |
| **Synthetic augmentation** | Generate rare class instances | When data collection is impractical |
| **Copy-paste** | Paste rare class instances into other images | When segmentation masks are available |

### Acceptable Imbalance
- Ratios up to 5:1 are generally acceptable without intervention.
- Ratios 5:1 to 20:1 benefit from oversampling or focal loss.
- Ratios > 20:1 require aggressive balancing strategies.

---

## Negative Examples (Background Images)

### What Are Negative Examples?
Images that contain **none** of your target classes. They have empty label files (0 bytes).

### Why They Matter
- Without negative examples, the model only sees images with objects. It may learn that "every image has objects," leading to high false positive rates on deployment images that are empty.
- Negative examples teach the model that **background exists** and not every image contains a detection target.

### How Many to Include
- **General rule:** 0-10% of your dataset should be negative examples.
- **Ultralytics recommendation:** Up to 10% of total images.
- **High false-positive domain:** If your deployment has many empty frames (e.g., security camera), increase to 10-20%.
- **Rare events:** If the target object appears infrequently, more negative examples help calibrate confidence.

### How to Add Them
1. Collect images from your deployment environment that contain no target objects.
2. Place them in the `images/` directory alongside other training images.
3. Create corresponding empty `.txt` files in the `labels/` directory.
4. The empty label file is critical — without it, YOLO assumes the image has no corresponding labels and may skip it or log a warning.

### Selecting Good Negative Examples
- Images that look similar to positive examples but lack the target object.
- Backgrounds typical of your deployment environment.
- Challenging negatives: images with objects that look similar to your target but aren't (e.g., for vehicle detection, include images with toy cars).

---

## Train/Val/Test Split Ratios and Strategies

### Standard Ratios

| Dataset Size | Train | Val | Test | Notes |
|-------------|-------|-----|------|-------|
| < 500 images | 80% | 20% | - | Not enough for separate test set |
| 500-5000 images | 70% | 20% | 10% | Standard split |
| 5000-50000 images | 80% | 10% | 10% | Can afford smaller val/test percentage |
| > 50000 images | 85% | 10% | 5% | Large dataset, even 5% test is sufficient |

### Splitting Strategies

#### Random Split
- Randomly assign images to train/val/test.
- **Risk:** Near-duplicate images (e.g., video frames) may end up in both train and val, causing data leakage.
- **When to use:** Images are independent (collected at different times/locations).

#### Stratified Split
- Ensure each split has similar class distributions.
- **How:** Group images by their class composition, then split within each group.
- **When to use:** Class imbalance exists and you want balanced evaluation.

#### Group Split
- Group images by scene/location/video and split by group.
- **Example:** All frames from video_001 go to train, all from video_002 go to val.
- **When to use:** Frames from the same video or location are highly correlated. This prevents data leakage.
- **Critical for:** Dashcam, surveillance, drone, and any video-derived datasets.

#### Temporal Split
- Split by time: oldest data for training, newest for validation/test.
- **When to use:** Data distribution shifts over time (seasonal changes, different conditions).
- **Benefit:** Tests how well the model generalizes to future data.

### Validation Set Requirements
- Must be representative of the deployment domain.
- Should contain all classes with sufficient instances (minimum 50 instances per class recommended).
- Must be fixed — never change the validation set during hyperparameter tuning.
- Never use for training decisions beyond early stopping and model selection.

---

## Dataset Augmentation: Offline vs Online

### Online Augmentation (During Training)
- Applied on-the-fly during each training epoch.
- Each epoch sees different augmented versions of the same image.
- **Pros:** No additional storage, effectively infinite augmentation variety, YOLO handles it automatically.
- **Cons:** Adds CPU overhead during training, limited to supported transformations.
- **YOLO built-in augmentations:** Mosaic, MixUp, Copy-Paste, HSV, geometric (rotate, scale, translate, shear, perspective, flip), erasing.

### Offline Augmentation (Pre-processing)
- Generate augmented images and save to disk before training.
- **Pros:** No training overhead, can use any augmentation library (Albumentations, imgaug), full control over augmentation pipeline.
- **Cons:** Requires additional storage, fixed augmentation per epoch (less variety), may lead to overfitting on specific augmentation patterns.
- **When to use:** When you need augmentations not supported by YOLO (e.g., weather effects, style transfer, GAN-generated variations), or when you want to oversample specific classes.

### Recommended Approach
- Use YOLO's built-in online augmentation as the primary pipeline.
- Apply offline augmentation only for:
  - Class balancing (oversample rare classes with augmented copies).
  - Domain-specific augmentations (rain, fog, snow effects).
  - When compute is limited and you want to pre-compute augmentations.

---

## Dataset Versioning

### Why Version Your Dataset
- Track changes: which images were added, removed, or relabeled.
- Reproducibility: know exactly which data produced which model.
- Rollback: revert to a previous version if new labels cause regression.
- Collaboration: multiple annotators can work without conflicts.

### Versioning Tools

#### DVC (Data Version Control)
- Git-like versioning for large files and datasets.
- Stores data in remote storage (S3, GCS, Azure, SSH), tracks metadata in Git.
- **Commands:** `dvc init`, `dvc add dataset/`, `dvc push`, `dvc checkout`.
- **Best for:** Teams using Git who need dataset versioning alongside code.

#### Roboflow
- Built-in dataset versioning with each preprocessing/augmentation configuration.
- Generates a unique version number for each dataset configuration.
- **Best for:** Teams already using Roboflow for annotation.

#### Manual Versioning
- Use timestamped directories: `dataset_v1_20240115/`, `dataset_v2_20240220/`.
- Maintain a changelog documenting additions and modifications.
- **Best for:** Small projects with infrequent updates.

### Best Practices
- Always version your dataset before training a model. Record the dataset version in the training config.
- Never modify images/labels in place. Create a new version.
- Store the `data.yaml` file with each version.
- Record annotation guidelines version alongside dataset version.

---

## Common Dataset Mistakes

### 1. Inconsistent Label Quality
- **Problem:** Different annotators have different standards (some draw tight boxes, others loose).
- **Impact:** Model learns inconsistent object boundaries, reducing localization accuracy.
- **Fix:** Create a detailed annotation guide with visual examples. Conduct calibration sessions. Review inter-annotator agreement.

### 2. Missing Labels
- **Problem:** Objects in images are not annotated (annotator missed them).
- **Impact:** Model is penalized for correctly detecting these objects (counted as FP), learns to suppress correct detections.
- **Fix:** Multi-pass annotation review. Use a trained model to find potential missed annotations.

### 3. Data Leakage Between Splits
- **Problem:** Very similar images (e.g., consecutive video frames) in both train and val sets.
- **Impact:** Inflated validation metrics that don't reflect real-world performance.
- **Fix:** Use group-based splitting. Check for near-duplicates using perceptual hashing.

### 4. Class Definition Ambiguity
- **Problem:** Unclear boundaries between classes (e.g., "truck" vs "van" vs "SUV").
- **Impact:** Inconsistent labels, confused model, high inter-class confusion.
- **Fix:** Define explicit rules for every ambiguous case. Use a decision tree for annotators.

### 5. Ignoring Image Quality
- **Problem:** Blurry, overexposed, or extremely dark images in the dataset.
- **Impact:** Model wastes capacity learning from uninformative images.
- **Fix:** Filter out images below a quality threshold (e.g., Laplacian variance for blur detection).

### 6. Wrong Coordinate Format
- **Problem:** Using pixel coordinates instead of normalized coordinates, or confusing (x1,y1,x2,y2) with (x_center,y_center,w,h).
- **Impact:** Bounding boxes appear random; model cannot learn.
- **Fix:** Validate a sample of labels by drawing boxes on images and visually verifying.

### 7. Class ID Mismatch
- **Problem:** Label files use class IDs not matching `data.yaml` names.
- **Impact:** Model assigns wrong class names, or crashes if class ID exceeds `nc-1`.
- **Fix:** Script to scan all label files and verify class IDs are within range [0, nc-1].

### 8. Not Enough Variety
- **Problem:** All images captured in same conditions (same lighting, same angle, same background).
- **Impact:** Model overfits to those conditions, fails in deployment.
- **Fix:** Collect data across different times of day, weather conditions, cameras, angles. Use augmentation to simulate variety.

### 9. Extremely Unbalanced Split
- **Problem:** All hard examples end up in val, or rare classes only in train.
- **Impact:** Unreliable metrics, model evaluated on unrepresentative data.
- **Fix:** Use stratified splitting by class composition and difficulty.

### 10. Ignoring the Test Set
- **Problem:** Using test set for hyperparameter tuning or model selection.
- **Impact:** Test metrics are no longer an unbiased estimate of generalization.
- **Fix:** Only evaluate on test set once, after all tuning is done on the validation set. The test set is your final report card.

---

## Dataset Size Guidelines

| Object Complexity | Minimum Images/Class | Recommended Images/Class | Notes |
|------------------|---------------------|-------------------------|-------|
| Simple (1-2 classes, clear objects) | 100 | 500+ | E.g., coin detection |
| Moderate (5-10 classes, varied) | 300 | 1000+ | E.g., vehicle types |
| Complex (20+ classes, diverse) | 500 | 2000+ | E.g., COCO-like tasks |
| Fine-grained (similar classes) | 1000 | 5000+ | E.g., bird species |

**Note:** These are guidelines for fine-tuning from pretrained weights. Training from scratch requires 5-10x more data.
