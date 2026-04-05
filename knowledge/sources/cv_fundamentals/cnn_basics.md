# Convolutional Neural Network (CNN) Fundamentals

## Overview

Convolutional Neural Networks are the backbone of modern computer vision. They exploit spatial locality and translation invariance to learn hierarchical feature representations from raw pixel data. This document covers the core building blocks used across all major detection architectures including YOLO, EfficientDet, and DETR variants.

---

## 1. Convolution Operation

### Mathematical Definition

For a 2D input feature map **X** and kernel **W**, the discrete convolution at position (i, j) is:

```
Y(i, j) = sum_m sum_n X(i*s + m, j*s + n) * W(m, n) + b
```

Where `s` is stride, `m` and `n` iterate over the kernel dimensions, and `b` is the bias term.

Technically, deep learning frameworks implement **cross-correlation** (no kernel flip), but by convention it is called convolution.

### Intuition

A convolution slides a small learnable filter across the input, computing element-wise multiplication and summation at each position. The output is a feature map that encodes where and how strongly a particular pattern is detected.

### Stride

- **Stride = 1**: The kernel moves one pixel at a time. Output spatial size is close to input size.
- **Stride = 2**: The kernel moves two pixels at a time. Output spatial size is roughly halved. This is often used instead of pooling for downsampling in modern architectures (e.g., YOLO, ResNet-D).

Output size formula: `O = floor((I - K + 2P) / S) + 1` where I = input size, K = kernel size, P = padding, S = stride.

### Padding

- **Valid (no padding)**: Output shrinks by `(K-1)` pixels. Used rarely in modern networks.
- **Same padding**: `P = (K-1) / 2`. Preserves spatial dimensions when stride = 1. Standard for 3x3 convolutions.
- **Causal padding**: Used in temporal/1D cases, not typical in vision.

---

## 2. Convolution Filters/Kernels

### What Filters Learn at Each Layer

Filters are **not hand-designed**; they are learned via backpropagation. However, there is a well-established pattern in what they learn:

| Layer Depth | What Filters Learn | Example |
|---|---|---|
| Layer 1-2 | Edges, color gradients, simple textures | Horizontal/vertical edges, color blobs |
| Layer 3-5 | Textures, patterns, parts | Grid patterns, circles, corners |
| Layer 6-10 | Object parts | Wheels, eyes, windows |
| Layer 10+ | Whole objects, scenes | Faces, cars, entire animals |

### Kernel Sizes in Practice

- **3x3**: The dominant kernel size in modern architectures. Two stacked 3x3 convolutions have the same receptive field as one 5x5 but with fewer parameters (18 vs 25) and more non-linearity.
- **1x1**: Channel mixing without spatial interaction. Used in bottleneck layers.
- **5x5, 7x7**: Sometimes used in the very first layer (e.g., ResNet stem uses 7x7) to capture larger initial patterns.
- **Large kernels (31x31, 51x51)**: Resurgent in architectures like RepLKNet and ConvNeXt v2 for capturing long-range dependencies.

---

## 3. Pooling Layers

### Max Pooling

Selects the maximum value within each pooling window. Provides slight translation invariance and retains the strongest activations. Common configuration: 2x2 window with stride 2 (halves spatial dimensions).

### Average Pooling

Computes the arithmetic mean within each pooling window. Smoother than max pooling but can lose sharp feature responses.

### Global Average Pooling (GAP)

Averages the entire spatial extent of each channel into a single value, producing a vector of length C (number of channels). Replaces fully connected layers at the end of classification networks. Benefits:
- No learnable parameters
- Acts as a structural regularizer
- Enables arbitrary input sizes at inference time

### Pooling in Modern Detection Networks

Modern architectures (YOLO, EfficientNet) largely **replace pooling with strided convolutions** for downsampling, as strided convolutions are learnable. Pooling survives mainly in SPP/SPPF modules and GAP in classification heads.

---

## 4. Activation Functions

### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

- Simple and computationally efficient
- Suffers from "dying ReLU" problem: neurons that output 0 for all inputs get zero gradient and stop learning
- Still widely used in older architectures

### Leaky ReLU

```
f(x) = x if x > 0, else alpha * x    (alpha typically 0.01 or 0.1)
```

- Addresses the dying ReLU problem with a small negative slope
- Used in YOLOv3, YOLOv4

### SiLU / Swish

```
f(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

- Smooth, non-monotonic activation
- Self-gating mechanism: the input gates itself via the sigmoid
- Default activation in YOLOv5, YOLOv8, and many modern architectures
- Slightly more expensive than ReLU but consistently better accuracy

### Mish

```
f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
```

- Similar properties to SiLU but with slightly different gradient landscape
- Used in YOLOv4 CSPDarknet backbone
- Marginally better than SiLU in some benchmarks but more computationally expensive

### GELU (Gaussian Error Linear Unit)

```
f(x) = x * Phi(x)    where Phi is the CDF of standard normal distribution
```

- Approximation: `f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- Standard in Transformer-based models (ViT, Swin, DETR)
- Smooth gating based on the probability of the input being positive

---

## 5. Batch Normalization

### Operation

For a mini-batch of activations at a given layer:

```
x_hat = (x - mean_batch) / sqrt(var_batch + epsilon)
y = gamma * x_hat + beta
```

Where `gamma` (scale) and `beta` (shift) are learnable parameters, and `epsilon` is a small constant for numerical stability (typically 1e-5).

### Why It Works

- **Reduces internal covariate shift**: Stabilizes the distribution of layer inputs during training
- **Enables higher learning rates**: Smoother loss landscape
- **Acts as a regularizer**: The noise from mini-batch statistics provides slight regularization
- **Accelerates convergence**: Networks with BN typically train 5-10x faster

### When to Use

- Standard choice for CNN-based architectures (YOLO, ResNet, etc.)
- Place **before or after** activation (both work; after is more common in practice)
- At inference: uses running mean/variance computed during training (not batch statistics)
- **Do not use** with very small batch sizes (< 8); use Group Normalization or Layer Normalization instead
- Interacts poorly with dropout in some configurations

---

## 6. Dropout

### Mechanism

During training, each neuron output is independently set to zero with probability `p` (typically 0.1-0.5). At inference, all neurons are active but outputs are scaled by `(1-p)` (or equivalently, training outputs are scaled by `1/(1-p)` using inverted dropout).

### Purpose

- **Regularization**: Prevents co-adaptation of neurons, forcing the network to learn redundant representations
- **Ensemble approximation**: Dropout can be interpreted as training an exponential number of sub-networks simultaneously

### Usage in Detection Networks

- Rarely used in convolutional layers of modern detection networks (BN provides sufficient regularization)
- Sometimes applied in the classification head or in fully connected layers
- **Spatial Dropout (Dropout2D)**: Drops entire channels rather than individual neurons; more appropriate for convolutional features
- **DropBlock**: Drops contiguous regions of feature maps; used in some YOLO variants

---

## 7. Skip/Residual Connections (ResNet)

### The Problem

Deep networks suffer from **degradation**: adding more layers eventually increases training error, not due to overfitting but due to optimization difficulty.

### Solution

A residual block computes:

```
y = F(x) + x
```

Where `F(x)` is the residual function (a stack of convolutions + BN + activation). The identity shortcut `x` allows gradients to flow directly through the network.

### Why Residuals Work

- **Gradient highway**: Gradients can propagate through the identity path without vanishing
- **Easier optimization**: The network only needs to learn the residual `F(x) = y - x`, which is typically easier than learning the full mapping
- **Enables very deep networks**: ResNet-152, ResNet-1000+ become trainable

### Variants

- **Bottleneck residual block**: 1x1 (reduce) -> 3x3 (convolve) -> 1x1 (expand). Standard in ResNet-50+.
- **Pre-activation ResNet**: BN -> ReLU -> Conv ordering (ResNet v2)
- **CSP (Cross Stage Partial)**: Splits channels, applies residual blocks to one part, then concatenates. Used in YOLOv4, YOLOv5, and YOLOv8.

---

## 8. Depthwise Separable Convolutions (MobileNet)

### Standard Convolution Cost

For input `C_in` channels, output `C_out` channels, kernel `K x K`, spatial size `H x W`:
```
Parameters: K * K * C_in * C_out
Computation: K * K * C_in * C_out * H * W
```

### Depthwise Separable Decomposition

**Step 1 — Depthwise Convolution**: Apply one `K x K` filter per input channel independently.
```
Parameters: K * K * C_in
```

**Step 2 — Pointwise Convolution**: Apply `1x1` convolutions to combine channel information.
```
Parameters: C_in * C_out
```

### Computational Savings

Reduction ratio: `1/C_out + 1/K^2`. For a 3x3 convolution with 256 output channels, this is approximately **8-9x fewer computations**.

### Usage

- Core building block of MobileNet v1/v2/v3
- Used in lightweight YOLO variants (YOLOv5-nano, YOLOv8-nano)
- Slightly lower representational capacity per layer but dramatically more efficient

---

## 9. 1x1 Convolutions (Bottleneck Layers)

### Purpose

- **Channel mixing**: Combines information across channels without spatial interaction
- **Dimensionality reduction**: Reduces channel count before expensive 3x3 convolutions (bottleneck design)
- **Dimensionality expansion**: Expands channels after a bottleneck
- **Adding non-linearity**: A 1x1 convolution followed by activation adds a non-linear transformation cheaply

### Applications in Detection

- ResNet bottleneck blocks: 256ch -> 1x1 -> 64ch -> 3x3 -> 64ch -> 1x1 -> 256ch
- YOLO CSP blocks use 1x1 convolutions extensively for channel adjustment
- Feature pyramid lateral connections often use 1x1 convolutions to match channel dimensions

---

## 10. Common Backbone Architectures

### VGG (2014)

- Stack of 3x3 convolutions with max pooling
- Simple but large: VGG-16 has 138M parameters
- Demonstrated that depth matters; rarely used as a backbone today

### ResNet (2015)

- Introduced residual connections
- Variants: ResNet-18/34 (basic blocks), ResNet-50/101/152 (bottleneck blocks)
- Still a common baseline backbone for detection (Faster R-CNN + ResNet-50)

### DarkNet (2016-2020)

- YOLO-specific backbone series
- Darknet-53 (YOLOv3): 53 layers, residual connections, no pooling (strided convolutions)
- CSPDarknet (YOLOv4/v5): Cross Stage Partial design for better gradient flow and reduced computation

### CSPNet (2019)

- Splits feature maps into two parts: one goes through dense blocks, the other bypasses
- Reduces computational cost while maintaining accuracy
- Foundation of YOLOv4, YOLOv5, and YOLOv7 backbones

### EfficientNet (2019)

- Uses Neural Architecture Search (NAS) to find optimal scaling of width, depth, and resolution
- Compound scaling: balances all three dimensions simultaneously
- EfficientNet-B0 through B7: increasingly larger and more accurate
- Used as backbone in EfficientDet

---

## 11. Feature Hierarchy

### How CNNs Build Representations

CNNs learn a hierarchical decomposition of visual information:

```
Input Image (RGB pixels)
    |
    v
Layer 1-2: Low-level features
    - Edges (horizontal, vertical, diagonal)
    - Color gradients and simple color patches
    - Gabor-like filters
    |
    v
Layer 3-5: Mid-level features
    - Textures (fur, brick, water)
    - Corners and junctions
    - Simple shapes (circles, rectangles)
    - Repetitive patterns
    |
    v
Layer 6-10: High-level features
    - Object parts (wheels, eyes, doors)
    - Compositional structures
    - Semantic regions
    |
    v
Layer 10+: Semantic features
    - Whole objects (car, person, dog)
    - Scene understanding
    - Abstract category representations
```

### Implications for Object Detection

- **Small objects** are best detected using low/mid-level feature maps (higher spatial resolution)
- **Large objects** are best detected using high-level feature maps (richer semantics)
- **Feature Pyramid Networks** (FPN) bridge this gap by combining features from multiple levels
- This hierarchy directly motivates multi-scale detection architectures used in all YOLO versions

### Receptive Field

The receptive field is the region of the input image that influences a single output neuron. It grows with network depth:
- After one 3x3 conv: 3x3 receptive field
- After two 3x3 convs: 5x5 receptive field
- After three 3x3 convs: 7x7 receptive field

Effective receptive field (the region that actually contributes significantly) is typically much smaller than the theoretical receptive field, concentrated in the center with a Gaussian-like distribution.

---

## Quick Reference: YOLO Backbone Evolution

| YOLO Version | Backbone | Key Feature |
|---|---|---|
| YOLOv1 | GoogLeNet-inspired | 24 conv layers |
| YOLOv2 | Darknet-19 | Batch normalization throughout |
| YOLOv3 | Darknet-53 | Residual connections |
| YOLOv4 | CSPDarknet-53 | Cross Stage Partial connections |
| YOLOv5 | CSPDarknet + Focus | Focus layer for initial downsampling |
| YOLOv7 | E-ELAN | Extended efficient layer aggregation |
| YOLOv8 | C2f-based CSPDarknet | C2f blocks replace C3 |
| YOLOv9 | GELAN | Generalized ELAN with PGI |
| YOLOv10 | CSPDarknet variant | NMS-free design changes |
| YOLOv11 | C3k2-based backbone | Efficient small-kernel design |
| YOLO12 | Attention-enhanced | Area attention integration |

---

## Summary

Understanding these CNN building blocks is essential for interpreting, modifying, and optimizing detection models. Every YOLO architecture is composed of these fundamental operations in different configurations. The trend has been toward: (1) replacing pooling with strided convolutions, (2) using smooth activations like SiLU, (3) incorporating residual and CSP connections for gradient flow, and (4) leveraging efficient designs like depthwise separable convolutions for mobile deployment.
