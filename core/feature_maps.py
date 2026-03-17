"""
Feature Map Gorsellestiremesi
=============================
Her katmanin gorseli nasil gordugunu gosterir:
  - Erken katmanlar: kenarlar, basit cizgiler
  - Orta katmanlar: dokular, tekrar eden desenler
  - Derin katmanlar: obje parcalari, soyut temsiller
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from ultralytics import YOLO
from .preprocess import load_and_preprocess


def visualize_feature_maps(model_path, image_path, output_dir, imgsz=640):
    """Feature map gorsellestiremesi olustur ve kaydet."""

    model = YOLO(model_path)
    nn_model = model.model.model  # Sequential katmanlar

    # Her katmandan aktivasyonlari topla
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor) and len(out.shape) == 4:
                activations[name] = out.detach().cpu()
        return hook

    for i, layer in enumerate(nn_model):
        cls_name = layer.__class__.__name__
        h = layer.register_forward_hook(make_hook(f"{i}_{cls_name}"))
        hooks.append(h)

    # Forward pass
    tensor, img_rgb, img_letterbox = load_and_preprocess(image_path, imgsz)
    device = next(model.model.parameters()).device
    with torch.no_grad():
        model.model(tensor.to(device))

    for h in hooks:
        h.remove()

    # Sadece uzaysal boyutu olan katmanlari filtrele
    conv_layers = {k: v for k, v in activations.items()
                   if v.shape[2] > 1 and v.shape[3] > 1}

    layer_names = list(conv_layers.keys())
    total = len(layer_names)

    # 5 temsili katman sec: erken -> derin
    if total >= 5:
        indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
    else:
        indices = list(range(total))

    selected = [layer_names[i] for i in indices]
    depth_labels = [
        "Erken Katman\n(Kenarlar/Cizgiler)",
        "Erken-Orta\n(Basit Dokular)",
        "Orta Katman\n(Karmasik Dokular)",
        "Orta-Derin\n(Obje Parcalari)",
        "Derin Katman\n(Soyut Temsiller)"
    ]

    # === Sekil 1: Katman ilerlemesi ozet ===
    n_sel = len(selected)
    fig, axes = plt.subplots(1, n_sel + 1, figsize=(4 * (n_sel + 1), 4))

    axes[0].imshow(img_letterbox)
    axes[0].set_title("Girdi Gorsel", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for idx, layer_name in enumerate(selected):
        feat = conv_layers[layer_name][0]  # (C, H, W)
        # Tum kanallarin ortalamasini goster
        mean_feat = feat.mean(dim=0).numpy()
        axes[idx + 1].imshow(mean_feat, cmap='inferno')
        label = depth_labels[idx] if idx < len(depth_labels) else f"Katman {idx}"
        ch_count = feat.shape[0]
        spatial = f"{feat.shape[1]}x{feat.shape[2]}"
        axes[idx + 1].set_title(
            f"{label}\n{layer_name}\n({ch_count} kanal, {spatial})",
            fontsize=8
        )
        axes[idx + 1].axis('off')

    fig.suptitle(
        "YOLO Katman Ilerlemesi: Gorsel Agdan Nasil Geciyor?",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    path1 = os.path.join(output_dir, "1_feature_progression.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {path1}")

    # === Sekil 2: Her katmanin detayli feature map gridi ===
    for sel_idx, layer_name in enumerate(selected):
        feat = conv_layers[layer_name][0]  # (C, H, W)
        n_show = min(feat.shape[0], 32)

        cols = 8
        rows = max(1, (n_show + cols - 1) // cols)

        fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i < n_show:
                ax.imshow(feat[i].numpy(), cmap='viridis')
                ax.set_title(f"Ch{i}", fontsize=7)
            ax.axis('off')

        depth_label = depth_labels[sel_idx] if sel_idx < len(depth_labels) else ""
        fig.suptitle(
            f"Feature Maps: {layer_name} - "
            f"{depth_label.replace(chr(10), ' ')}\n"
            f"Toplam {feat.shape[0]} kanal, "
            f"{feat.shape[1]}x{feat.shape[2]} boyut",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        layer_id = layer_name.split('_')[0]
        path = os.path.join(output_dir, f"1_detail_{sel_idx}_{layer_id}.png")
        plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  -> {path}")

    print(f"  Toplam {len(activations)} katman analiz edildi, "
          f"{len(selected)} tanesi gorsellesti.")
