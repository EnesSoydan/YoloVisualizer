"""
Konvolusyon Filtre Gorsellestiremesi
=====================================
Modelin ogrendigi filtreleri (cekirdekleri) gosterir:
  - Ilk katman filtreleri: kenar, renk, doku algilayicilari (insanlar icin anlasilir)
  - Derin katman filtreleri: soyut desenler (insanlar icin yorumlamasi zor)
  - Istatistikler: parametre dagilimi, katman buyuklukleri
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from ultralytics import YOLO


def visualize_filters(model_path, output_dir):
    """Konvolusyon filtrelerini gorsellestir ve kaydet."""

    model = YOLO(model_path)
    nn_model = model.model.model

    # Tum Conv2d katmanlarini topla
    conv_layers = []
    for i, layer in enumerate(nn_model):
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                full_name = f"layer{i}.{name}" if name else f"layer{i}"
                conv_layers.append((full_name, module))

    if not conv_layers:
        print("  Conv2d katmani bulunamadi")
        return

    # === Sekil 1: Ilk katman filtreleri (en yorumlanabilir) ===
    first_name, first_conv = conv_layers[0]
    weights = first_conv.weight.detach().cpu()  # (out_ch, in_ch, kH, kW)

    n_filters = min(weights.shape[0], 64)
    cols = 8
    rows = max(1, (n_filters + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 2 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_filters:
            w = weights[i]
            if w.shape[0] == 3:
                # 3 kanalli (RGB) -> renkli goster
                w = w.permute(1, 2, 0).numpy()
                w = (w - w.min()) / (w.max() - w.min() + 1e-8)
                ax.imshow(w)
            else:
                # Tek kanal veya ortalama goster
                w_show = w.mean(dim=0).numpy()
                limit = max(abs(w_show.min()), abs(w_show.max()))
                ax.imshow(w_show, cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set_title(f"F{i}", fontsize=7)
        ax.axis('off')

    fig.suptitle(
        f"Ilk Katman Filtreleri: {first_name}\n"
        f"{weights.shape[0]} filtre, "
        f"{weights.shape[2]}x{weights.shape[3]} boyut, "
        f"{weights.shape[1]} girdi kanali\n"
        f"(Bu filtreler kenar, koyu-acik gecis ve renk farki algiliyor)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path1 = os.path.join(output_dir, "3_filters_first_layer.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {path1}")

    # === Sekil 2: Katman bazli istatistikler ===
    layer_stats = []
    for name, conv in conv_layers[:25]:  # Ilk 25 katman
        w = conv.weight.detach().cpu()
        layer_stats.append({
            'name': name,
            'out_ch': w.shape[0],
            'in_ch': w.shape[1],
            'kernel': f"{w.shape[2]}x{w.shape[3]}",
            'std': w.std().item(),
            'n_params': w.numel()
        })

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    x = range(len(layer_stats))
    short_names = [s['name'][:12] for s in layer_stats]

    # Standart sapma (ogrenme aktivitesi gostergesi)
    stds = [s['std'] for s in layer_stats]
    ax1.bar(x, stds, color='steelblue')
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax1.set_ylabel('Standart Sapma')
    ax1.set_title('Filtre Agirlik Dagilimi (yuksek = daha aktif ogrenme)',
                   fontweight='bold')

    # Parametre sayisi
    params = [s['n_params'] / 1000 for s in layer_stats]
    ax2.bar(x, params, color='coral')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('Parametre (x1000)')
    ax2.set_title('Parametre Sayisi per Katman', fontweight='bold')

    # Kanal sayisi
    channels = [s['out_ch'] for s in layer_stats]
    ax3.bar(x, channels, color='seagreen')
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Cikti Kanali')
    ax3.set_title('Kanal Sayisi (derinlestikce artar)', fontweight='bold')

    total_params = sum(s['n_params'] for s in layer_stats)
    all_params = sum(p.numel() for p in model.model.parameters())
    fig.suptitle(
        f"Model Mimarisi Ozeti\n"
        f"Gosterilen: {total_params:,} parametre | "
        f"Toplam model: {all_params:,} parametre",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    path2 = os.path.join(output_dir, "3_filter_statistics.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {path2}")
