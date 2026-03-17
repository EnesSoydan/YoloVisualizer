"""
t-SNE Embedding Gorsellestiremesi
==================================
Modelin ogrendigi temsilleri 2 boyutta gosterir.
Her nokta bir gorsel, renkler sinif etiketlerini temsil eder.
Yakin noktalar = model icin benzer gorunen gorseller.
Iyi ayrilmis kumeler = modelin siniflari basariyla ogrendigi anlamina gelir.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

from tqdm import tqdm
from ultralytics import YOLO
from sklearn.manifold import TSNE

from .preprocess import load_and_preprocess


def get_dominant_class(label_path):
    """Label dosyasindan en cok gecen sinifi bul."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    classes = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            try:
                classes.append(int(parts[0]))
            except ValueError:
                continue
    if not classes:
        return None
    return max(set(classes), key=classes.count)


def visualize_tsne(model_path, images_dir, dataset_path, output_dir,
                   class_names, num_images=300, imgsz=640):
    """
    Feature extraction + t-SNE gorsellestiremesi.
    Modelin siniflari feature uzayinda nasil ayirdigini gosterir.
    """

    model = YOLO(model_path)
    nn_model = model.model
    nn_model.eval()

    # Feature extraction icin hedef katmani bul
    # SPPF veya son C2f katmani ideal (backbone'un ciktisi)
    target_layer = None
    target_name = ""
    for i, layer in enumerate(nn_model.model):
        cls_name = layer.__class__.__name__
        if 'SPPF' in cls_name:
            target_layer = layer
            target_name = f"model.{i}_{cls_name}"
            break

    if target_layer is None:
        # Fallback: son birkac katmandan C2f veya Conv olan birini sec
        for i in range(len(nn_model.model) - 1, -1, -1):
            layer = nn_model.model[i]
            cls_name = layer.__class__.__name__
            if cls_name in ('C2f', 'C3', 'Conv', 'SPPF'):
                target_layer = layer
                target_name = f"model.{i}_{cls_name}"
                break

    if target_layer is None:
        target_layer = nn_model.model[9]
        target_name = "model.9"

    # Forward hook
    features_store = {}

    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            features_store['feat'] = out.detach().cpu()

    handle = target_layer.register_forward_hook(hook)

    # Etiketli gorselleri topla
    labels_dir = os.path.join(dataset_path, "train", "labels")
    all_images = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    labeled_images = []
    for img_name in all_images:
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        cls = get_dominant_class(label_path)
        if cls is not None:
            labeled_images.append((img_name, cls))

    if len(labeled_images) > num_images:
        # Her siniftan dengeli ornekleme yap
        by_class = {}
        for img, cls in labeled_images:
            by_class.setdefault(cls, []).append((img, cls))

        per_class = max(1, num_images // len(by_class))
        sampled = []
        for cls_items in by_class.values():
            n = min(len(cls_items), per_class)
            sampled.extend(random.sample(cls_items, n))

        # Eksik kalirsa rastgele tamamla
        remaining = num_images - len(sampled)
        if remaining > 0:
            pool = [x for x in labeled_images if x not in sampled]
            sampled.extend(random.sample(pool, min(remaining, len(pool))))

        labeled_images = sampled

    print(f"  {len(labeled_images)} gorsel isleniyor...")

    all_features = []
    all_labels = []
    device = next(nn_model.parameters()).device

    for img_name, cls in tqdm(labeled_images, desc="  Feature extraction"):
        img_path = os.path.join(images_dir, img_name)
        try:
            tensor, _, _ = load_and_preprocess(img_path, imgsz)
            with torch.no_grad():
                nn_model(tensor.to(device))

            if 'feat' in features_store:
                feat = features_store['feat']
                # Global Average Pooling -> tek bir vektor
                feat_vec = feat.mean(dim=(2, 3)).squeeze().numpy()
                all_features.append(feat_vec)
                all_labels.append(cls)
        except Exception:
            continue

    handle.remove()

    if len(all_features) < 10:
        print("  Yeterli feature toplanamadi (min 10 gorsel gerekli)")
        return

    features = np.array(all_features)
    labels = np.array(all_labels)

    print(f"  t-SNE hesaplaniyor ({features.shape[0]} gorsel, "
          f"{features.shape[1]} boyutlu feature)...")

    perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                n_iter=1000)
    embeddings = tsne.fit_transform(features)

    # === Cizim ===
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f',
              '#9b59b6', '#e67e22', '#1abc9c', '#34495e']

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for cls_idx in sorted(class_names.keys()):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[cls_idx % len(colors)],
            label=f"{class_names[cls_idx]} (n={mask.sum()})",
            alpha=0.7, s=60, edgecolors='white', linewidth=0.5
        )

    ax.set_title(
        "t-SNE: Model Sinif Temsillerini Nasil Ayiriyor?\n"
        "(Yakin noktalar = model icin benzer gorunen gorseller)",
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel("t-SNE Boyut 1", fontsize=11)
    ax.set_ylabel("t-SNE Boyut 2", fontsize=11)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Bilgi kutusu
    ax.text(
        0.02, 0.02,
        f"Feature katmani: {target_name}\n"
        f"Toplam gorsel: {len(features)}\n"
        f"Feature boyutu: {features.shape[1]}\n"
        f"Perplexity: {perplexity}",
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "4_tsne_embeddings.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {path}")
