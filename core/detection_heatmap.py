"""
Tespit Bazli Grad-CAM Heatmap Raporu
=====================================
Her tespit icin ayri Grad-CAM heatmap crop'u olusturur.
Model bir nesneyi tespit ederken tam olarak neye baktigini gosterir.

Cikti: Tam gorsel (heatmap overlay + bbox'lar) + her tespit icin
       orijinal crop ve heatmap crop yan yana.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os

from ultralytics import YOLO
from .gradcam import YOLOGradCAM
from .preprocess import load_and_preprocess


def _cam_to_original_space(cam, img_shape, imgsz):
    """Grad-CAM'i letterbox uzayindan orijinal gorsel uzayina donustur."""
    h_orig, w_orig = img_shape[:2]
    scale = min(imgsz / h_orig, imgsz / w_orig)
    new_h, new_w = int(h_orig * scale), int(w_orig * scale)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2

    cam_lb = cv2.resize(cam, (imgsz, imgsz))
    cam_content = cam_lb[top:top + new_h, left:left + new_w]
    cam_orig = cv2.resize(cam_content, (w_orig, h_orig))
    return cam_orig


def _make_overlay(img_rgb, cam, alpha=0.45):
    """Gorsel uzerine heatmap overlay uygula."""
    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * img_rgb + (1 - alpha) * heatmap)
    return overlay


def visualize_detection_heatmap(model_path, image_path, output_dir,
                                class_names, imgsz=640, max_detections=8):
    """
    Tespit bazli Grad-CAM rapor gorseli olustur.

    Args:
        model_path: YOLO model dosyasi
        image_path: Analiz edilecek gorsel
        output_dir: Cikti klasoru
        class_names: {idx: name} sozlugu
        imgsz: Model giris boyutu
        max_detections: Maksimum gosterilecek tespit sayisi
    """
    # 1. Grad-CAM hazirla
    try:
        gradcam = YOLOGradCAM(model_path)
    except Exception as e:
        print(f"  Grad-CAM baslatilamadi: {e}")
        return

    tensor, img_rgb, img_letterbox = load_and_preprocess(image_path, imgsz)

    # 2. Inference - tespitleri al
    yolo = YOLO(model_path)
    results = yolo(image_path, verbose=False)
    detections = results[0].boxes

    nc = len(class_names)

    # 3. Her sinif icin Grad-CAM uret
    print("  Sinif bazli Grad-CAM uretiliyor...")
    cams = {}
    for cls_idx in range(nc):
        try:
            cam = gradcam.generate(tensor.clone(), class_idx=cls_idx)
            if cam is not None:
                cams[cls_idx] = cam
        except Exception as e:
            print(f"  Sinif {class_names[cls_idx]} icin hata: {e}")
            continue

    if not cams:
        print("  Grad-CAM uretilemedi")
        return

    # 4. CAM'leri orijinal gorsel uzayina donustur
    cams_orig = {}
    for cls_idx, cam in cams.items():
        cams_orig[cls_idx] = _cam_to_original_space(cam, img_rgb.shape, imgsz)

    # 5. Birlesik heatmap (tum siniflarin max'i)
    combined_cam = np.max(list(cams_orig.values()), axis=0)

    # 6. Tespit listesini hazirla (conf sirali)
    det_list = []
    if detections is not None and len(detections) > 0:
        confs = detections.conf.cpu().numpy()
        sorted_indices = np.argsort(-confs)

        h_img, w_img = img_rgb.shape[:2]
        for idx in sorted_indices[:max_detections]:
            box = detections[idx]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            det_list.append({
                'bbox': (x1, y1, x2, y2),
                'cls': cls,
                'conf': conf,
                'name': class_names.get(cls, f"cls_{cls}")
            })

    n_det = len(det_list)

    # Renk paleti
    box_colors_rgb = [
        (230, 50, 50), (50, 200, 50), (50, 100, 230),
        (230, 200, 50), (200, 50, 200), (50, 200, 200),
        (255, 140, 0), (140, 0, 255)
    ]
    box_colors_mpl = [(r / 255, g / 255, b / 255)
                      for r, g, b in box_colors_rgb]

    # 7. Full overlay gorsel
    full_overlay = _make_overlay(img_rgb, combined_cam)

    # Bbox'lari ciz
    for i, det in enumerate(det_list):
        x1, y1, x2, y2 = det['bbox']
        color = box_colors_rgb[det['cls'] % len(box_colors_rgb)]
        cv2.rectangle(full_overlay, (x1, y1), (x2, y2), color, 2)
        label = f"#{i + 1} {det['name']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(full_overlay, (x1, max(y1 - th - 6, 0)),
                      (x1 + tw + 4, max(y1, th + 6)), color, -1)
        cv2.putText(full_overlay, label, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 8. Rapor gorseli olustur
    model_name = os.path.basename(model_path)
    image_name = os.path.basename(image_path)

    if n_det == 0:
        # Tespit yok - sinif bazli Grad-CAM goster
        print("  Tespit bulunamadi, genel sinif bazli Grad-CAM gosteriliyor")
        n_panels = len(cams_orig) + 1
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        axes[0].imshow(full_overlay)
        axes[0].set_title("Birlesik Heatmap\n(tespit yok)",
                          fontsize=11, fontweight='bold')
        axes[0].axis('off')

        for i, (cls_idx, cam_o) in enumerate(cams_orig.items()):
            overlay = _make_overlay(img_rgb, cam_o)
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(
                f"Grad-CAM: {class_names[cls_idx]}",
                fontsize=11, fontweight='bold')
            axes[i + 1].axis('off')

        fig.suptitle(
            f"HASAR TESPIT RAPORU - Grad-CAM Analizi\n"
            f"Model: {model_name}  |  Gorsel: {image_name}",
            fontsize=13, fontweight='bold')
    else:
        # Tespit var - per-detection crops ile rapor
        print(f"  {n_det} tespit bulundu, per-detection heatmap olusturuluyor")
        cols = n_det
        fig = plt.figure(figsize=(max(3.5 * cols, 10), 11))
        gs = gridspec.GridSpec(3, cols, height_ratios=[2.5, 1, 1],
                               hspace=0.35, wspace=0.3)

        # Ust: tam gorsel + heatmap overlay
        ax_full = fig.add_subplot(gs[0, :])
        ax_full.imshow(full_overlay)
        ax_full.set_title(f"{n_det} tespit bulundu", fontsize=11,
                          fontweight='bold')
        ax_full.axis('off')

        # Alt: per-detection crops
        for i, det in enumerate(det_list):
            x1, y1, x2, y2 = det['bbox']
            cls_idx = det['cls']

            # Orijinal crop
            crop_orig = img_rgb[y1:y2, x1:x2]

            # Sinifa ozgu heatmap crop
            cam_for_det = cams_orig.get(cls_idx, combined_cam)
            det_overlay = _make_overlay(img_rgb, cam_for_det)
            crop_heatmap = det_overlay[y1:y2, x1:x2]

            color = box_colors_mpl[cls_idx % len(box_colors_mpl)]

            # Orijinal crop
            ax_orig = fig.add_subplot(gs[1, i])
            ax_orig.imshow(crop_orig)
            ax_orig.set_title(
                f"#{i + 1} {det['name']}\nconf: {det['conf']:.2f}",
                fontsize=9, fontweight='bold', color=color)
            ax_orig.axis('off')

            # Heatmap crop
            ax_heat = fig.add_subplot(gs[2, i])
            ax_heat.imshow(crop_heatmap)
            ax_heat.set_title("Grad-CAM", fontsize=8)
            ax_heat.axis('off')

        fig.suptitle(
            f"HASAR TESPIT RAPORU - Grad-CAM Analizi\n"
            f"Model: {model_name}  |  Gorsel: {image_name}",
            fontsize=13, fontweight='bold')

    fig.set_tight_layout(False)
    fig.subplots_adjust(top=0.88)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "5_detection_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {path}")
