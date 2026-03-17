"""Gorsel on-isleme: YOLO modeline girecek tensoru hazirlar."""

import cv2
import torch
import numpy as np


def load_and_preprocess(image_path, imgsz=640):
    """
    Gorseli yukle, letterbox resize uygula, tensor'a cevir.

    Returns:
        tensor: (1, 3, imgsz, imgsz) normalize edilmis tensor
        img_rgb: Orijinal gorsel (RGB)
        img_letterbox: Letterbox uygulanmis gorsel (numpy)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gorsel bulunamadi: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Letterbox resize: en-boy oranini koruyarak yeniden boyutlandir
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Gri padding ile kare yap
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - new_h) // 2
    left = (imgsz - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = img_resized

    # PyTorch tensoruna cevir
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)

    return tensor, img_rgb, canvas
