"""
Grad-CAM Gorsellestiremesi
===========================
Model bir nesneyi tespit ederken gorselin neresine baktigini gosterir.
Her sinif icin ayri isi haritasi uretir:
  - Kirmizi/sari bolgeler: modelin en cok odaklandigi yerler
  - Mavi/siyah bolgeler: modelin ilgilenmedigi yerler
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from ultralytics import YOLO
from .preprocess import load_and_preprocess


class YOLOGradCAM:
    """YOLO detection modeli icin Grad-CAM uygulamasi."""

    def __init__(self, model_path, target_layer_name=None):
        self.yolo = YOLO(model_path)
        self.model = self.yolo.model
        self.model.eval()

        # Hedef katmani bul (genelde SPPF - backbone'un son katmani)
        if target_layer_name is None:
            for i, layer in enumerate(self.model.model):
                if 'SPPF' in layer.__class__.__name__:
                    target_layer_name = f"model.{i}"
                    break
            if target_layer_name is None:
                target_layer_name = "model.9"

        self.target_layer_name = target_layer_name
        modules = dict(self.model.named_modules())

        if target_layer_name not in modules:
            available = [n for n in modules if n.startswith("model.")]
            raise ValueError(
                f"Katman bulunamadi: {target_layer_name}\n"
                f"Mevcut katmanlar: {available[:10]}"
            )

        self.target_layer = modules[target_layer_name]
        self.activations = None

        # Detect katmanini bul (NMS'i devre disi birakmak icin)
        self.detect_layer = None
        for m in self.model.modules():
            if m.__class__.__name__ == 'Detect':
                self.detect_layer = m
                break

        # Forward hook: aktivasyonu yakala ve retain_grad ile gradient'i koru
        self.target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, out):
        if isinstance(out, torch.Tensor):
            self.activations = out
            out.retain_grad()
        elif isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor):
            self.activations = out[0]
            out[0].retain_grad()

    def generate(self, img_tensor, class_idx=None):
        """
        Bir sinif icin Grad-CAM isi haritasi uret.

        Args:
            img_tensor: (1, 3, H, W) tensor
            class_idx: Hangi sinif icin CAM uretilecek (None = en yuksek skor)

        Returns:
            cam: (H, W) numpy array, 0-1 arasi normalize edilmis
        """
        self.model.zero_grad()
        self.activations = None

        # Gradient akisini etkinlestir
        for p in self.model.parameters():
            p.requires_grad_(True)

        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device).requires_grad_(True)

        # Detect katmanini gecici olarak train moduna al
        # Boylece NMS uygulanmaz ve gradient akisi korunur
        # (NMS diferansiyel degildir, gradient'i keser)
        detect_was_training = None
        if self.detect_layer is not None:
            detect_was_training = self.detect_layer.training
            self.detect_layer.training = True

        try:
            with torch.enable_grad():
                output = self.model(img_tensor)
        finally:
            # Detect katmanini eski haline dondur
            if self.detect_layer is not None and detect_was_training is not None:
                self.detect_layer.training = detect_was_training

        # Training modunda cikti: dict{one2many: {scores: (B,nc,8400), ...}}
        # Export modunda cikti: tensor (B, 300, 6)
        # Normal eval modunda: tuple (tensor, dict)
        if isinstance(output, dict):
            # Training mode output - one2many.scores kullan
            branch = output.get('one2many', output)
            scores = branch['scores']  # (B, nc, num_preds)
            scores = scores[0]  # (nc, num_preds)
        elif isinstance(output, (list, tuple)):
            output = output[0]
            if output.shape[-1] == 6:
                # Post-NMS format (B, 300, 6) - conf at index 4
                scores = output[0, :, 4:5].T  # (1, 300)
            else:
                nc = output.shape[1] - 4
                scores = output[0, 4:, :]  # (nc, num_preds)
        else:
            nc = output.shape[1] - 4
            scores = output[0, 4:, :]

        if class_idx is not None and len(scores.shape) > 1 and scores.shape[0] > class_idx:
            target_score = scores[class_idx].max()
        else:
            target_score = scores.max()

        # Backward pass
        target_score.backward(retain_graph=True)

        if self.activations is None:
            return None

        # Gradient'i aktivasyondan al (retain_grad sayesinde)
        gradients = self.activations.grad
        if gradients is None:
            return None

        act = self.activations.detach()
        grad = gradients.detach()

        # Grad-CAM: gradient'lerin global ortalama havuzlamasi ile
        # aktivasyonlarin agirlikli toplami
        weights = grad.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Sadece pozitif katkilar

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def visualize_gradcam(model_path, image_path, output_dir, class_names, imgsz=640):
    """Grad-CAM gorsellestiremesi olustur ve kaydet."""

    try:
        gradcam = YOLOGradCAM(model_path)
    except Exception as e:
        print(f"  Grad-CAM baslatilamadi: {e}")
        return

    tensor, img_rgb, img_letterbox = load_and_preprocess(image_path, imgsz)

    # Normal inference ile tespitleri al
    yolo = YOLO(model_path)
    results = yolo(image_path, verbose=False)
    detections = results[0].boxes

    nc = len(class_names)

    # Her sinif icin Grad-CAM uret
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

    # === Gorsel: Tespitler + her sinif icin Grad-CAM ===
    n_cams = len(cams)
    fig, axes = plt.subplots(1, n_cams + 1, figsize=(5 * (n_cams + 1), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Orijinal gorsel + tespit kutulari
    det_img = img_rgb.copy()
    box_colors = [
        (230, 50, 50), (50, 200, 50), (50, 100, 230), (230, 200, 50)
    ]
    if detections is not None and len(detections) > 0:
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            color = box_colors[cls % len(box_colors)]
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names.get(cls, cls)} {conf:.2f}"
            cv2.putText(det_img, label, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    axes[0].imshow(det_img)
    n_det = len(detections) if detections is not None else 0
    axes[0].set_title(f"Tespit Sonuclari ({n_det} obje)", fontsize=11,
                      fontweight='bold')
    axes[0].axis('off')

    # Grad-CAM isi haritalari
    for i, (cls_idx, cam) in enumerate(cams.items()):
        cam_resized = cv2.resize(cam, (img_letterbox.shape[1],
                                       img_letterbox.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255),
                                    cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.45 * img_letterbox + 0.55 * heatmap)

        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(
            f"Grad-CAM: {class_names[cls_idx]}",
            fontsize=11, fontweight='bold'
        )
        axes[i + 1].axis('off')

    fig.suptitle(
        "Model Nereye Bakiyor? (Grad-CAM Isi Haritasi)\n"
        f"Hedef katman: {gradcam.target_layer_name}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "2_gradcam.png")
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  -> {path}")
