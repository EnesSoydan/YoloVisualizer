# YOLO Instance Segmentation: Detaylı Rehber

## Detection vs Segmentation Farkı

| Özellik | Detection (BBox) | Instance Segmentation |
|---------|-----------------|----------------------|
| Çıktı | Dikdörtgen kutu | Piksel maskesi + kutu |
| Etiket formatı | `class x y w h` | `class x1 y1 x2 y2 ... xN yN` |
| Eğitim süresi | Daha kısa | %15–30 daha uzun |
| Inference hızı | Daha hızlı | Biraz daha yavaş |
| Ne zaman kullan | Nesnenin yeri yeterliyse | Şekli önemliyse, üst üste nesneler varsa |

---

## YOLOv8-seg Nasıl Çalışır

YOLOv8-seg iki çıktı üretir:
1. **BBox:** Her nesne için bounding box koordinatları (detection ile aynı)
2. **Maske:** Her nesne için piksel düzeyinde binary maske (0/1)

Maske üretimi için bir **mask head** eklenmiştir — bu head 32 adet prototip maske öğrenir ve her nesne için bu prototipler doğrusal olarak birleştirilir (Mask R-CNN yaklaşımından farklı, çok daha hızlı).

---

## Etiket Hazırlama

### Polygon Etiket Formatı
```
0 0.1 0.2 0.35 0.15 0.6 0.25 0.55 0.5 0.3 0.55 0.12 0.4
```
- İlk sayı: class_id
- Geri kalanlar: normalize edilmiş (x, y) çiftleri
- Minimum 3 köşe noktası (üçgen), pratik olarak 6–20 arası yeterli

### Etiketleme Araçları
- **Roboflow:** Web tabanlı, polygon çizme + otomatik YOLO formatı export
- **CVAT:** Açık kaynak, güçlü polygon aracı
- **LabelMe:** Hafif, Python tabanlı
- **Labelbox:** Kurumsal kullanım

### Etiket Kalitesi İpuçları
- Polygon noktaları çok fazla olmasın — objeyi yeterince tanımlayan minimum nokta daha iyi
- Düz kenarlı nesneler için 4–6 köşe yeterli
- Karmaşık şekiller (insan silueti) için 20–30 köşe
- Aşırı detaylı polygon eğitimi zorlaştırır, overfitting riskini artırır

---

## Eğitim

```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # pretrained segmentation modeli

results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,          # GPU index
)
```

**data.yaml — segmentation için detection ile aynı:**
```yaml
path: /dataset
train: images/train
val: images/val
nc: 3
names: ['class1', 'class2', 'class3']
```

---

## Inference ve Sonuçları Kullanma

```python
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('best.pt')  # eğitilmiş model
results = model('image.jpg')

for result in results:
    # BBox erişimi
    boxes = result.boxes.xyxy    # [x1, y1, x2, y2]
    classes = result.boxes.cls   # class id'leri
    confs = result.boxes.conf    # confidence skorları

    # Maske erişimi
    if result.masks is not None:
        masks = result.masks.data  # (N, H, W) tensor — binary maskeler
        for i, mask in enumerate(masks):
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            # mask_np artık 0/255 değerli numpy array

    # Görsel üzerine çiz
    annotated = result.plot()
    cv2.imwrite('result.jpg', annotated)
```

---

## Segmentation Metrikleri

Detection metriklerine ek olarak segmentation modelleri şu metrikleri üretir:

| Metrik | Açıklama | İyi Değer |
|--------|----------|-----------|
| mAP50(B) | BBox için mAP@50 | >0.70 |
| mAP50-95(B) | BBox için mAP@50:95 | >0.50 |
| mAP50(M) | Maske için mAP@50 | >0.65 |
| mAP50-95(M) | Maske için mAP@50:95 | >0.45 |

Maske mAP'i her zaman BBox mAP'inden düşük olur — bu normal.

---

## Karışık Dataset: Hem BBox Hem Polygon Etiketi

Eğer dataset'inde bazı görseller polygon, bazıları bbox formatındaysa:

### YOLO'nun Davranışı
- YOLOv8-seg **her ikisini de kabul eder**
- BBox etiketli görseller detection loss'una katkı yapar
- Polygon etiketli görseller hem detection hem segmentation loss'una katkı yapar
- Sadece bbox etiketi olan görseller için maske loss hesaplanmaz

### Pratikte Ne Yapmalısın
Eğer zamanın varsa: **tüm görselleri polygon formatına çevir** — model daha iyi maske öğrenir.
Eğer zamanın yoksa: **karışık bırak** — çalışır ama maske kalitesi daha düşük olur.

---

## Yaygın Hatalar

**Hata: Model maske üretmiyor**
- Etiketler polygon formatında değil bbox formatında — segmentation etiketleri gerekli

**Hata: Maskeler çok kaba (piksel değil blok gibi)**
- `imgsz` değeri düşük — 640 yerine 1280 dene
- Model çok küçük — `n` yerine `s` veya `m` kullan

**Hata: "No masks found" inference sırasında**
- `result.masks` None dönüyor — model detection modeli, segmentation modeli değil
- `best.pt` yerine seg modelini yükle

**Hata: Eğitim çok yavaş**
- Batch size'ı düşür (16 → 8)
- `workers` parametresini artır
- Polygon noktalarını azalt (aşırı detaylı etiketler yavaşlatır)

---

## Export ve Deployment

```python
# ONNX formatına export
model.export(format='onnx', imgsz=640)

# TensorRT (Jetson veya NVIDIA GPU için)
model.export(format='engine', imgsz=640, device=0)

# OpenVINO (Intel CPU için)
model.export(format='openvino', imgsz=640)
```

Segmentation modeli export edildiğinde hem bbox hem maske çıktısı korunur.
