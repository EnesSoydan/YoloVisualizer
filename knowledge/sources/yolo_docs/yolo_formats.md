# YOLO Etiket Formatları: BBox, Segmentation ve OBB

## Genel Bakış

YOLO modelleri dört farklı görev tipini destekler: detection (nesne tespiti), segmentation (örnek bölütleme), pose estimation (poz tahmini) ve classification (sınıflandırma). Her görev tipi farklı bir etiket formatı kullanır.

---

## 1. Detection (BBox) Formatı

**Kullanıldığı modeller:** YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12 — tüm `-det` (detection) varyantları

**Dosya formatı:** `.txt` (her görsel için ayrı dosya)

**Satır yapısı:**
```
<class_id> <x_center> <y_center> <width> <height>
```

- Tüm değerler **0–1 arasında normalize edilmiş** (görsel genişliği/yüksekliğine bölünmüş)
- `x_center`, `y_center`: bounding box'ın merkez koordinatları
- `width`, `height`: bounding box genişlik ve yüksekliği

**Örnek:**
```
0 0.5 0.4 0.3 0.2
1 0.7 0.6 0.15 0.25
```

---

## 2. Segmentation (Polygon) Formatı

**Kullanıldığı modeller:** YOLOv5-seg (kısmi), YOLOv8-seg, YOLOv9-seg, YOLOv11-seg — yalnızca `-seg` varyantları

**Dosya formatı:** `.txt` (detection formatıyla aynı dizin yapısı)

**Satır yapısı:**
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... <xN> <yN>
```

- Koordinatlar **normalize edilmiş** (0–1 arası)
- Poligon köşe sayısı değişken — objenin şekline göre 4'ten yüzlerce noktaya kadar çıkabilir
- Köşe noktaları saat yönünde veya saat yönünün tersinde sıralanabilir

**Örnek (6 köşeli poligon):**
```
0 0.1 0.2 0.3 0.1 0.5 0.2 0.6 0.4 0.4 0.5 0.2 0.4
```

**Önemli:** YOLOv8-seg eğitiminde segmentation etiketleri verilirse model hem **maskeleri** hem de **bounding box'ları** otomatik çıkarır. Segmentation etiketi olan bir dataset detection modeli için de kullanılabilir — YOLO segmentation etiketlerini detection için otomatik dönüştürür.

---

## 3. OBB (Oriented Bounding Box) Formatı

**Kullanıldığı modeller:** YOLOv8-obb, YOLOv11-obb

**Satır yapısı:**
```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

- 4 köşe noktası ile döndürülmüş dikdörtgen tanımlanır
- Havadan görüntü ve uydu görüntüsü analizinde kullanılır

---

## 4. Hem BBox Hem Polygon Etiketi Olan Dataset İçin Rehber

Eğer bir dataset'te bazı görseller bbox, bazıları polygon formatında etiketlenmişse:

### Seçenek A — Hepsini YOLOv8-seg ile Kullan (Önerilen)
- Polygon etiketleri doğrudan kullanılır
- BBox etiketleri de kabul edilir (YOLO karışık formatı destekler)
- Model: `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt` vb.

### Seçenek B — Polygon'ları BBox'a Dönüştür, Detection Modeli Kullan
```python
# Polygon'dan BBox çıkarma
def polygon_to_bbox(points):
    xs = points[0::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return x_center, y_center, width, height
```
- Bu durumda `yolov8n.pt`, `yolov8s.pt` vb. kullanılır
- Segmentation bilgisi kaybolur — sadece bbox kalır

### Seçenek C — İki Ayrı Model
- Detection için: bbox etiketleri → `yolov8-det`
- Segmentation için: polygon etiketleri → `yolov8-seg`

---

## Dizin Yapısı (Tüm Formatlar İçin Aynı)

```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       └── img003.jpg
└── labels/
    ├── train/
    │   ├── img001.txt
│   │   └── img002.txt
    └── val/
        └── img003.txt
```

**data.yaml örneği:**
```yaml
path: /dataset
train: images/train
val: images/val

nc: 3
names: ['cat', 'dog', 'bird']
```

---

## Model Varyantları Karşılaştırması

| Model | Detection | Segmentation | OBB | Pose |
|-------|-----------|--------------|-----|------|
| YOLOv5 | ✅ | ✅ (v5-seg) | ❌ | ❌ |
| YOLOv8 | ✅ | ✅ (-seg) | ✅ (-obb) | ✅ (-pose) |
| YOLOv9 | ✅ | ✅ (-seg) | ❌ | ❌ |
| YOLOv10 | ✅ | ❌ | ❌ | ❌ |
| YOLOv11 | ✅ | ✅ (-seg) | ✅ (-obb) | ✅ (-pose) |
| YOLOv12 | ✅ | ✅ (-seg) | ✅ (-obb) | ✅ (-pose) |

---

## Etiket Doğrulama

Ultralytics ile etiket formatını doğrulamak için:
```python
from ultralytics.data.utils import check_det_dataset
results = check_det_dataset('data.yaml')
```

Yaygın hatalar:
- Koordinatların 0–1 dışına çıkması → normalize etmeyi unutmak
- Boş `.txt` dosyası → negatif örnek (sorun değil, kasıtlıysa)
- Eksik `.txt` dosyası → YOLO bu görseli atlayabilir veya hata verir
