# YOLO Model Seçimi: Göreve ve Koşullara Göre Rehber

## Görev Tipine Göre Model Seçimi

### Nesne Tespiti (Object Detection)
Sadece bounding box çizmek istiyorsan:
- **Önerilen:** YOLOv8, YOLOv11, YOLOv12
- **Neden:** En olgun ekosistem, geniş topluluk desteği, Ultralytics entegrasyonu
- **Kullanım:** `yolov8n.pt` → `yolov8s.pt` → `yolov8m.pt` → `yolov8l.pt` → `yolov8x.pt`

### Örnek Bölütleme (Instance Segmentation)
Nesnenin piksel maskesini de çıkarmak istiyorsan (poligon etiket):
- **Önerilen:** YOLOv8-seg, YOLOv11-seg
- **Kullanım:** `yolov8n-seg.pt`, `yolov8s-seg.pt` vb.
- **Not:** Detection'dan %10–20 daha yavaş, maskeyi de döndürür

### Döndürülmüş BBox (OBB — Oriented Bounding Box)
Havadan görüntü, uydu, belge gibi nesnelerin yönü önemliyse:
- **Önerilen:** YOLOv8-obb, YOLOv11-obb
- **Kullanım:** `yolov8n-obb.pt`

### Poz Tahmini (Pose Estimation)
İnsan iskelet noktaları (keypoint) çıkarmak istiyorsan:
- **Önerilen:** YOLOv8-pose, YOLOv11-pose
- **Kullanım:** `yolov8n-pose.pt`

### Görsel Sınıflandırma (Classification)
Nesneyi tespit değil, görselin sınıfını tahmin etmek istiyorsan:
- **Önerilen:** YOLOv8-cls, YOLOv11-cls
- **Kullanım:** `yolov8n-cls.pt`

---

## Model Boyutuna Göre Seçim

| Suffix | Parametre | Hız | Doğruluk | Kullanım Durumu |
|--------|-----------|-----|----------|-----------------|
| n (nano) | ~3M | En hızlı | En düşük | Edge cihaz, gerçek zamanlı düşük kaynak |
| s (small) | ~11M | Hızlı | Orta-düşük | Mobil, Raspberry Pi, Jetson Nano |
| m (medium) | ~26M | Orta | Orta-iyi | Genel amaçlı, dengeli |
| l (large) | ~44M | Yavaş | İyi | GPU'lu sunucu, yüksek doğruluk |
| x (extra) | ~68M | En yavaş | En yüksek | Maksimum doğruluk, GPU zorunlu |

**Pratik kural:** Önce `yolov8s` ile başla, sonuç yeterliyse küçült (n), yetersizse büyüt (m → l).

---

## YOLO Sürüm Karşılaştırması

### YOLOv5
- Hâlâ yaygın kullanılan, kararlı
- Detection ve segmentation (-seg) destekler
- PyTorch tabanlı, kolay deployment
- **Ne zaman kullan:** Eski projelerde uyumluluk gerekiyorsa veya topluluğun büyüklüğü önemliyse

### YOLOv8 (Ultralytics)
- Detection, segmentation, OBB, pose, classification
- En geniş ekosistem ve dokümantasyon
- Python API'si çok olgun
- **Ne zaman kullan:** Yeni proje başlıyorsan ilk tercih bu olmalı

### YOLOv9
- GELAN (Generalized Efficient Layer Aggregation Network) mimarisi
- v8'e kıyasla küçük doğruluk artışı, benzer hız
- Segmentation (-seg) var, OBB/pose yok
- **Ne zaman kullan:** v8 ile iyi sonuç alamıyorsan dene

### YOLOv10
- NMS (Non-Maximum Suppression) kaldırıldı — end-to-end detection
- Segmentation yok — sadece detection
- **Ne zaman kullan:** NMS latency sorunun varsa veya deployment pipeline'ı basitleştirmek istiyorsan

### YOLOv11 (Ultralytics)
- v8'in geliştirilmiş versiyonu, daha az parametre ile daha iyi doğruluk
- Detection, segmentation, OBB, pose, classification hepsi var
- **Ne zaman kullan:** v8 ile iyi çalışıyorsan v11'e geçmek mantıklı — API aynı

### YOLOv12
- Attention tabanlı mimari (Flash Attention)
- Transformer avantajlarını YOLO'ya entegre eder
- Tüm görev tipleri destekleniyor
- **Ne zaman kullan:** Büyük ve karmaşık sahneler, global context önemliyse

---

## Gerçek Zamanlı mı, Yüksek Doğruluk mu?

**Gerçek zamanlı (>30 FPS):**
```
GPU varsa:   YOLOv8n / YOLOv11n
CPU'da:      YOLOv8n (640px input) veya daha küçük input boyutu
```

**Yüksek doğruluk (FPS önemli değil):**
```
YOLOv8x / YOLOv11x / YOLOv12x
```

**Edge cihaz (Jetson, Raspberry Pi):**
```
YOLOv8n → TensorRT ile export → 3-5x hız artışı
```

---

## Transfer Learning İçin Başlangıç Ağırlıkları

```python
from ultralytics import YOLO

# Detection
model = YOLO('yolov8s.pt')

# Segmentation
model = YOLO('yolov8s-seg.pt')

# OBB
model = YOLO('yolov8s-obb.pt')

# Pose
model = YOLO('yolov8s-pose.pt')

# Eğitim
model.train(data='data.yaml', epochs=100, imgsz=640)
```

COCO üzerinde önceden eğitilmiş ağırlıklar Ultralytics tarafından otomatik indirilir.

---

## Karar Ağacı

```
Görevin ne?
├── Bounding box çizmek → Detection modeli (yolov8s.pt)
├── Piksel maskesi de lazım → Segmentation (-seg)
├── Nesne yönü önemli (havadan görüntü) → OBB (-obb)
├── İnsan pozu → Pose (-pose)
└── Görsel sınıfı → Classification (-cls)

Kaynakların ne?
├── Güçlü GPU (RTX 3060+) → yolov8m veya üstü
├── Orta GPU (GTX 1660) → yolov8s
├── CPU veya zayıf GPU → yolov8n
└── Edge cihaz → yolov8n + TensorRT export

Önceliğin ne?
├── En iyi doğruluk → yolov8x veya yolov11x
├── Hız/doğruluk dengesi → yolov8s veya yolov11s
└── Sadece hız → yolov8n
```
