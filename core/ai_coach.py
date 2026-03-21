"""
AI Egitim Kocu
===============
Claude API kullanarak model egitim surecini analiz eder ve
Turkce oneriler sunar:
  - Egitim metrikleri yorumlama (loss, mAP, precision, recall)
  - Confusion matrix analizi (hangi siniflar karisiyor)
  - Dataset dengesi ve augmentation onerileri
  - Grad-CAM / t-SNE bulgularina dayali yorumlar
  - Sonraki adim onerileri
"""

import os
import json
import csv
import numpy as np
from pathlib import Path

try:
    import anthropic
except ImportError:
    raise ImportError("anthropic paketi gerekli: pip install anthropic")


SYSTEM_PROMPT = """Sen bir bilgisayarla gorme (Computer Vision) ve derin ogrenme uzmanisin.
YOLO tabanli nesne tespit modelleri konusunda derin bilgin var.

Gorevin: Kullanicinin model egitim verilerini, dataset istatistiklerini ve
gorsellestirme sonuclarini analiz edip Turkce olarak detayli, uygulanabilir
oneriler sunmak.

Yanitlarinda su formati kullan:
1. MEVCUT DURUM: Verilerin kisa ozeti
2. SORUNLAR: Tespit ettigin problemler (oncelik sirasina gore)
3. ONERILER: Her sorun icin somut, uygulanabilir cozumler
4. SONRAKI ADIMLAR: Oncelikli yapilmasi gerekenler (1-2-3 seklinde)

Teknik terimler icin Turkce aciklama ekle. Kullanicinin seviyesine uygun,
anlasilir bir dil kullan. Gereksiz uzatma, kisa ve oz ol."""


class AICoach:
    """Claude API ile model egitim analizi."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key bulunamadi.\n"
                "Environment variable olarak ayarla:\n"
                "  [Environment]::SetEnvironmentVariable("
                "'ANTHROPIC_API_KEY', 'sk-...', 'User')"
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _ask(self, prompt, max_tokens=2000):
        """Claude'a soru sor."""
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _resolve_path(self, path):
        """Turkce karakterli yollari duzelt."""
        if os.path.exists(path):
            return path
        # Masaustu yolunu dinamik coz
        home = os.path.expanduser("~")
        for desktop_name in ["Masaüstü", "Masa\u00fcst\u00fc", "Desktop"]:
            desktop = os.path.join(home, desktop_name)
            if os.path.isdir(desktop):
                # Path icindeki olasi desktop referanslarini degistir
                for pattern in ["Masaüstü", "Masa├╝st├╝", "Masaüstü",
                                "Desktop", "Masa\\u00fcst\\u00fc"]:
                    if pattern in path:
                        fixed = path.replace(pattern, desktop_name)
                        full = os.path.join(home, fixed.split(home)[-1].lstrip(os.sep)) if home in fixed else fixed
                        if os.path.exists(full):
                            return full
                # Dosya adini al ve desktop uzerinden bul
                parts = path.replace("\\", "/").split("/")
                # TEKNOFEST_GUNCEL'den itibaren yolu yeniden olustur
                for i, part in enumerate(parts):
                    if part == "TEKNOFEST_GUNCEL":
                        remainder = os.path.join(*parts[i:])
                        candidate = os.path.join(desktop, remainder)
                        if os.path.exists(candidate):
                            return candidate
                        break
        return path

    def analyze_training(self, results_csv_path):
        """
        Ultralytics results.csv dosyasini analiz et.
        Egitim egrilerini yorumla ve oneriler sun.
        """
        results_csv_path = self._resolve_path(results_csv_path)
        if not os.path.exists(results_csv_path):
            return f"Dosya bulunamadi: {results_csv_path}"

        # CSV'yi oku
        rows = []
        with open(results_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k.strip(): v.strip() for k, v in row.items()})

        if not rows:
            return "CSV dosyasi bos."

        # Ilk, orta ve son epoch verilerini al
        n = len(rows)
        sample_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        samples = [rows[i] for i in sample_indices if i < n]

        # Son epoch metrikleri
        last = rows[-1]

        prompt = f"""Asagidaki YOLO model egitim verilerini analiz et:

EGITIM OZETI:
- Toplam epoch: {n}
- Son epoch metrikleri: {json.dumps(last, ensure_ascii=False)}

EGITIM ILERLEMESI (secilmis epoch'lar):
{json.dumps(samples, ensure_ascii=False, indent=2)}

Analiz et:
1. Loss degerleri dusuyor mu, platoya mi girdi?
2. mAP yeterli mi, iyilestirme potansiyeli var mi?
3. Overfitting belirtisi var mi (train vs val loss farki)?
4. Learning rate ayari uygun mu?
5. Daha fazla epoch egitim faydali olur mu?"""

        return self._ask(prompt, max_tokens=2500)

    def analyze_dataset(self, dataset_path, class_names):
        """
        Dataset yapisini analiz et:
        sinif dagilimi, gorsel boyutlari, veri miktari.
        """
        train_labels = os.path.join(dataset_path, "train", "labels")
        valid_labels = os.path.join(dataset_path, "valid", "labels")
        train_images = os.path.join(dataset_path, "train", "images")

        # Sinif dagilimini hesapla
        class_counts = {name: 0 for name in class_names.values()}
        class_id_map = {v: k for k, v in class_names.items()}
        total_objects = 0
        empty_labels = 0
        images_per_class = {name: set() for name in class_names.values()}

        if os.path.exists(train_labels):
            for label_file in os.listdir(train_labels):
                if not label_file.endswith('.txt'):
                    continue
                fpath = os.path.join(train_labels, label_file)
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    empty_labels += 1
                    continue
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cls_id = int(parts[0])
                            cls_name = class_names.get(cls_id, f"unknown_{cls_id}")
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                            images_per_class.get(cls_name, set()).add(label_file)
                            total_objects += 1
                        except (ValueError, IndexError):
                            continue

        # Train/valid sayilari
        n_train = len(os.listdir(train_images)) if os.path.exists(train_images) else 0
        n_valid_imgs = os.path.join(dataset_path, "valid", "images")
        n_valid = len(os.listdir(n_valid_imgs)) if os.path.exists(n_valid_imgs) else 0

        # Gorsel boyutlari (ilk 20 gorselden ornekle)
        # numpy ile oku - cv2 Turkce yollarda encoding sorunu yasayabiliyor
        from PIL import Image
        sizes = []
        if os.path.exists(train_images):
            sample_imgs = [f for f in os.listdir(train_images)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:20]
            for img_name in sample_imgs:
                try:
                    with Image.open(os.path.join(train_images, img_name)) as img:
                        sizes.append(f"{img.width}x{img.height}")
                except Exception:
                    pass

        unique_sizes = list(set(sizes))

        prompt = f"""Asagidaki YOLO nesne tespit dataset'ini analiz et:

DATASET BILGILERI:
- Train gorselleri: {n_train}
- Validation gorselleri: {n_valid}
- Train/Val orani: {n_train/(n_valid+1):.1f}
- Bos label dosyasi (negatif ornek): {empty_labels}
- Toplam obje (annotation): {total_objects}

SINIF DAGILIMI:
{json.dumps(class_counts, ensure_ascii=False, indent=2)}

SINIF BASINA GORSEL SAYISI:
{json.dumps({k: len(v) for k, v in images_per_class.items()}, ensure_ascii=False, indent=2)}

GORSEL BOYUTLARI (ornek): {unique_sizes[:5]}

Analiz et:
1. Sinif dengesi nasil? Hangi siniflar az/cok?
2. Toplam veri miktari yeterli mi?
3. Train/Val orani uygun mu?
4. Augmentation onerileri (hangi teknikler, neden)?
5. Negatif ornek sayisi yeterli mi?
6. Veri toplama onerileri (hangi sinifa, ne tur gorseller)?"""

        return self._ask(prompt, max_tokens=2500)

    def analyze_confusion(self, confusion_matrix, class_names):
        """
        Confusion matrix analizi.
        Hangi siniflar karisiyor, neden, ne yapilmali.
        """
        names = list(class_names.values())

        # Matrix'i okunabilir formata cevir
        matrix_str = "Gercek \\ Tahmin | " + " | ".join(names) + "\n"
        matrix_str += "-" * 60 + "\n"
        for i, row in enumerate(confusion_matrix):
            row_name = names[i] if i < len(names) else f"cls_{i}"
            matrix_str += f"{row_name:12} | "
            matrix_str += " | ".join(f"{v:5.0f}" for v in row) + "\n"

        prompt = f"""Asagidaki YOLO nesne tespit confusion matrix'ini analiz et:

SINIFLAR: {names}

CONFUSION MATRIX:
{matrix_str}

Analiz et:
1. Hangi siniflar birbiriyle en cok karisiyor?
2. En dusuk ve en yuksek dogruluk hangi siniflarda?
3. False positive ve false negative oranlarini yorumla
4. Karisiklik icin somut cozum onerileri
5. Augmentation veya veri toplama onerileri"""

        return self._ask(prompt, max_tokens=2000)

    def analyze_visuals(self, vis_report):
        """
        Gorsellestirme sonuclarini yorumla.
        Feature maps, Grad-CAM, t-SNE bulgularini analiz et.
        """
        prompt = f"""Asagidaki YOLO model gorsellestirme bulgularini analiz et:

{vis_report}

Bu bulgulara dayanarak:
1. Model dogru ozellikleri ogrenmis mi?
2. Grad-CAM sonuclari modelin dogru bolgelere odaklandigini gosteriyor mu?
3. t-SNE'de siniflar net ayrisiyorsa/ayrismiyorsa ne anlama gelir?
4. Iyilestirme icin somut oneriler"""

        return self._ask(prompt, max_tokens=2000)

    def ask(self, question, context=None):
        """
        Serbest soru sor.
        Model egitimi, dataset, augmentation vb. konularda.
        """
        prompt = question
        if context:
            prompt = f"Bagalam:\n{context}\n\nSoru: {question}"

        return self._ask(prompt, max_tokens=2000)


def run_coach(config_module):
    """Interaktif AI Coach oturumu baslat."""

    from config import (MODEL_PATH, DATASET_PATH, CLASS_NAMES, OUTPUT_DIR)

    print("=" * 58)
    print("  YOLO AI EGITIM KOCU")
    print("=" * 58)
    print("  Claude API ile model egitim analizi")
    print("=" * 58)

    try:
        coach = AICoach()
    except ValueError as e:
        print(f"\nHATA: {e}")
        return

    while True:
        print("\nNe yapmak istiyorsun?")
        print("  1. Dataset analizi (sinif dagilimi, denge, oneriler)")
        print("  2. Egitim sonuclari analizi (results.csv)")
        print("  3. Serbest soru sor")
        print("  4. Cikis")

        choice = input("\nSecimin (1-4): ").strip()

        if choice == "1":
            print("\nDataset analiz ediliyor...\n")
            result = coach.analyze_dataset(DATASET_PATH, CLASS_NAMES)
            print(result)

        elif choice == "2":
            csv_path = input(
                "results.csv yolu (veya Enter ile varsayilan): "
            ).strip()
            if not csv_path:
                # Otomatik bul: TEKNOFEST_GUNCEL/models altinda ara
                home = os.path.expanduser("~")
                for dname in ["Masaüstü", "Masa\u00fcst\u00fc", "Desktop"]:
                    models_dir = os.path.join(home, dname, "TEKNOFEST_GUNCEL", "models")
                    if os.path.isdir(models_dir):
                        for root, dirs, files in os.walk(models_dir):
                            if "results.csv" in files:
                                csv_path = os.path.join(root, "results.csv")
                                print(f"  Bulundu: {csv_path}")
                                break
                    if csv_path:
                        break
                if not csv_path:
                    # Ultralytics varsayilan yolu dene
                    possible = [
                        os.path.join(DATASET_PATH, "..", "runs", "detect",
                                     "train", "results.csv"),
                        os.path.join(os.path.expanduser("~"), "runs", "detect",
                                     "train", "results.csv"),
                    ]
                    for p in possible:
                        if os.path.exists(p):
                            csv_path = p
                            break
                if not csv_path:
                    print("results.csv bulunamadi. Yolu manuel gir.")
                    continue
            print(f"\nAnaliz ediliyor: {csv_path}\n")
            result = coach.analyze_training(csv_path)
            print(result)

        elif choice == "3":
            question = input("\nSorun: ").strip()
            if question:
                print("\nDusunuyor...\n")
                context = (
                    f"Model: YOLO (best.pt)\n"
                    f"Siniflar: {list(CLASS_NAMES.values())}\n"
                    f"Dataset: {DATASET_PATH}\n"
                    f"Gorev: Nesne tespiti (object detection)"
                )
                result = coach.ask(question, context)
                print(result)

        elif choice == "4":
            print("Cikis.")
            break

        else:
            print("Gecersiz secim.")
