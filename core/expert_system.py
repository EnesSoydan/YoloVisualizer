"""
Kural Tabanli Uzman Sistem (Expert System)
============================================
YOLO model egitimini analiz eden, API gerektirmeyen yerel uzman sistem.
Kurallara dayali deterministik analiz:
  - Dataset dengesi ve yeterliligi
  - Egitim metriklerinden overfitting/underfitting tespiti
  - Confusion matrix yorumlama
  - Gorsellestirme bulgularini degerlendirme
  - Augmentation ve hiperparametre onerileri

API maliyeti: $0 — tamamen yerel calisir.
"""

import os
import csv
import math
import numpy as np
from pathlib import Path


# ============================================================
#  BILGI TABANI: YOLO egitimi icin kurallar ve esik degerleri
# ============================================================

# Sinif basina minimum gorsel sayisi esikleri
MIN_IMAGES_PER_CLASS = 300       # minimum kabul edilebilir
GOOD_IMAGES_PER_CLASS = 1000     # iyi seviye
IDEAL_IMAGES_PER_CLASS = 3000    # ideal seviye

# Sinif dengesizlik esikleri
IMBALANCE_WARNING = 2.0    # max/min > 2 → uyari
IMBALANCE_CRITICAL = 5.0   # max/min > 5 → kritik

# Train/Val split
IDEAL_VAL_RATIO_MIN = 0.10   # %10
IDEAL_VAL_RATIO_MAX = 0.25   # %25

# Overfitting tespiti
OVERFIT_LOSS_DIVERGENCE = 0.15  # val_loss - train_loss farki
OVERFIT_TREND_EPOCHS = 5        # son N epoch'ta trend

# mAP esikleri
MAP50_POOR = 0.50
MAP50_OKAY = 0.70
MAP50_GOOD = 0.85
MAP50_EXCELLENT = 0.92

MAP5095_POOR = 0.30
MAP5095_OKAY = 0.50
MAP5095_GOOD = 0.65


def _severity(level):
    """Onem seviyesi etiketi."""
    return {"critical": "[!] KRITIK", "warning": "[*] UYARI",
            "info": "[i] BILGI", "good": "[+] OLUMLU"}[level]


# ============================================================
#  DATASET ANALIZI
# ============================================================

class DatasetAnalyzer:
    """Dataset istatistiklerini topla ve kural tabanli analiz yap."""

    def analyze(self, dataset_path, class_names):
        stats = self._collect_stats(dataset_path, class_names)
        issues = []
        recommendations = []

        # --- Kural 1: Sinif dengesizligi ---
        counts = list(stats['class_counts'].values())
        if counts:
            max_c, min_c = max(counts), max(min(counts), 1)
            ratio = max_c / min_c

            if ratio >= IMBALANCE_CRITICAL:
                max_cls = max(stats['class_counts'], key=stats['class_counts'].get)
                min_cls = min(stats['class_counts'], key=stats['class_counts'].get)
                issues.append((
                    "critical",
                    f"Ciddi sinif dengesizligi: {max_cls} ({max_c}) vs "
                    f"{min_cls} ({min_c}) — oran {ratio:.1f}x",
                    f"Model {max_cls} sinifina asiri odaklanacak, "
                    f"{min_cls} sinifini kaciracak."
                ))
                recommendations.append(
                    f"- {min_cls} sinifi icin ek veri topla "
                    f"(en az {max_c // 2} gorsele cikar)\n"
                    f"- Weighted Loss kullan "
                    f"(az olan sinifa yuksek agirlik ver)\n"
                    f"- {min_cls} sinifina ozel augmentation uygula "
                    f"(flip, rotation, scale)"
                )
            elif ratio >= IMBALANCE_WARNING:
                issues.append((
                    "warning",
                    f"Orta seviye sinif dengesizligi (oran: {ratio:.1f}x)",
                    "Performans farki olusabilir."
                ))
                recommendations.append(
                    "- Oversampling (az olan siniftan daha cok ornekleme) "
                    "veya class weights kullan\n"
                    "- Focal Loss dene (zor orneklere odaklanir)"
                )
            else:
                issues.append(("good", "Sinif dagilimi dengeli", ""))

        # --- Kural 2: Sinif basina yeterlilik ---
        for cls_name, count in stats['class_counts'].items():
            if count < MIN_IMAGES_PER_CLASS:
                issues.append((
                    "critical",
                    f"{cls_name}: sadece {count} gorsel "
                    f"(minimum {MIN_IMAGES_PER_CLASS} olmali)",
                    "Bu sinif icin model iyi ogrenemeyecek."
                ))
                recommendations.append(
                    f"- {cls_name} icin en az "
                    f"{MIN_IMAGES_PER_CLASS - count} ek gorsel topla\n"
                    f"- Gecici cozum: augmentation ile cogalt "
                    f"(mosaic, mixup, copy-paste)"
                )
            elif count < GOOD_IMAGES_PER_CLASS:
                issues.append((
                    "warning",
                    f"{cls_name}: {count} gorsel "
                    f"(ideal: {GOOD_IMAGES_PER_CLASS}+)",
                    ""
                ))

        # --- Kural 3: Train/Val split ---
        total = stats['n_train'] + stats['n_valid']
        if total > 0:
            val_ratio = stats['n_valid'] / total
            if val_ratio < IDEAL_VAL_RATIO_MIN:
                issues.append((
                    "warning",
                    f"Validation orani cok dusuk: %{val_ratio*100:.1f} "
                    f"(ideal: %{IDEAL_VAL_RATIO_MIN*100:.0f}-"
                    f"%{IDEAL_VAL_RATIO_MAX*100:.0f})",
                    "Model performansi dogru olculemiyor olabilir."
                ))
                ideal_val = int(total * 0.15)
                recommendations.append(
                    f"- Dataset'i yeniden bol: "
                    f"~{total - ideal_val} train / ~{ideal_val} val\n"
                    f"- Stratified split kullan "
                    f"(her siniftan esit oran)"
                )
            elif val_ratio > IDEAL_VAL_RATIO_MAX:
                issues.append((
                    "info",
                    f"Validation orani yuksek: %{val_ratio*100:.1f}",
                    "Train seti icin daha fazla veri ayirabilirsin."
                ))
            else:
                issues.append((
                    "good",
                    f"Train/Val orani uygun: "
                    f"%{(1-val_ratio)*100:.0f} / %{val_ratio*100:.0f}",
                    ""
                ))

        # --- Kural 4: Negatif ornekler ---
        if stats['empty_labels'] == 0:
            issues.append((
                "warning",
                "Negatif ornek (arka plan gorseli) yok",
                "Model her goruntude obje bulmaya calisacak "
                "(false positive riski)."
            ))
            neg_count = max(100, int(stats['n_train'] * 0.08))
            recommendations.append(
                f"- {neg_count} adet arka plan gorseli ekle "
                f"(obje icermeyen, bos label dosyali)\n"
                f"- Gok, arazi, sehir manzarasi gibi "
                f"gorseller kullanilabilir"
            )
        else:
            neg_ratio = stats['empty_labels'] / max(stats['n_train'], 1)
            if neg_ratio < 0.05:
                issues.append((
                    "info",
                    f"Negatif ornek az: {stats['empty_labels']} "
                    f"(%{neg_ratio*100:.1f})",
                    "Ideal: %5-15 arasi"
                ))

        # --- Kural 5: Toplam veri ---
        total_objects = stats['total_objects']
        if total_objects < 5000:
            issues.append((
                "warning",
                f"Toplam annotation sayisi dusuk: {total_objects:,}",
                "Karmasik sahneler icin yetersiz olabilir."
            ))
        elif total_objects > 50000:
            issues.append((
                "good",
                f"Zengin dataset: {total_objects:,} annotation", ""
            ))

        # --- Rapor ---
        return self._format_report(stats, issues, recommendations)

    def _collect_stats(self, dataset_path, class_names):
        train_labels = os.path.join(dataset_path, "train", "labels")
        train_images = os.path.join(dataset_path, "train", "images")
        valid_images = os.path.join(dataset_path, "valid", "images")

        class_counts = {name: 0 for name in class_names.values()}
        images_per_class = {name: set() for name in class_names.values()}
        total_objects = 0
        empty_labels = 0
        bbox_sizes = []

        if os.path.exists(train_labels):
            for label_file in os.listdir(train_labels):
                if not label_file.endswith('.txt'):
                    continue
                fpath = os.path.join(train_labels, label_file)
                with open(fpath, 'r') as f:
                    lines = f.readlines()
                if not lines or all(not l.strip() for l in lines):
                    empty_labels += 1
                    continue
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(parts[0])
                            cls_name = class_names.get(cls_id, f"unknown_{cls_id}")
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                            if cls_name in images_per_class:
                                images_per_class[cls_name].add(label_file)
                            total_objects += 1
                            # bbox boyutu (w*h orani)
                            w, h = float(parts[3]), float(parts[4])
                            bbox_sizes.append(w * h)
                        except (ValueError, IndexError):
                            continue

        n_train = len(os.listdir(train_images)) if os.path.exists(train_images) else 0
        n_valid = len(os.listdir(valid_images)) if os.path.exists(valid_images) else 0

        # Bbox boyut analizi
        small_objects = sum(1 for s in bbox_sizes if s < 0.01)  # < %1 alan
        medium_objects = sum(1 for s in bbox_sizes if 0.01 <= s < 0.1)
        large_objects = sum(1 for s in bbox_sizes if s >= 0.1)

        return {
            'class_counts': class_counts,
            'images_per_class': {k: len(v) for k, v in images_per_class.items()},
            'total_objects': total_objects,
            'empty_labels': empty_labels,
            'n_train': n_train,
            'n_valid': n_valid,
            'small_objects': small_objects,
            'medium_objects': medium_objects,
            'large_objects': large_objects,
            'total_bbox': len(bbox_sizes),
            'avg_bbox_size': np.mean(bbox_sizes) if bbox_sizes else 0,
        }

    def _format_report(self, stats, issues, recommendations):
        lines = []
        lines.append("=" * 55)
        lines.append("  DATASET ANALIZI (Yerel Uzman Sistem)")
        lines.append("=" * 55)

        # Ozet
        lines.append("\n--- MEVCUT DURUM ---")
        lines.append(f"  Train gorselleri : {stats['n_train']:,}")
        lines.append(f"  Valid gorselleri : {stats['n_valid']:,}")
        lines.append(f"  Toplam annotation: {stats['total_objects']:,}")
        lines.append(f"  Negatif ornekler : {stats['empty_labels']}")
        lines.append(f"\n  Sinif Dagilimi:")
        for cls, count in stats['class_counts'].items():
            bar_len = min(30, int(count / max(max(stats['class_counts'].values()), 1) * 30))
            bar = "#" * bar_len
            img_count = stats['images_per_class'].get(cls, 0)
            lines.append(f"    {cls:15} {count:6,} obje | {img_count:5,} gorsel  {bar}")

        if stats['total_bbox'] > 0:
            lines.append(f"\n  Obje Boyut Dagilimi:")
            total = stats['total_bbox']
            lines.append(f"    Kucuk  (<%1 alan) : {stats['small_objects']:,} "
                        f"(%{stats['small_objects']/total*100:.0f})")
            lines.append(f"    Orta   (1-10%)    : {stats['medium_objects']:,} "
                        f"(%{stats['medium_objects']/total*100:.0f})")
            lines.append(f"    Buyuk  (>10%)     : {stats['large_objects']:,} "
                        f"(%{stats['large_objects']/total*100:.0f})")

        # Sorunlar
        lines.append("\n--- TESPIT EDILEN SORUNLAR ---")
        for severity, msg, detail in issues:
            lines.append(f"  {_severity(severity)}: {msg}")
            if detail:
                lines.append(f"    > {detail}")

        # Oneriler
        if recommendations:
            lines.append("\n--- ONERILER ---")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"\n  {i}.")
                for line in rec.split('\n'):
                    lines.append(f"    {line}")

        lines.append("")
        return '\n'.join(lines)


# ============================================================
#  EGITIM METRIKLERI ANALIZI
# ============================================================

class TrainingAnalyzer:
    """results.csv'den egitim surecini analiz et."""

    def analyze(self, csv_path):
        metrics = self._parse_csv(csv_path)
        if not metrics:
            return "CSV dosyasi okunamadi veya bos."

        issues = []
        recommendations = []
        n_epochs = len(metrics['epoch'])

        # Son epoch degerleri
        last = {k: v[-1] for k, v in metrics.items() if v}

        # --- Kural 1: Overfitting tespiti ---
        if 'val_box_loss' in metrics and 'train_box_loss' in metrics:
            val_losses = metrics['val_box_loss']
            train_losses = metrics['train_box_loss']

            if len(val_losses) >= OVERFIT_TREND_EPOCHS:
                # Son N epoch'ta val loss artiyor mu?
                recent_val = val_losses[-OVERFIT_TREND_EPOCHS:]
                recent_train = train_losses[-OVERFIT_TREND_EPOCHS:]
                val_trend = recent_val[-1] - recent_val[0]
                train_trend = recent_train[-1] - recent_train[0]

                if val_trend > 0 and train_trend < 0:
                    issues.append((
                        "critical",
                        f"Overfitting tespit edildi!",
                        f"Son {OVERFIT_TREND_EPOCHS} epoch'ta: "
                        f"train loss {train_trend:+.4f}, "
                        f"val loss {val_trend:+.4f} "
                        f"(train dusuyor ama val artiyor)"
                    ))
                    recommendations.append(
                        "- Early stopping uygula "
                        f"(val loss'un en dusuk oldugu epoch'u kullan)\n"
                        "- Regularization ekle: dropout, weight decay artir\n"
                        "- Augmentation guclendir "
                        "(mosaic, mixup, random erase)\n"
                        "- Daha fazla veri topla (en etkili cozum)"
                    )
                elif val_trend > 0 and train_trend > 0:
                    issues.append((
                        "warning",
                        "Her iki loss da artiyor (son epoch'larda)",
                        "Learning rate cok yuksek olabilir."
                    ))
                    recommendations.append(
                        "- Learning rate'i azalt (x0.5 veya x0.1)\n"
                        "- Cosine annealing scheduler dene"
                    )
                elif val_trend <= 0 and train_trend <= 0:
                    issues.append((
                        "good",
                        "Overfitting belirtisi yok — loss'lar dusuyor",
                        ""
                    ))

                # Train-val gap kontrolu
                gap = val_losses[-1] - train_losses[-1]
                if gap > OVERFIT_LOSS_DIVERGENCE:
                    issues.append((
                        "warning",
                        f"Train/Val loss arasi buyuk: {gap:.4f}",
                        "Hafif overfitting baslamis olabilir."
                    ))

        # --- Kural 2: mAP analizi ---
        map50_key = None
        map5095_key = None
        for k in metrics:
            if 'map50' in k.lower() and '95' not in k.lower():
                map50_key = k
            if 'map50-95' in k.lower() or 'map50_95' in k.lower():
                map5095_key = k

        if map50_key and metrics[map50_key]:
            final_map50 = metrics[map50_key][-1]
            if final_map50 < MAP50_POOR:
                issues.append((
                    "critical",
                    f"mAP@50 cok dusuk: {final_map50:.3f}",
                    "Model nesne tespit edemiyor denecek seviyede."
                ))
                recommendations.append(
                    "- Veri kalitesini kontrol et "
                    "(yanlis etiketler, eksik annotationlar)\n"
                    "- Daha kucuk model dene (overfitting'i azaltir)\n"
                    "- Pretrained weights ile basla "
                    "(sifirdan egitme)\n"
                    "- Imgsz'yi artir (640 → 1280)"
                )
            elif final_map50 < MAP50_OKAY:
                issues.append((
                    "warning",
                    f"mAP@50 orta seviye: {final_map50:.3f}",
                    "Iyilestirme potansiyeli var."
                ))
                recommendations.append(
                    "- Daha fazla epoch dene (platoya girmediyse)\n"
                    "- Augmentation cesitlendir\n"
                    "- Zor ornekleri artir (hard example mining)"
                )
            elif final_map50 < MAP50_GOOD:
                issues.append((
                    "info",
                    f"mAP@50 iyi seviye: {final_map50:.3f}",
                    "Uretim ortami icin kabul edilebilir."
                ))
            else:
                issues.append((
                    "good",
                    f"mAP@50 cok iyi: {final_map50:.3f}", ""
                ))

            # mAP platoya girmis mi?
            if len(metrics[map50_key]) >= 10:
                recent = metrics[map50_key][-10:]
                improvement = recent[-1] - recent[0]
                if abs(improvement) < 0.005:
                    issues.append((
                        "info",
                        f"mAP son 10 epoch'ta platoya girmis "
                        f"(degisim: {improvement:+.4f})",
                        ""
                    ))
                    recommendations.append(
                        "- Egitimi durdurabilirsin "
                        "(daha fazla epoch fayda vermeyecek)\n"
                        "- Veya learning rate'i 10x dusur "
                        "ve kisa sure daha egit"
                    )

        # --- Kural 3: Precision/Recall dengesi ---
        prec_key = None
        rec_key = None
        for k in metrics:
            if 'precision' in k.lower():
                prec_key = k
            if 'recall' in k.lower():
                rec_key = k

        if prec_key and rec_key and metrics[prec_key] and metrics[rec_key]:
            final_prec = metrics[prec_key][-1]
            final_rec = metrics[rec_key][-1]

            if final_prec > 0 and final_rec > 0:
                pr_ratio = final_prec / final_rec

                if pr_ratio > 1.5:
                    issues.append((
                        "warning",
                        f"Precision ({final_prec:.3f}) >> "
                        f"Recall ({final_rec:.3f})",
                        "Model temkinli — buldugu dogru ama cok kaciriyor."
                    ))
                    recommendations.append(
                        "- Confidence threshold'u dusur\n"
                        "- Daha fazla pozitif ornek ekle\n"
                        "- NMS IoU threshold'u artir"
                    )
                elif pr_ratio < 0.67:
                    issues.append((
                        "warning",
                        f"Recall ({final_rec:.3f}) >> "
                        f"Precision ({final_prec:.3f})",
                        "Model cok sey buluyor ama cogu yanlis (false positive)."
                    ))
                    recommendations.append(
                        "- Confidence threshold'u artir\n"
                        "- Negatif ornek ekle (arka plan gorselleri)\n"
                        "- Hard negative mining uygula"
                    )
                else:
                    issues.append((
                        "good",
                        f"Precision/Recall dengeli: "
                        f"P={final_prec:.3f}, R={final_rec:.3f}",
                        ""
                    ))

        # --- Kural 4: Loss convergence ---
        if 'train_box_loss' in metrics and len(metrics['train_box_loss']) >= 5:
            losses = metrics['train_box_loss']
            first_quarter = np.mean(losses[:len(losses)//4]) if losses else 0
            last_quarter = np.mean(losses[-len(losses)//4:]) if losses else 0
            total_drop = (first_quarter - last_quarter) / (first_quarter + 1e-8)

            if total_drop < 0.1:
                issues.append((
                    "warning",
                    f"Loss cok az dustu (toplam %{total_drop*100:.0f})",
                    "Learning rate cok dusuk veya model ogrenemiyor."
                ))
                recommendations.append(
                    "- Learning rate'i artir\n"
                    "- Model mimarisini degistir "
                    "(daha buyuk model dene)\n"
                    "- Verinin dogru etiketlendiginden emin ol"
                )

        # --- Kural 5: Epoch sayisi ---
        if n_epochs < 50:
            issues.append((
                "info",
                f"Az epoch ile egitilmis: {n_epochs}",
                "YOLO icin genellikle 100-300 epoch onerilir."
            ))
        elif n_epochs > 500:
            issues.append((
                "info",
                f"Cok fazla epoch: {n_epochs}",
                "Overfitting riski artar. Early stopping kontrol et."
            ))

        # --- Kural 6: Learning rate ---
        lr_key = None
        for k in metrics:
            if 'lr' in k.lower():
                lr_key = k
                break

        if lr_key and metrics[lr_key]:
            final_lr = metrics[lr_key][-1]
            initial_lr = metrics[lr_key][0]
            if final_lr > 0.01:
                issues.append((
                    "warning",
                    f"Learning rate yuksek: {final_lr:.5f}",
                    "Ince ayar (fine-tuning) icin 0.001-0.0001 arasi daha iyi."
                ))

        return self._format_report(metrics, n_epochs, last, issues, recommendations)

    def _parse_csv(self, csv_path):
        """results.csv'yi oku ve metrik dict'e cevir."""
        metrics = {}
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key, val in row.items():
                        key = key.strip()
                        if key not in metrics:
                            metrics[key] = []
                        try:
                            metrics[key].append(float(val.strip()))
                        except (ValueError, AttributeError):
                            pass
        except Exception:
            return None

        # Kolon isimlerini normalize et
        normalized = {}
        for k, v in metrics.items():
            k_lower = k.lower().replace(' ', '_').replace('/', '_')
            if 'epoch' in k_lower:
                normalized['epoch'] = v
            elif 'train' in k_lower and 'box' in k_lower:
                normalized['train_box_loss'] = v
            elif 'train' in k_lower and 'cls' in k_lower:
                normalized['train_cls_loss'] = v
            elif 'train' in k_lower and 'dfl' in k_lower:
                normalized['train_dfl_loss'] = v
            elif 'val' in k_lower and 'box' in k_lower:
                normalized['val_box_loss'] = v
            elif 'val' in k_lower and 'cls' in k_lower:
                normalized['val_cls_loss'] = v
            elif 'val' in k_lower and 'dfl' in k_lower:
                normalized['val_dfl_loss'] = v
            elif 'precision' in k_lower:
                normalized['precision'] = v
            elif 'recall' in k_lower:
                normalized['recall'] = v
            elif 'map50-95' in k_lower or 'map50_95' in k_lower:
                normalized['mAP50-95'] = v
            elif 'map50' in k_lower:
                normalized['mAP50'] = v
            elif 'lr' in k_lower:
                if 'lr' not in normalized:
                    normalized['lr'] = v
            normalized[k] = v  # orijinali de tut

        return normalized

    def _format_report(self, metrics, n_epochs, last, issues, recommendations):
        lines = []
        lines.append("=" * 55)
        lines.append("  EGITIM ANALIZI (Yerel Uzman Sistem)")
        lines.append("=" * 55)

        lines.append(f"\n--- MEVCUT DURUM ---")
        lines.append(f"  Toplam epoch: {n_epochs}")

        # Son epoch metriklerini goster
        display_keys = [
            ('train_box_loss', 'Train Box Loss'),
            ('val_box_loss', 'Val Box Loss'),
            ('train_cls_loss', 'Train Cls Loss'),
            ('val_cls_loss', 'Val Cls Loss'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('mAP50', 'mAP@50'),
            ('mAP50-95', 'mAP@50-95'),
            ('lr', 'Learning Rate'),
        ]
        lines.append(f"\n  Son Epoch Metrikleri:")
        for key, label in display_keys:
            if key in metrics and metrics[key]:
                val = metrics[key][-1]
                # Degisim (ilk epoch'a gore)
                first_val = metrics[key][0] if metrics[key] else val
                change = val - first_val
                arrow = "(artti)" if change > 0.001 else ("(dustu)" if change < -0.001 else "(ayni)")
                lines.append(f"    {label:18}: {val:.4f} {arrow} "
                           f"(ilk: {first_val:.4f}, degisim: {change:+.4f})")

        lines.append("\n--- TESPIT EDILEN SORUNLAR ---")
        for severity, msg, detail in issues:
            lines.append(f"  {_severity(severity)}: {msg}")
            if detail:
                lines.append(f"    > {detail}")

        if recommendations:
            lines.append("\n--- ONERILER ---")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"\n  {i}.")
                for line in rec.split('\n'):
                    lines.append(f"    {line}")

        # Egitim ozeti
        lines.append("\n--- EGITIM GRAFIGI (ASCII) ---")
        if 'mAP50' in metrics and metrics['mAP50']:
            lines.append("  mAP@50 ilerlemesi:")
            self._ascii_plot(lines, metrics['mAP50'], width=50)

        if 'train_box_loss' in metrics and 'val_box_loss' in metrics:
            lines.append("\n  Train(T) vs Val(V) Box Loss:")
            self._ascii_dual_plot(lines, metrics['train_box_loss'],
                                  metrics['val_box_loss'], width=50)

        lines.append("")
        return '\n'.join(lines)

    def _ascii_plot(self, lines, values, width=50):
        """Basit ASCII grafik."""
        if not values:
            return
        min_v, max_v = min(values), max(values)
        range_v = max_v - min_v if max_v != min_v else 1

        # Her 'width' noktasindan biri secilir
        step = max(1, len(values) // width)
        sampled = values[::step]

        for i, v in enumerate(sampled):
            bar_len = int((v - min_v) / range_v * 40)
            bar = "#" * bar_len
            if i == 0:
                lines.append(f"  E{0:3d} |{bar} {v:.3f}")
            elif i == len(sampled) - 1:
                lines.append(f"  E{len(values):3d} |{bar} {v:.3f}")
            elif i % max(1, len(sampled) // 5) == 0:
                epoch = i * step
                lines.append(f"  E{epoch:3d} |{bar}")

    def _ascii_dual_plot(self, lines, train_vals, val_vals, width=50):
        """Train vs Val karsilastirma grafigi."""
        if not train_vals or not val_vals:
            return
        all_vals = train_vals + val_vals
        min_v, max_v = min(all_vals), max(all_vals)
        range_v = max_v - min_v if max_v != min_v else 1

        step = max(1, len(train_vals) // 10)
        for i in range(0, len(train_vals), step):
            t = train_vals[i]
            v = val_vals[i] if i < len(val_vals) else 0
            t_bar = int((t - min_v) / range_v * 30)
            v_bar = int((v - min_v) / range_v * 30)
            lines.append(f"  E{i:3d} T|{'█' * t_bar} {t:.3f}")
            lines.append(f"       V|{'░' * v_bar} {v:.3f}")


# ============================================================
#  ANA FONKSIYON: Interaktif yerel coach
# ============================================================

def run_local_coach(config_module):
    """API gerektirmeyen yerel uzman sistem oturumu."""
    from config import (MODEL_PATH, DATASET_PATH, CLASS_NAMES, OUTPUT_DIR)

    print("=" * 55)
    print("  YOLO YEREL UZMAN SISTEM")
    print("  (API gerektirmez — tamamen yerel)")
    print("=" * 55)

    dataset_analyzer = DatasetAnalyzer()
    training_analyzer = TrainingAnalyzer()

    while True:
        print("\nNe yapmak istiyorsun?")
        print("  1. Dataset analizi")
        print("  2. Egitim sonuclari analizi (results.csv)")
        print("  3. Cikis")

        choice = input("\nSecimin (1-3): ").strip()

        if choice == "1":
            print("\nDataset analiz ediliyor...\n")
            result = dataset_analyzer.analyze(DATASET_PATH, CLASS_NAMES)
            print(result)

        elif choice == "2":
            csv_path = input(
                "results.csv yolu (veya Enter ile otomatik bul): "
            ).strip()
            if not csv_path:
                home = os.path.expanduser("~")
                for dname in ["Masaüstü", "Masa\u00fcst\u00fc", "Desktop"]:
                    models_dir = os.path.join(home, dname,
                                              "TEKNOFEST_GUNCEL", "models")
                    if os.path.isdir(models_dir):
                        for root, dirs, files in os.walk(models_dir):
                            if "results.csv" in files:
                                csv_path = os.path.join(root, "results.csv")
                                print(f"  Bulundu: {csv_path}")
                                break
                    if csv_path:
                        break
            if not csv_path:
                print("  results.csv bulunamadi.")
                continue

            # Turkce path fix
            if not os.path.exists(csv_path):
                home = os.path.expanduser("~")
                parts = csv_path.replace("\\", "/").split("/")
                for i, part in enumerate(parts):
                    if part == "TEKNOFEST_GUNCEL":
                        for dname in ["Masaüstü", "Masa\u00fcst\u00fc", "Desktop"]:
                            candidate = os.path.join(home, dname, *parts[i:])
                            if os.path.exists(candidate):
                                csv_path = candidate
                                break
                        break

            if not os.path.exists(csv_path):
                print(f"  Dosya bulunamadi: {csv_path}")
                continue

            print(f"\nAnaliz ediliyor...\n")
            result = training_analyzer.analyze(csv_path)
            print(result)

        elif choice == "3":
            print("Cikis.")
            break

        else:
            print("Gecersiz secim.")
