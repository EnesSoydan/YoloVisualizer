"""
Data Analyzer — Yapisal Veri Analizi
======================================
expert_system.py'den tasinan dataset ve egitim metrikleri analiz mantigi.
LLM'e beslenecek yapisal dict'ler uretir.
"""

import os
import csv
import numpy as np


# ============================================================
#  ESIK DEGERLERI (expert_system.py'den korundu)
# ============================================================

MIN_IMAGES_PER_CLASS = 300
GOOD_IMAGES_PER_CLASS = 1000
IDEAL_IMAGES_PER_CLASS = 3000

IMBALANCE_WARNING = 2.0
IMBALANCE_CRITICAL = 5.0

IDEAL_VAL_RATIO_MIN = 0.10
IDEAL_VAL_RATIO_MAX = 0.25

OVERFIT_LOSS_DIVERGENCE = 0.15
OVERFIT_TREND_EPOCHS = 5

MAP50_POOR = 0.50
MAP50_OKAY = 0.70
MAP50_GOOD = 0.85
MAP50_EXCELLENT = 0.92

MAP5095_POOR = 0.30
MAP5095_OKAY = 0.50
MAP5095_GOOD = 0.65


def _resolve_path(path):
    """Turkce karakterli yollari duzelt."""
    if os.path.exists(path):
        return path
    home = os.path.expanduser("~")
    for desktop_name in ["Masaüstü", "Masa\u00fcst\u00fc", "Desktop"]:
        desktop = os.path.join(home, desktop_name)
        if os.path.isdir(desktop):
            for pattern in ["Masaüstü", "Masa├╝st├╝", "Masaüstü",
                            "Desktop", "Masa\\u00fcst\\u00fc"]:
                if pattern in path:
                    fixed = path.replace(pattern, desktop_name)
                    full = os.path.join(
                        home, fixed.split(home)[-1].lstrip(os.sep)
                    ) if home in fixed else fixed
                    if os.path.exists(full):
                        return full
            parts = path.replace("\\", "/").split("/")
            for i, part in enumerate(parts):
                if part == "TEKNOFEST_GUNCEL":
                    remainder = os.path.join(*parts[i:])
                    candidate = os.path.join(desktop, remainder)
                    if os.path.exists(candidate):
                        return candidate
                    break
    return path


class DataAnalyzer:
    """Dataset, egitim metrikleri ve model analizi."""

    # ----- DATASET ANALIZI -----

    def analyze_dataset(self, dataset_path, class_names):
        """Dataset istatistiklerini topla ve yapisal dict dondur."""
        dataset_path = _resolve_path(dataset_path)
        stats = self._collect_dataset_stats(dataset_path, class_names)
        stats["issues"] = self._check_dataset_issues(stats)
        return stats

    def _collect_dataset_stats(self, dataset_path, class_names):
        """Dataset istatistiklerini topla."""
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
                            w, h = float(parts[3]), float(parts[4])
                            bbox_sizes.append(w * h)
                        except (ValueError, IndexError):
                            continue

        n_train = len(os.listdir(train_images)) if os.path.exists(train_images) else 0
        n_valid = len(os.listdir(valid_images)) if os.path.exists(valid_images) else 0

        small_objects = sum(1 for s in bbox_sizes if s < 0.01)
        medium_objects = sum(1 for s in bbox_sizes if 0.01 <= s < 0.1)
        large_objects = sum(1 for s in bbox_sizes if s >= 0.1)

        total = n_train + n_valid
        val_ratio = f"%{(n_valid / total * 100):.1f}" if total > 0 else "N/A"

        return {
            "n_train": n_train,
            "n_valid": n_valid,
            "val_ratio": val_ratio,
            "class_counts": class_counts,
            "images_per_class": {k: len(v) for k, v in images_per_class.items()},
            "total_objects": total_objects,
            "empty_labels": empty_labels,
            "small_objects": small_objects,
            "medium_objects": medium_objects,
            "large_objects": large_objects,
            "avg_bbox_size": float(np.mean(bbox_sizes)) if bbox_sizes else 0,
        }

    def _check_dataset_issues(self, stats):
        """Dataset sorunlarini tespit et."""
        issues = []
        counts = list(stats["class_counts"].values())
        if counts:
            max_c, min_c = max(counts), max(min(counts), 1)
            ratio = max_c / min_c
            if ratio >= IMBALANCE_CRITICAL:
                issues.append(f"KRITIK: Ciddi sinif dengesizligi (oran: {ratio:.1f}x)")
            elif ratio >= IMBALANCE_WARNING:
                issues.append(f"UYARI: Sinif dengesizligi (oran: {ratio:.1f}x)")

        for cls_name, count in stats["class_counts"].items():
            if count < MIN_IMAGES_PER_CLASS:
                issues.append(f"KRITIK: {cls_name} sadece {count} gorsel (min {MIN_IMAGES_PER_CLASS})")

        if stats["empty_labels"] == 0:
            issues.append("UYARI: Negatif ornek (arka plan gorseli) yok")

        if stats["total_objects"] < 5000:
            issues.append(f"UYARI: Toplam annotation sayisi dusuk ({stats['total_objects']})")

        return issues

    # ----- EGITIM METRIKLERI ANALIZI -----

    def analyze_training(self, csv_path):
        """results.csv'yi analiz et."""
        csv_path = _resolve_path(csv_path)
        if not os.path.exists(csv_path):
            return None

        metrics = self._parse_csv(csv_path)
        if not metrics:
            return None

        n_epochs = len(metrics.get("epoch", []))
        last = {k: v[-1] for k, v in metrics.items() if v}

        # Ilerleme ozeti
        progress = {}
        sample_indices = [0, n_epochs // 4, n_epochs // 2, 3 * n_epochs // 4, n_epochs - 1]
        for key in ["train_box_loss", "val_box_loss", "mAP50", "mAP50-95", "precision", "recall"]:
            if key in metrics and metrics[key]:
                progress[key] = [
                    round(metrics[key][i], 4)
                    for i in sample_indices if i < len(metrics[key])
                ]

        issues = self._check_training_issues(metrics, n_epochs)

        return {
            "n_epochs": n_epochs,
            "last_metrics": last,
            "progress": progress,
            "issues": issues,
            "raw_metrics": metrics,
        }

    def _parse_csv(self, csv_path):
        """results.csv'yi oku ve normalize et."""
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
            normalized[k] = v

        return normalized

    def _check_training_issues(self, metrics, n_epochs):
        """Egitim sorunlarini tespit et."""
        issues = []

        # Overfitting kontrolu
        if 'val_box_loss' in metrics and 'train_box_loss' in metrics:
            val_losses = metrics['val_box_loss']
            train_losses = metrics['train_box_loss']
            if len(val_losses) >= OVERFIT_TREND_EPOCHS:
                recent_val = val_losses[-OVERFIT_TREND_EPOCHS:]
                recent_train = train_losses[-OVERFIT_TREND_EPOCHS:]
                val_trend = recent_val[-1] - recent_val[0]
                train_trend = recent_train[-1] - recent_train[0]
                if val_trend > 0 and train_trend < 0:
                    issues.append("KRITIK: Overfitting — val loss artarken train loss dusuyor")
                gap = val_losses[-1] - train_losses[-1]
                if gap > OVERFIT_LOSS_DIVERGENCE:
                    issues.append(f"UYARI: Train/Val loss arasi buyuk ({gap:.4f})")

        # mAP kontrolu
        for key in metrics:
            if 'map50' in key.lower() and '95' not in key.lower() and metrics[key]:
                final_map50 = metrics[key][-1]
                if final_map50 < MAP50_POOR:
                    issues.append(f"KRITIK: mAP@50 cok dusuk ({final_map50:.3f})")
                elif final_map50 < MAP50_OKAY:
                    issues.append(f"UYARI: mAP@50 orta seviye ({final_map50:.3f})")
                break

        # Precision/Recall dengesi
        prec = metrics.get('precision', [])
        rec = metrics.get('recall', [])
        if prec and rec:
            p, r = prec[-1], rec[-1]
            if p > 0 and r > 0:
                ratio = p / r
                if ratio > 1.5:
                    issues.append(f"UYARI: Precision >> Recall (P={p:.3f}, R={r:.3f}) — cok kaciriyor")
                elif ratio < 0.67:
                    issues.append(f"UYARI: Recall >> Precision (P={p:.3f}, R={r:.3f}) — false positive fazla")

        if n_epochs < 50:
            issues.append(f"BILGI: Az epoch ({n_epochs}) — YOLO icin 100-300 epoch onerilir")

        return issues

    # ----- MODEL ANALIZI -----

    def analyze_model(self, model_path):
        """YOLO model mimarisini incele."""
        model_path = _resolve_path(model_path)
        if not os.path.exists(model_path):
            return {"error": f"Model bulunamadi: {model_path}"}

        try:
            from ultralytics import YOLO
            model = YOLO(model_path)

            # Temel bilgiler
            info = {
                "model_path": os.path.basename(model_path),
                "task": getattr(model, 'task', 'detect'),
            }

            # Model parametreleri
            if hasattr(model, 'model'):
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.model.parameters() if p.requires_grad
                )
                info["total_params"] = f"{total_params:,}"
                info["trainable_params"] = f"{trainable_params:,}"
                info["model_size_mb"] = f"{total_params * 4 / 1024 / 1024:.1f}"

            # Sinif bilgileri
            if hasattr(model, 'names'):
                info["classes"] = model.names
                info["num_classes"] = len(model.names)

            # Katman sayisi
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                info["num_layers"] = len(list(model.model.model.children()))

            return info

        except Exception as e:
            return {"error": str(e)}

    # ----- YARDIMCI: CSV YOLU OTOMATIK BUL -----

    @staticmethod
    def find_results_csv():
        """results.csv'yi otomatik bul."""
        home = os.path.expanduser("~")
        for dname in ["Masaüstü", "Masa\u00fcst\u00fc", "Desktop"]:
            models_dir = os.path.join(home, dname, "TEKNOFEST_GUNCEL", "models")
            if os.path.isdir(models_dir):
                for root, dirs, files in os.walk(models_dir):
                    if "results.csv" in files:
                        return os.path.join(root, "results.csv")
        return None
