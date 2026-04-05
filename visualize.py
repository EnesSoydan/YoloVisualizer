"""
YOLO Model Gorsellestiricisi
=============================
YOLO modelinin nasil ogrendigini katman katman gorsellestir.

Kullanim:
    python visualize.py                    # Tum gorsellestirmeler
    python visualize.py --feature-maps     # Sadece feature maps
    python visualize.py --gradcam          # Sadece Grad-CAM
    python visualize.py --filters          # Sadece filtreler
    python visualize.py --tsne             # Sadece t-SNE
    python visualize.py --image YOLO.jpg   # Belirli bir gorsel icin
    python visualize.py --agent            # AI uzman agent (yerel, RAG destekli)
    python visualize.py --heatmap          # Tespit bazli Grad-CAM heatmap
      --model MODEL  --data DATA.yaml     # Ozel model/dataset ile

Ciktilar outputs/ klasorune kaydedilir.
"""

import argparse
import os
import sys
import random
import time


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Model Gorsellestiricisi"
    )
    parser.add_argument("--image", type=str,
                        help="Analiz edilecek gorsel yolu")
    parser.add_argument("--feature-maps", action="store_true",
                        help="Feature map gorsellestiremesi")
    parser.add_argument("--gradcam", action="store_true",
                        help="Grad-CAM isi haritasi")
    parser.add_argument("--filters", action="store_true",
                        help="Konvolusyon filtre gorsellestiremesi")
    parser.add_argument("--tsne", action="store_true",
                        help="t-SNE embedding gorsellestiremesi")
    parser.add_argument("--agent", action="store_true",
                        help="AI uzman agent - yerel, akilli, RAG destekli")
    parser.add_argument("--heatmap", action="store_true",
                        help="Tespit bazli Grad-CAM heatmap raporu")
    parser.add_argument("--model", type=str,
                        help="Ozel model yolu (config.py yerine)")
    parser.add_argument("--data", type=str,
                        help="data.yaml yolu (sinif isimleri + gorsel dizini)")
    args = parser.parse_args()

    # Agent modu
    if args.agent:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config
        from core.agent import run_agent
        run_agent(config)
        return

    run_all = not any([
        args.feature_maps, args.gradcam, args.filters, args.tsne,
        args.heatmap
    ])

    # Config'i import et
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        MODEL_PATH, TRAIN_IMAGES, DATASET_PATH,
        OUTPUT_DIR, CLASS_NAMES, IMGSZ, TSNE_NUM_IMAGES
    )

    # --model / --data override
    if args.data:
        import yaml
        if not os.path.exists(args.data):
            print(f"HATA: data.yaml bulunamadi: {args.data}")
            sys.exit(1)
        with open(args.data, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
        names = data_cfg['names']
        if isinstance(names, dict):
            CLASS_NAMES = {int(k): str(v) for k, v in names.items()}
        else:
            CLASS_NAMES = {i: str(name) for i, name in enumerate(names)}
        data_dir = os.path.dirname(os.path.abspath(args.data))
        img_subdir = data_cfg.get('val', data_cfg.get('train', 'images'))
        TRAIN_IMAGES = os.path.normpath(os.path.join(data_dir, img_subdir))

    if args.model:
        MODEL_PATH = args.model

    # Yollari dogrula
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model bulunamadi: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(TRAIN_IMAGES):
        print(f"HATA: Dataset bulunamadi: {TRAIN_IMAGES}")
        sys.exit(1)

    # Gorsel sec
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"HATA: Gorsel bulunamadi: {image_path}")
            sys.exit(1)
    else:
        images = [
            f for f in os.listdir(TRAIN_IMAGES)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if not images:
            print(f"HATA: {TRAIN_IMAGES} klasorunde gorsel bulunamadi")
            sys.exit(1)
        image_path = os.path.join(TRAIN_IMAGES, random.choice(images))

    print("=" * 58)
    print("  YOLO MODEL GORSELLESTIRICISI")
    print("=" * 58)
    print(f"  Gorsel  : {os.path.basename(image_path)}")
    print(f"  Model   : {os.path.basename(MODEL_PATH)}")
    print(f"  Siniflar: {', '.join(CLASS_NAMES.values())}")
    print(f"  Ciktilar: {OUTPUT_DIR}")
    print("=" * 58)

    t0 = time.time()

    # 1. Feature Maps
    if run_all or args.feature_maps:
        print("\n[1/4] Feature Map Gorsellestiremesi")
        print("  Her katmanin gorseli nasil gordugunu gosterir.")
        print("  Erken: kenarlar -> Orta: dokular -> Derin: objeler")
        from core.feature_maps import visualize_feature_maps
        visualize_feature_maps(MODEL_PATH, image_path, OUTPUT_DIR, IMGSZ)

    # 2. Grad-CAM
    if run_all or args.gradcam:
        print("\n[2/4] Grad-CAM Isi Haritasi")
        print("  Model tespit yaparken nereye baktigini gosterir.")
        from core.gradcam import visualize_gradcam
        visualize_gradcam(
            MODEL_PATH, image_path, OUTPUT_DIR, CLASS_NAMES, IMGSZ
        )

    # 3. Filters
    if run_all or args.filters:
        print("\n[3/4] Konvolusyon Filtre Gorsellestiremesi")
        print("  Modelin ogrendigi kenar/doku algilayicilari.")
        from core.filters import visualize_filters
        visualize_filters(MODEL_PATH, OUTPUT_DIR)

    # 4. t-SNE
    if run_all or args.tsne:
        print("\n[4/4] t-SNE Embedding Gorsellestiremesi")
        print("  Modelin siniflari feature uzayinda nasil ayirdigini gosterir.")
        from core.embeddings import visualize_tsne
        visualize_tsne(
            MODEL_PATH, TRAIN_IMAGES, DATASET_PATH, OUTPUT_DIR,
            CLASS_NAMES, num_images=TSNE_NUM_IMAGES
        )

    # 5. Tespit Bazli Grad-CAM Heatmap
    if args.heatmap:
        print("\n[5] Tespit Bazli Grad-CAM Heatmap")
        print("  Her tespit icin modelin neye baktigini gosterir.")
        from core.detection_heatmap import visualize_detection_heatmap
        visualize_detection_heatmap(
            MODEL_PATH, image_path, OUTPUT_DIR, CLASS_NAMES, IMGSZ
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 58}")
    print(f"  Tamamlandi! Sure: {elapsed:.1f} saniye")
    print(f"  Ciktilari gormek icin: {OUTPUT_DIR}")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()
