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
    python visualize.py --coach            # AI egitim kocu (interaktif)

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
    parser.add_argument("--coach", action="store_true",
                        help="AI egitim kocu - Claude API (ucretli)")
    parser.add_argument("--coach-local", action="store_true",
                        help="Yerel uzman sistem (ucretsiz, API gerektirmez)")
    args = parser.parse_args()

    # Coach modlari ayri calisir
    if args.coach:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config
        from core.ai_coach import run_coach
        run_coach(config)
        return

    if args.coach_local:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config
        from core.expert_system import run_local_coach
        run_local_coach(config)
        return

    run_all = not any([
        args.feature_maps, args.gradcam, args.filters, args.tsne
    ])

    # Config'i import et
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import (
        MODEL_PATH, TRAIN_IMAGES, DATASET_PATH,
        OUTPUT_DIR, CLASS_NAMES, IMGSZ, TSNE_NUM_IMAGES
    )

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

    elapsed = time.time() - t0
    print(f"\n{'=' * 58}")
    print(f"  Tamamlandi! Sure: {elapsed:.1f} saniye")
    print(f"  Ciktilari gormek icin: {OUTPUT_DIR}")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()
