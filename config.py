import os

# Desktop path (Turkish Windows uyumlu)
# Windows'ta "Desktop" symlink olabilir ama icerik farkli olabilir.
# Gercek Masaustu klasorunu bul.
_home = os.path.expanduser("~")
DESKTOP = None
for name in ["Masaüstü", "Masa\u00fcst\u00fc", "Desktop"]:
    _try = os.path.join(_home, name)
    if os.path.isdir(_try) and os.path.exists(os.path.join(_try, "TEKNOFEST_GUNCEL")):
        DESKTOP = _try
        break
if DESKTOP is None:
    # Fallback
    DESKTOP = os.path.join(_home, "Desktop")

# Model ve dataset yollari
MODEL_PATH = os.path.join(
    DESKTOP, "TEKNOFEST_GUNCEL", "models", "teknofest_v1", "weights", "best.pt"
)
DATASET_PATH = os.path.join(
    DESKTOP, "TEKNOFEST_GUNCEL", "03.03_karma_tekno_dataset"
)
TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_PATH, "train", "labels")
VALID_IMAGES = os.path.join(DATASET_PATH, "valid", "images")

# Sinif isimleri
CLASS_NAMES = {0: "F16", 1: "helicopter", 2: "drone", 3: "balistic"}
NUM_CLASSES = len(CLASS_NAMES)

# Cikti klasoru
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ayarlar
IMGSZ = 640
MAX_FEATURE_CHANNELS = 32
TSNE_NUM_IMAGES = 300
