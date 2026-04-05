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

# Agent Ayarlari
AGENT_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"
AGENT_OLLAMA_URL = "http://localhost:11434"
AGENT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
AGENT_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
AGENT_CHROMA_DIR = os.path.join(AGENT_KB_DIR, "chroma_db")
AGENT_NUM_CTX = 8192
AGENT_TEMPERATURE = 0.3
