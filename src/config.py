"""
Configuration file for AI Tutor System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", str(MODEL_DIR / "llama-2-7b-chat.Q4_K_M.gguf"))
MODEL_CONTEXT_LENGTH = 4096
MODEL_MAX_TOKENS = 512
MODEL_TEMPERATURE = 0.7
MODEL_TOP_P = 0.95

# Question generation settings
DEFAULT_NUM_SAQ = 5
DEFAULT_NUM_MCQ = 5
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Evaluation settings
ENABLE_BERTSCORE = True
ENABLE_ROUGE = True

# Streamlit settings
PAGE_TITLE = "AI Tutor - Question Generation System"
PAGE_ICON = "🎓"
LAYOUT = "wide"