"""
Configuration file for AI Tutor System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# LLM Selection
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "gemini")  # Options: "gemini" or "llama"

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Get from https://makersuite.google.com/app/apikey
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")  # Gemini 2.0 Flash (lightweight & fast)

# LLaMA configuration
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", str(MODEL_DIR / "llama-2-7b-chat.Q4_K_M.gguf"))

# Model configuration (shared)
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

# RAG (Retrieval-Augmented Generation) settings
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Lightweight & fast
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "2"))  # Number of context chunks to retrieve (reduced for quality)
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))  # Minimum similarity score (increased for relevance)
RAG_VECTOR_CACHE_DIR = DATA_DIR / "vector_cache"  # Cache for vector stores

# Create RAG cache directory
RAG_VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit settings
PAGE_TITLE = "Tutor - Question Generation System"
PAGE_ICON = "🎓"
LAYOUT = "wide"