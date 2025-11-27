# TutorAI - Intelligent Question Generation & Learning Assistant

**Transform any PDF into intelligent questions and get instant AI-powered answers**

---

## Overview

**TutorAI** is an advanced educational platform that leverages cutting-edge AI to revolutionize learning from documents. Built with state-of-the-art NLP and semantic search, it offers:

**Intelligent Question Generation** - Automatically create contextual MCQs and SAQs from any PDF  
**AI-Powered Chat Assistant** - Ask questions and get answers with FAISS semantic search  
**Quality Analytics** - Comprehensive metrics with BLEU, ROUGE, and BERT scores  
**Adaptive Difficulty** - Generate questions across easy, medium, and hard levels

## Features

### **Advanced Question Generation**

**Multiple Question Types**
- **MCQ (Multiple Choice)** - 4 options with detailed explanations
- **SAQ (Short Answer)** - Comprehensive 4-5 line responses

**Adaptive Difficulty Levels**
- **Easy** - Direct recall & basic comprehension
- **Medium** - Concept application & inference
- **Hard** - Critical thinking & synthesis

### **Dual AI Architecture**

| LLM | Type | Speed | Quality | Use Case |
|-----|------|-------|---------|----------|
| **Google Gemini 2.0 Flash** | Cloud API | Very Fast | Excellent | Primary (Recommended) |
| **LLaMA 2 7B (Quantized)** | Local | Moderate | Good | Privacy/Offline Mode |

**Smart Fallback System** - Automatically switches to alternative models if primary fails

### **RAG-Powered Chat Assistant** 

- **FAISS Semantic Search** - State-of-the-art vector similarity using Sentence Transformers
- **Contextual Answers** - Get precise answers grounded in your document
- **Source Citations** - Every answer includes relevant text snippets
- **Conversation History** - Maintains context across multiple questions
- **Semantic Understanding** - Understands synonyms and concepts, not just keywords

### **Quality Analytics Dashboard**

- **BLEU Score** - Translation-quality metric for text similarity
- **ROUGE-L F1** - Longest common subsequence analysis
- **BERT Score** - Semantic similarity using neural embeddings
- **Interactive Charts**:
  - Scatter plots (Difficulty vs Quality)
  - Histograms (Score distributions)
  - Radar charts (Multi-metric comparison)
- **Feedback System** - Collect user ratings and comments

### Evaluation Metrics

1. **BLEU Score** (via ROUGE-1 precision)
   - Measures n-gram overlap
   - Range: 0-1 (higher = better)

2. **ROUGE-L F1**
   - Longest common subsequence
   - Balances precision & recall

3. **BERT Score** (proxy via semantic similarity)
   - Neural embedding comparison
   - Context-aware evaluation

4. **Diversity Score**
   - Unique n-grams / total n-grams
   - Prevents repetitive content

### **Technical Capabilities**

- **PDF Processing** - Multi-method extraction (pdfplumber, PyPDF2) with fallbacks
- **Intelligent Chunking** - spaCy-based sentence segmentation
- **Smart Caching** - MD5-based caching to avoid redundant processing
- **Custom Configuration** - Temperature, chunk size, top-k retrieval tuning
- **Export Options** - JSON, CSV, TXT formats

## Quick Start

### Installation

**1. Clone the Repository**
```bash
git clone https://github.com/mukundhr/TutorAi.git
cd TutorAi
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**Optional: For Local LLaMA Support (CPU-only)**
```bash
pip install llama-cpp-python
```

**Optional: For Local LLaMA Support (GPU with CUDA)**
```bash
# Requires NVIDIA CUDA Toolkit 12.1+ installed first
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

> **Note**: LLaMA is optional. The app works perfectly with just Gemini (recommended).

**3. Download Required Models**

```bash
# For spaCy (required)
python -m spacy download en_core_web_sm

**4. Configure API Keys**

Create a `.env` file in the project root:

```bash
# Google Gemini (Recommended)
GEMINI_API_KEY=your_api_key_here
DEFAULT_LLM=gemini

# Optional: Local LLaMA Configuration
DEFAULT_LLM=llama
LLAMA_MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
```

> **Get your free Gemini API key**: [Google AI Studio](https://makersuite.google.com/app/apikey)

**5. Launch the Application**
```bash
streamlit run src/app.py
```

**6. Start Learning!**
- Navigate to `http://localhost:8501`
- Upload a PDF document
- Generate questions or chat with your document

---

### Local LLaMA Setup

For **offline**, download the LLaMA 2 7B model:

1. Visit [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
2. Download `llama-2-7b-chat.Q4_K_M.gguf` (~4GB)
3. Place in `models/` directory
4. Update `.env` with `DEFAULT_LLM=llama`

**Note**: First run with internet required to download Sentence Transformer embedding models.



### RAG Pipeline (FAISS Semantic Search)

```python
Document → Chunking → Sentence Transformers → 384D Embeddings
                                                      ↓
Query → Embedding → FAISS Search (L2) → Top-K Chunks → LLM → Answer
```

**Key Technologies:**
- **Embeddings**: `all-MiniLM-L6-v2` (384 dimensions, fast & accurate)
- **Vector Store**: FAISS `IndexFlatL2` (exact similarity search)
- **Retrieval**: Top-K semantic matching with similarity scores
