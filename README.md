#  AI Tutor - Intelligent Question Generation System


An advanced AI-powered question generation system that automatically creates **Short Answer Questions (SAQ)** and **Multiple Choice Questions (MCQ)** from educational content. Built with a robust dual-model architecture and comprehensive fallback system for maximum reliability.

##  Key Features

###  **Dual AI Architecture with Multiple LLM Options**
- **Google Gemini 2.0 Flash** - Main LLM for high-quality MCQ generation (API-based, fast & efficient)
- **LLaMA 2 7B Quantized** - Alternative local LLM for MCQ generation
- **Transformers Models** - For SAQ generation (Hugging Face)
- **Smart Fallback System** - Always generates questions, never fails
- **Easy Switching** - Toggle between Gemini and LLaMA in the UI

###  **Question Generation**
- **Short Answer Questions** - Educational, contextual questions
- **Multiple Choice Questions** - Structured A/B/C/D format with explanations
- **Fast Processing** - Optimized chunk processing (3-5 chunks vs 74+)
- **Customizable Settings** - Temperature, chunk size, question count

###  **Advanced Features**
- **PDF Processing** - Extract text from lecture notes, textbooks
- **Text Preprocessing** - Intelligent cleaning and chunking
- **RAG (Retrieval-Augmented Generation)** - Semantic search across the entire document for enhanced context
- **Smart Caching** - Avoid regenerating same content
- **Quality Analytics** - ROUGE score, BERT score, diversity metrics
- **Multiple Export Formats** - JSON, CSV, TXT, Moodle XML

##  Quick Start

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **4GB+ RAM** (8GB+ for larger models)
- **Gemini API Key** (Get free from [Google AI Studio](https://makersuite.google.com/app/apikey))
- **Optional**: NVIDIA GPU for faster local model processing

---
**Note:**
- **For Gemini (Recommended)**: Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **For Local LLaMA**: The LLaMA 2 7B model file (`models/llama-2-7b-chat.Q4_K_M.gguf`) is not included due to its large size. Download it manually from [TheBloke/Llama-2-7B-Chat-GGUF on Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) and place it in the `models/` directory.
---

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mukundhr/TutorAi.git
   cd TutorAi
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your LLM (Choose one or both)**

   **Option A: Google Gemini (Recommended)**
   - Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root
   - Add your API key:
     ```bash
     GEMINI_API_KEY=your_api_key_here
     DEFAULT_LLM=gemini
     ```

   **Option B: Local LLaMA 2**
   - Download the model from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
   - Place in `models/llama-2-7b-chat.Q4_K_M.gguf`
   - Set in `.env`:
     ```bash
     DEFAULT_LLM=llama
     LLAMA_MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
     ```

4. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

5. **Access the application**
   - Open http://localhost:8501 in your browser
   - Select your preferred LLM (Gemini or LLaMA) in the sidebar
   - Upload a PDF and start generating questions!



### Reliability Strategy

| Priority | Method | Models | Success Rate | Use Case |
|----------|--------|--------|--------------|----------|
| **1** | **Google Gemini 2.0** | gemini-2.0-flash-exp | 98%+ | Primary MCQ Generation (API) |
| **2** | **Local LLaMA** | llama-2-7b-chat.Q4_K_M.gguf | 90%+ | Alternative MCQ (Privacy) |
| **3** | **Transformers** | DialoGPT → GPT-2 → DistilGPT-2 → OPT | 95%+ | SAQ Generation |
| **4** | **Templates** | Rule-based generation | 100% | Ultimate Fallback |

## 🎛️ LLM Selection Guide

### Google Gemini (Recommended)
**Pros:**
-  Higher quality questions
-  Faster generation (API-based)
-  No local storage needed
-  Free tier available
-  No GPU required

**Cons:**
-  Requires internet connection
-  Needs API key
-  Usage limits on free tier

### Local LLaMA 2 7B
**Pros:**
-  Complete privacy (runs locally)
-  No internet required
-  No API costs
-  Unlimited usage

**Cons:**
-  Large model file (~4GB)
-  Slower generation
-  Requires more RAM
-  GPU recommended for speed

### Switching Between LLMs
You can easily switch between Gemini and LLaMA:
1. Open the sidebar in the app
2. Select "Google Gemini (API)" or "Local LLaMA 2 7B"
3. Click "Load Models"
4. Start generating questions!

##  RAG (Retrieval-Augmented Generation)

### What is RAG?
RAG enhances question generation by using **semantic search** to find the most relevant context from the entire document, not just the current chunk. This results in:
-  **Better context understanding** - Questions aware of the full document
-  **More coherent questions** - Better connection between concepts
-  **Smarter distractors** - MCQ options that are more challenging and relevant
-  **Reduced hallucination** - Questions grounded in actual content

### How it Works
1. **Document Upload** → PDF text is extracted
2. **Vector Store Creation** → All chunks are embedded using `all-MiniLM-L6-v2` model
3. **Semantic Search** → For each chunk, RAG retrieves the top 3 most relevant chunks
4. **Enhanced Generation** → LLM uses both the current chunk + retrieved context
5. **Better Questions** → Results in more comprehensive and accurate questions

### Enabling RAG
In the app sidebar:
- ☑️ Check "Enable RAG (Retrieval-Augmented Generation)"
- Questions will be marked as `rag_enhanced: true`

### RAG Configuration
In your `.env` file:
```bash
ENABLE_RAG=true
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast & lightweight
RAG_TOP_K=3  # Number of relevant chunks to retrieve
RAG_MIN_SCORE=0.3  # Minimum similarity threshold
```

### Performance Impact
| Mode | Speed | Quality | Memory |
|------|-------|---------|--------|
| **Without RAG** | Fast | Good | Low |
| **With RAG** | Slightly slower (~10-20% overhead) | Excellent | Medium (embeddings cached) |

**Recommendation:** Enable RAG for better quality questions, especially for complex or lengthy documents.

##  Technical Stack

### Core AI Libraries
```python
transformers   # Hugging Face Transformers
torch>         # PyTorch for ML models
llama-cpp-python  # LLaMA C++ bindings
accelerate     # Model acceleration
bitsandbytes   # Quantization support
```

### Text Processing
```python
pdfplumber   # PDF text extraction
spacy        # NLP processing
nltk            # Natural language toolkit
sentence-transformers  # Semantic embeddings
```

### Web Framework & UI
```python
streamlit   # Web application framework
pandas         # Data manipulation
numpy         # Numerical computing
```

### Quality Assessment
```python
rouge-score   # Text quality metrics
bert-score   # Semantic similarity
scikit-learn   # ML utilities
```

##  Usage Guide

### 1. **Upload Content**
-  **PDF Files**: Lecture notes, textbooks, research papers
-  **Preview**: View extracted text and metadata
-  **Statistics**: Pages, characters, processing info

### 2. **Configure Generation**
```python
# Question Settings
num_saq = 5          # Short Answer Questions (0-20)
num_mcq = 5          # Multiple Choice Questions (0-20)

# RAG Settings
enable_rag = True    # Enable semantic search for better context

# Model Settings  
temperature = 0.7    # Creativity level (0.1-1.0)
chunk_size = 800     # Text chunk size (200-1500)

# Performance Settings
enable_cache = True  # Cache results for faster regeneration
```

### 3. **Question Generation Process**
1. **Text Extraction** - PDF → Raw Text
2. **Preprocessing** - Clean → Segment → Chunk
3. **SAQ Generation** - Transformers model processing
4. **MCQ Generation** - LLaMA model processing  
5. **Validation** - Quality checks and filtering
6. **Results** - Display with answers and explanations

### 4. **Export Options**

#### JSON Format
```json
{
  "type": "Short Answer",
  "question": "What are the key principles of machine learning?",
  "answer": "Supervised learning, unsupervised learning, and reinforcement learning...",
  "points": 5
}
```

#### Moodle XML
```xml
<question type="essay">
  <name><text>Machine Learning Principles</text></name>
  <questiontext format="html">
    <text>What are the key principles of machine learning?</text>
  </questiontext>
</question>
```

### 5. **Quality Analytics**
-  **Quality Score**: Overall question quality (0-100%)
-  **Diversity Score**: Question variety measurement
-  **Type Distribution**: SAQ vs MCQ breakdown
-  **ROUGE/BERT Scores**: Semantic quality metrics

##  Configuration

### Environment Setup
```bash
# Create .env file
# Choose your LLM
DEFAULT_LLM=gemini  # or "llama" for local model

# Gemini Configuration (if using Gemini)
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash-exp  # Gemini 2.0 Flash (recommended)

# LLaMA Configuration (if using local model)
LLAMA_MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf

# General settings
CUDA_VISIBLE_DEVICES=0
TRANSFORMERS_CACHE=./cache/transformers
HF_HOME=./cache/huggingface
```

### Model Configuration
```python
# config.py customization
MODEL_CONTEXT_LENGTH = 4096    # LLaMA context window
MODEL_MAX_TOKENS = 512         # Maximum generation length
MODEL_TEMPERATURE = 0.7        # Default creativity
MAX_CHUNK_SIZE = 800          # Text chunk size
MIN_CHUNK_SIZE = 200          # Minimum chunk size
DEFAULT_NUM_SAQ = 5           # Default SAQ count
DEFAULT_NUM_MCQ = 5           # Default MCQ count
```

### Performance Tuning
```python
# GPU Settings
n_gpu_layers = 32             # GPU layers for LLaMA (0 = CPU only)
n_threads = 4                 # CPU threads for processing

# Memory Management  
torch_dtype = torch.float16   # Use half precision (GPU)
device_map = "auto"          # Automatic device placement
```


### Model Comparison
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **DialoGPT-medium** | ~400MB | Fast | Excellent | Primary SAQ |
| **GPT-2** | ~500MB | Medium | Good | Fallback SAQ |
| **LLaMA 2 7B** | ~4GB | Slower | Excellent | Primary MCQ |
| **Templates** | ~0MB | Instant | Basic | Ultimate Fallback |

---