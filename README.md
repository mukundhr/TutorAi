#  AI Tutor - Intelligent Question Generation System


An advanced AI-powered question generation system that automatically creates **Short Answer Questions (SAQ)** and **Multiple Choice Questions (MCQ)** from educational content. Built with a robust dual-model architecture and comprehensive fallback system for maximum reliability.

##  Key Features

###  **Dual AI Architecture with Multiple LLM Options**
- **Google Gemini 2.0 Flash** - Main LLM for high-quality MCQ generation (API-based, fast & efficient)
- **LLaMA 2 7B Quantized** - Alternative local LLM for MCQ generation
- **Transformers Models** - For SAQ generation (Hugging Face)
- **Smart Fallback System** - Uses different methods to generate questions even if one method fails
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

3. **Configure your LLM**

   **Google Gemini**
   - Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root
   - Add your API key:
     ```bash
     GEMINI_API_KEY=your_api_key_here
     DEFAULT_LLM=gemini
     ```

   **Local LLaMA 2**
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
   - Select your preferred LLM (Gemini or LLaMA) in the sidebar
   - Upload a PDF and start generating questions!



### Reliability Strategy

| Priority | Method | Models | Success Rate | Use Case |
|----------|--------|--------|--------------|----------|
| **1** | **Google Gemini 2.0** | gemini-2.0-flash-exp | 98%+ | Primary MCQ Generation (API) |
| **2** | **Local LLaMA** | llama-2-7b-chat.Q4_K_M.gguf | 90%+ | Alternative MCQ (Privacy) |
| **3** | **Transformers** | DialoGPT → GPT-2 → DistilGPT-2 → OPT | 95%+ | SAQ Generation |

##  LLM Selection Guide

| Feature | Google Gemini (Recommended) | Local LLaMA 2 7B |
|---------|----------------------------|------------------|
| **Quality** | Higher quality | Good quality |
| **Speed** | Faster (API) | Slower (local) |
| **Internet** | Required | Not required* |
| **Storage** | None | ~4GB model file |
| **Cost** | Free tier + limits | No cost, unlimited |
| **Setup** | API key only | Download 4GB model |

### Offline Usage
**For complete offline operation:**
1. Run once with internet (downloads transformers to cache)
2. Download LLaMA model manually
3. Use LLaMA for MCQ (not Gemini)
4. All future runs work 100% offline

### Switching Between LLMs
In the app sidebar → Select LLM → Click "Load Models" → Generate questions!

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

