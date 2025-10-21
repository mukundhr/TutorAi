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
- **Short Answer Questions** - Educational, contextual questions with 4-5 line answers
- **Multiple Choice Questions** - Structured A/B/C/D format with explanations
- **Exact Question Count** - Get exactly the number of questions you request (e.g., 5 MCQs + 5 SAQs = 10 total)
- **Three Difficulty Levels**:
  - **Easy**: Direct recall, basic comprehension, definitions (answer clearly stated in text)
  - **Medium**: Application of concepts, connecting ideas, some inference required
  - **Hard**: Critical thinking, synthesis, evaluation, applying to new situations
- **Fast Processing** - Combines chunks for better context
- **Customizable Settings** - Temperature, chunk size, question count, difficulty

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
```

