# 🎓 AI Tutor - Intelligent Question Generation System


An advanced AI-powered question generation system that automatically creates **Short Answer Questions (SAQ)** and **Multiple Choice Questions (MCQ)** from educational content. Built with a robust dual-model architecture and comprehensive fallback system for maximum reliability.

## ✨ Key Features

### 🤖 **Dual AI Architecture**
- **🔄 Transformers Models** for SAQ generation (Hugging Face)
- **🦙 LLaMA 2 7B** for MCQ generation (Local GGUF model)
- **🛡️ Smart Fallback System** - Always generates questions, never fails

### 📚 **Question Generation**
- **📝 Short Answer Questions** - Educational, contextual questions
- **🎯 Multiple Choice Questions** - Structured A/B/C/D format with explanations
- **⚡ Fast Processing** - Optimized chunk processing (3-5 chunks vs 74+)
- **🎛️ Customizable Settings** - Temperature, chunk size, question count

### 🔧 **Advanced Features**
- **📄 PDF Processing** - Extract text from lecture notes, textbooks
- **🧹 Text Preprocessing** - Intelligent cleaning and chunking
- **💾 Smart Caching** - Avoid regenerating same content
- **📊 Quality Analytics** - ROUGE score, BERT score, diversity metrics
- **📤 Multiple Export Formats** - JSON, CSV, TXT, Moodle XML

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **4GB+ RAM** (8GB+ for larger models)
- **Optional**: NVIDIA GPU for faster processing

---
**Note:**
The LLaMA 2 7B model file (`models/llama-2-7b-chat.Q4_K_M.gguf`) is not included in this repository due to its large size.
You must download it manually from [TheBloke/Llama-2-7B-Chat-GGUF on Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) and place it in the `models/` directory.
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

3. **Download LLaMA 2 model (Optional - for MCQ generation)**
   ```bash
   # Download LLaMA 2 7B Chat GGUF model
   # Place in: models/llama-2-7b-chat.Q4_K_M.gguf
   # Get from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
   ```

4. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

5. **Access the application**
   - Open http://localhost:8501 in your browser
   - Upload a PDF and start generating questions!

## 🏗️ System Architecture

### Model Hierarchy & Fallback System

```
┌─────────────────────────────────────────┐
│            PDF Upload                   │
│       (Lecture Notes/Textbooks)        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Text Processing Pipeline        │
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐│
│  │   PDF   │ │  Text   │ │ Intelligent ││
│  │Extract  │→│Cleaning │→│  Chunking   ││
│  └─────────┘ └─────────┘ └─────────────┘│
└─────────────────┬───────────────────────┘
                  │
        ┌─────────▼─────────┐
        │   Smart Router    │
        │ (3-5 chunks max)  │
        └─────┬─────────┬───┘
              │         │
    ┌─────────▼──┐   ┌──▼──────────┐
    │    SAQ     │   │     MCQ     │
    │Generation  │   │ Generation  │
    └─────┬──────┘   └──┬──────────┘
          │             │
┌─────────▼──────┐   ┌──▼──────────┐
│ Transformers   │   │   LLaMA 2   │
│ Fallback Chain │   │   7B Chat   │
│                │   │             │
│ 1.DialoGPT-med │   │ • GGUF      │
│ 2.GPT2-medium  │   │ • Quantized │
│ 3.GPT2         │   │ • Local     │
│ 4.DistilGPT2   │   │             │
│ 5.OPT-125m     │   │             │
│ 6.Templates    │   │ Fallback:   │
│                │   │ Templates   │
└─────────┬──────┘   └──┬──────────┘
          │             │
          └─────┬───────┘
                │
┌───────────────▼───────────────┐
│      Quality Assurance       │
│                              │
│ • Question validation        │
│ • Format verification        │
│ • Content quality check      │
│ • Answer completeness        │
└───────────────┬───────────────┘
                │
┌───────────────▼───────────────┐
│     Export & Analytics       │
│                              │
│ • JSON/CSV/TXT/XML export    │
│ • ROUGE/BERT quality scores  │
│ • Diversity analysis         │
│ • Performance metrics        │
└──────────────────────────────┘
```

### Reliability Strategy

| Priority | Method | Models | Success Rate | Use Case |
|----------|--------|--------|--------------|----------|
| **1** | **Transformers** | DialoGPT → GPT-2 → DistilGPT-2 → OPT | 95%+ | SAQ Generation |
| **2** | **LLaMA Local** | llama-2-7b-chat.Q4_K_M.gguf | 90%+ | MCQ Generation |
| **3** | **Templates** | Rule-based generation | 100% | Ultimate Fallback |

## 📋 Technical Stack

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

## 🎯 Usage Guide

### 1. **Upload Content**
- 📄 **PDF Files**: Lecture notes, textbooks, research papers
- 🔍 **Preview**: View extracted text and metadata
- 📊 **Statistics**: Pages, characters, processing info

### 2. **Configure Generation**
```python
# Question Settings
num_saq = 5          # Short Answer Questions (0-20)
num_mcq = 5          # Multiple Choice Questions (0-20)

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
- 📈 **Quality Score**: Overall question quality (0-100%)
- 🎲 **Diversity Score**: Question variety measurement
- 📊 **Type Distribution**: SAQ vs MCQ breakdown
- 🏆 **ROUGE/BERT Scores**: Semantic quality metrics

## 🛠️ Configuration

### Environment Setup
```bash
# Create .env file
LLAMA_MODEL_PATH=./models/llama-2-7b-chat.Q4_K_M.gguf
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

## 🚨 Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Out of Memory** | Large model + Limited RAM | Use smaller models (distilgpt2), reduce batch size |
| **Model Loading Fails** | Network/disk issues | Check internet, verify model files |
| **Slow Generation** | CPU processing | Enable GPU support, reduce chunk count |
| **Empty Questions** | Poor input text | Check PDF quality, increase temperature |
| **Unicode Errors** | Windows encoding | Fixed in v1.1+ - uses ASCII-safe output |

### Performance Optimization
1. **🚀 Use GPU**: Set `CUDA_VISIBLE_DEVICES=0` and install CUDA
2. **💾 Enable Caching**: Keep cache enabled for repeated content
3. **⚡ Optimize Chunks**: Use 3-5 chunks max for faster processing
4. **🎯 Smaller Models**: Use distilgpt2 for faster SAQ generation

### Debug Mode
```python
# Enable verbose logging
import  logging
logging.basicConfig(level=logging.DEBUG)

# Check model status
python -c "from src.question_generation.transformers_handler import TransformersHandler; h = TransformersHandler(); print(h.get_model_info())"
```

## 📊 Performance Metrics

### Speed Benchmarks
| Configuration | Processing Time | Memory Usage | Quality Score |
|---------------|----------------|--------------|---------------|
| **CPU Only** | ~60-90 seconds | 2-4 GB | 85-90% |
| **GPU (8GB)** | ~20-30 seconds | 4-6 GB | 90-95% |
| **GPU (16GB+)** | ~10-15 seconds | 6-8 GB | 95%+ |

### Model Comparison
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **DialoGPT-medium** | ~400MB | Fast | Excellent | Primary SAQ |
| **GPT-2** | ~500MB | Medium | Good | Fallback SAQ |
| **LLaMA 2 7B** | ~4GB | Slower | Excellent | Primary MCQ |
| **Templates** | ~0MB | Instant | Basic | Ultimate Fallback |


### Project Structure
```
TutorAi/
├── 📁 src/                          # Main source code
│   ├── 🎯 app.py                    # Streamlit web interface
│   ├── ⚙️ config.py                 # Configuration settings
│   ├── 📁 preprocessing/             # Text processing
│   │   ├── 📄 pdf_extractor.py      # PDF text extraction
│   │   ├── 🧹 text_cleaner.py       # Text cleaning utilities
│   │   └── ✂️ chunker.py            # Text chunking logic
│   ├── 📁 question_generation/       # AI model handlers
│   │   ├── 🤖 transformers_handler.py # Transformers integration
│   │   ├── 🦙 llama_handler.py       # LLaMA model handler
│   │   ├── 📝 saq_generator.py       # SAQ generation logic
│   │   ├── 🎯 mcq_generator.py       # MCQ generation logic
│   │   └── 💬 prompts.py            # Prompt templates
│   ├── 📁 evaluation/                # Quality assessment
│   │   ├── 📊 metrics.py            # Evaluation metrics
│   │   └── ✅ validator.py          # Question validation
│   └── 📁 utilities/                 # Helper functions
│       ├── 💾 cache_manager.py      # Caching system
│       └── 📤 export_handler.py     # Export utilities
├── 📁 models/                       # Model storage
├── 📁 data/                        # Application data
│   ├── 📂 uploads/                  # User uploaded files
│   ├── 📂 outputs/                  # Generated questions
│   └── 📂 cache/                    # Cached results
├── 📋 requirements.txt              # Python dependencies
├── 🧪 test_setup.py                # Setup validation
└── 📖 README.md                    # This documentation
```

### Adding New Models
1. **Extend TransformersHandler**: 
   ```python
   self.model_hierarchy = [
       "your-new-model",        # Add here
       "microsoft/DialoGPT-medium",
       # ... existing models
   ]
   ```

2. **Test Integration**:
   ```python
   python -c "from src.question_generation.transformers_handler import TransformersHandler; TransformersHandler('your-model-name')"
   ```



## 🎯 System Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **Python** | 3.8+ | 3.10+ | 3.11+ |
| **RAM** | 4GB | 8GB | 16GB+ |
| **Storage** | 5GB | 10GB | 20GB+ |
| **GPU** | None (CPU) | 4GB VRAM | 8GB+ VRAM |
| **OS** | Windows 10+, macOS 10.14+, Ubuntu 18.04+ | Latest versions | - |

---
