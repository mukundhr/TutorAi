"""
Text preprocessing for question generation
Enhanced PDF extraction with multiple fallback methods
"""

import re
import pdfplumber
import spacy
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
from io import BytesIO

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class TextPreprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize text preprocessor
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from PDF with multiple fallback methods
        
        Method 1: pdfplumber (best for most PDFs)
        Method 2: PyPDF2 (fallback)
        Method 3: Extract with layout preservation
        """
        text = ""
        
        # Method 1: Try pdfplumber first (best quality)
        try:
            print("🔍 Trying pdfplumber extraction...")
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        # Try standard extraction
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 0:
                            text += page_text + "\n\n"
                        else:
                            # Try with layout preservation
                            page_text = page.extract_text(layout=True)
                            if page_text:
                                text += page_text + "\n\n"
                    except Exception as e:
                        print(f"⚠️ Page {i+1} extraction failed: {e}")
                        continue
            
            if len(text.strip()) > 100:  # Success threshold
                print(f"✅ pdfplumber: Extracted {len(text)} characters")
                return text
            else:
                print("⚠️ pdfplumber extraction insufficient, trying fallback...")
        
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}")
        
        # Method 2: Try PyPDF2 as fallback
        try:
            print("🔍 Trying PyPDF2 extraction...")
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                    except Exception as e:
                        print(f"⚠️ Page {i+1} failed: {e}")
                        continue
            
            if len(text.strip()) > 100:
                print(f"✅ PyPDF2: Extracted {len(text)} characters")
                return text
        
        except Exception as e:
            print(f"⚠️ PyPDF2 failed: {e}")
        
        # Method 3: Try pdfplumber with different settings
        try:
            print("🔍 Trying pdfplumber with custom settings...")
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        # Extract with custom settings
                        page_text = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=False,
                            x_density=7.25,
                            y_density=13
                        )
                        if page_text:
                            text += page_text + "\n\n"
                    except:
                        continue
            
            if len(text.strip()) > 100:
                print(f"✅ Custom extraction: Extracted {len(text)} characters")
                return text
        
        except Exception as e:
            print(f"⚠️ Custom extraction failed: {e}")
        
        # If all methods fail
        if len(text.strip()) < 100:
            raise Exception(
                "Failed to extract sufficient text from PDF. "
                "Possible issues:\n"
                "1. PDF might be image-based (scanned document) - needs OCR\n"
                "2. PDF might be encrypted or password-protected\n"
                "3. PDF might have unusual formatting\n"
                "Try converting the PDF to text format or use a different PDF."
            )
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        
        if not text or len(text.strip()) == 0:
            return ""
        
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove page numbers (standalone numbers)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Fix common PDF artifacts
        text = text.replace('', "'")  # Smart quotes
        text = text.replace('', "'")
        text = text.replace('', '"')
        text = text.replace('', '"')
        text = text.replace('', '-')  # Em dash
        text = text.replace('', '-')  # En dash
        
        # Remove headers/footers (lines with <5 words at start/end of pages)
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                cleaned_lines.append('')
                continue
            
            # Skip very short lines that look like headers/footers
            word_count = len(line.split())
            if word_count < 3 and (i < 5 or i > len(lines) - 5):
                continue
            
            # Skip lines that are just page numbers
            if line.isdigit() and len(line) < 4:
                continue
            
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove multiple spaces again
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks
        Uses sentence boundaries to avoid breaking mid-sentence
        """
        
        if not text or len(text.strip()) == 0:
            return []
        
        # Use spaCy for sentence segmentation (more reliable than NLTK for this)
        try:
            doc = nlp(text[:1000000])  # Limit to 1M chars to avoid memory issues
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            print(f"⚠️ spaCy failed, using NLTK: {e}")
            # Fallback to NLTK
            sentences = sent_tokenize(text[:1000000])
        
        if not sentences:
            # Ultimate fallback: split by periods
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunks.append({
                    'id': f'chunk_{chunk_id:03d}',
                    'text': current_chunk.strip(),
                    'char_count': current_size,
                    'word_count': len(current_chunk.split())
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_size += sentence_len
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'id': f'chunk_{chunk_id:03d}',
                'text': current_chunk.strip(),
                'char_count': current_size,
                'word_count': len(current_chunk.split())
            })
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get last N characters for overlap"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break at sentence boundary
        overlap_text = text[-self.chunk_overlap:]
        
        # Find the first period to start from complete sentence
        first_period = overlap_text.find('. ')
        if first_period != -1:
            return overlap_text[first_period + 2:]
        
        return overlap_text
    
    def filter_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter out low-quality chunks"""
        filtered = []
        
        for chunk in chunks:
            text = chunk['text']
            
            # Skip if too short
            if chunk['word_count'] < 30:  # Reduced from 50 for shorter documents
                continue
            
            # Skip if mostly numbers/symbols
            alpha_count = sum(c.isalpha() or c.isspace() for c in text)
            if len(text) > 0:
                alpha_ratio = alpha_count / len(text)
                if alpha_ratio < 0.6:  # Reduced from 0.7
                    continue
            
            # Skip if looks like table of contents
            if text.lower().count('chapter') > 3 or text.count('....') > 3:
                continue
            
            filtered.append(chunk)
        
        print(f"✅ Filtered to {len(filtered)} quality chunks")
        return filtered
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Complete pipeline: Extract → Clean → Chunk → Filter
        
        Returns:
            List of processed text chunks ready for question generation
        """
        print("\n📄 Processing PDF...")
        
        try:
            # Step 1: Extract
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            if not raw_text or len(raw_text.strip()) < 100:
                raise Exception("Extracted text is too short or empty")
            
            print(f"✅ Extracted {len(raw_text)} characters")
            
            # Step 2: Clean
            clean_text = self.clean_text(raw_text)
            print(f"✅ Cleaned to {len(clean_text)} characters")
            
            if len(clean_text.strip()) < 50:
                raise Exception("Cleaned text is too short")
            
            # Step 3: Chunk
            chunks = self.chunk_text(clean_text)
            
            if not chunks:
                raise Exception("No chunks created from text")
            
            # Step 4: Filter
            quality_chunks = self.filter_chunks(chunks)
            
            if not quality_chunks:
                # If filtering removed everything, return unfiltered chunks
                print("⚠️ Filtering removed all chunks, using unfiltered")
                return chunks
            
            return quality_chunks
        
        except Exception as e:
            print(f"❌ Error processing PDF: {str(e)}")
            raise


# Example usage and testing
if __name__ == "__main__":
    preprocessor = TextPreprocessor(chunk_size=800, chunk_overlap=100)
    
    # Test with sample text
    sample_text = """
    Machine Learning is a subset of artificial intelligence. It focuses on 
    building systems that can learn from data. The main types include 
    supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised learning uses labeled data. The algorithm learns to map inputs 
    to outputs. Common applications include image classification and spam detection.
    
    Unsupervised learning finds patterns in unlabeled data. Clustering and 
    dimensionality reduction are popular techniques. It's useful for customer 
    segmentation and anomaly detection.
    
    Deep learning uses neural networks with multiple layers. It has revolutionized 
    computer vision, natural language processing, and speech recognition.
    """ * 5  # Repeat to create longer text
    
    clean = preprocessor.clean_text(sample_text)
    chunks = preprocessor.chunk_text(clean)
    
    print(f"\n📊 Test Results:")
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks[:2]:  # Show first 2
        print(f"\nChunk ID: {chunk['id']}")
        print(f"Words: {chunk['word_count']}")
        print(f"Preview: {chunk['text'][:100]}...")