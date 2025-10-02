"""
Text cleaning and preprocessing module
"""
import re
import nltk
from typing import List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class TextCleaner:
    def __init__(self):
        self.patterns_to_remove = [
            r'©.*?\d{4}',  # Copyright notices
            r'Page \d+',  # Page headers
            r'\b(Chapter|Section|Unit)\s+\d+\b',  # Chapter/Section headers (optional)
        ]
    
    def clean(self, text: str) -> str:
        """Clean extracted text"""
        # Remove patterns
        for pattern in self.patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 20]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract potential key terms (simple approach)"""
        # Extract capitalized words (potential named entities/key terms)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Remove duplicates and common words
        common_words = {'The', 'A', 'An', 'This', 'That', 'These', 'Those'}
        key_terms = list(set([w for w in words if w not in common_words]))
        return key_terms[:20]  # Return top 20