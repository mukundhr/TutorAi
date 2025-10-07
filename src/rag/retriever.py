"""
Document retriever with multiple strategies
"""
from typing import List, Dict
from .vector_store import VectorStore
import re
import sys
sys.path.append('..')
import config

class DocumentRetriever:
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store
        
    def retrieve_for_question(self, 
                             topic: str, 
                             top_k: int = None,
                             min_score: float = None) -> List[str]:
        """Retrieve relevant context for question generation"""
        if not self.vector_store:
            return []
        
        top_k = top_k or config.RAG_TOP_K
        min_score = min_score or config.RAG_MIN_SCORE
        
        results = self.vector_store.search(topic, top_k=top_k)
        
        # Filter by minimum score
        relevant_texts = [
            text for text, score, meta in results 
            if score >= min_score
        ]
        
        return relevant_texts
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Extract capitalized phrases (potential concepts)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Remove common words (expanded list)
        common_words = {
            'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'It', 'In', 'On', 'At',
            'For', 'With', 'From', 'To', 'By', 'As', 'Is', 'Are', 'Was', 'Were',
            'Be', 'Been', 'Being', 'Have', 'Has', 'Had', 'Do', 'Does', 'Did',
            'Will', 'Would', 'Should', 'Could', 'May', 'Might', 'Must', 'Can',
            'However', 'Therefore', 'Thus', 'Hence', 'Also', 'Furthermore'
        }
        concepts = [c for c in concepts if c not in common_words and len(c) > 2]
        
        # Prioritize longer, more specific concepts
        concepts.sort(key=lambda x: len(x), reverse=True)
        
        return list(set(concepts))[:5]  # Top 5 unique concepts (reduced for focus)