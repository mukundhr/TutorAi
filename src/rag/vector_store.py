"""
Vector store for document embeddings using FAISS
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import sys
sys.path.append('..')
import config

class VectorStore:
    def __init__(self, model_name: str = None):
        """Initialize vector store with sentence transformer"""
        model_name = model_name or config.RAG_EMBEDDING_MODEL
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.texts = []
        self.metadata = []
        
    def create_index(self, texts: List[str], metadata: List[Dict] = None):
        """Create FAISS index from text chunks"""
        print(f"🔄 Creating embeddings for {len(texts)} chunks...")
        
        self.texts = texts
        self.metadata = metadata or [{'index': i} for i in range(len(texts))]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ Vector index created with {self.index.ntotal} vectors")
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """Search for similar chunks"""
        if self.index is None:
            raise ValueError("Index not created")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append((
                    self.texts[idx],
                    float(score),
                    self.metadata[idx]
                ))
        
        return results
    
    def save(self, path: str):
        """Save vector store"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        with open(save_path / "data.pkl", 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata
            }, f)
        
        print(f"✓ Vector store saved to {path}")
    
    def load(self, path: str):
        """Load vector store"""
        load_path = Path(path)
        
        self.index = faiss.read_index(str(load_path / "index.faiss"))
        
        with open(load_path / "data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
        
        print(f"✓ Vector store loaded from {path}")