"""
RAG Manager for Chat with PDF
Handles vector store creation and similarity search using FAISS
"""

import os
from typing import List, Dict
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()


class RAGManager:
    def __init__(self, model_type: str = "gemini"):
        """
        Initialize RAG Manager
        
        Args:
            model_type: "gemini" or "llama"
        """
        self.model_type = model_type
        self.chunks = []
        self.model = None
        self.embeddings = None
        self.index = None
        
        # Load embedding model
        print("📦 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and good quality
        print("✅ Embedding model loaded")
        
    def load_model(self):
        """Load the LLM model"""
        if self.model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in .env file")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini loaded for RAG")
        else:
            # TODO: Add LLaMA support if needed
            raise NotImplementedError("LLaMA RAG not yet implemented")
    
    def index_documents(self, chunks: List[Dict]):
        """
        Index document chunks for retrieval using FAISS
        
        Args:
            chunks: List of text chunks with metadata
        """
        self.chunks = chunks
        
        if not chunks:
            print("⚠️ No chunks to index")
            return
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ Indexed {len(chunks)} chunks with FAISS")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant chunks for a query using semantic search (FAISS)
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not self.chunks or self.index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Safety check
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generate answer using retrieved context
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer
        """
        if not context_chunks:
            return "I couldn't find relevant information in the document to answer this question. Please try rephrasing or ask about topics covered in the PDF."
        
        # Combine context
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Build prompt
        prompt = f"""You are a helpful educational assistant. Answer the user's question based ONLY on the provided context from the document.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer the question using only information from the context above
- Be clear, concise, and educational
- If the context doesn't contain enough information, say so
- Use examples from the context when helpful
- Format your answer in a clear, readable way

ANSWER:"""
        
        try:
            if self.model_type == "gemini":
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
                return response.text
            else:
                return "Model not supported for RAG yet"
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def chat(self, query: str) -> Dict[str, any]:
        """
        Main chat function - retrieves context and generates answer
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with answer and sources
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=3)
        
        # Generate answer
        answer = self.generate_answer(query, relevant_chunks)
        
        # Return response with sources
        return {
            'answer': answer,
            'sources': [
                {
                    'chunk_id': chunk['id'],
                    'text': chunk['text'][:200] + "...",  # Preview
                    'word_count': chunk['word_count']
                }
                for chunk in relevant_chunks
            ],
            'num_sources': len(relevant_chunks)
        }


# Example usage
if __name__ == "__main__":
    rag = RAGManager(model_type="gemini")
    rag.load_model()
    
    # Test chunks
    test_chunks = [
        {
            'id': 'chunk_000',
            'text': 'Photosynthesis is the process by which plants convert light energy into chemical energy.',
            'word_count': 13
        },
        {
            'id': 'chunk_001',
            'text': 'Chlorophyll is the green pigment in plants that absorbs sunlight during photosynthesis.',
            'word_count': 13
        }
    ]
    
    rag.index_documents(test_chunks)
    
    response = rag.chat("What is photosynthesis?")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {response['num_sources']}")
