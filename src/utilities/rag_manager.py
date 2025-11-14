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
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> tuple[List[Dict], float]:
        """
        Retrieve most relevant chunks for a query using semantic search (FAISS)
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (List of relevant chunks with similarity scores, max_similarity_score)
        """
        if not self.chunks or self.index is None:
            return [], 0.0
        
        # Expand query for better matching
        expanded_query = self._expand_query(query)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([expanded_query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        max_similarity = 0.0
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Safety check
                chunk = self.chunks[idx].copy()
                # Use exponential decay for better similarity scores
                similarity = float(np.exp(-distance / 10))  # Better scoring than 1/(1+d)
                chunk['similarity_score'] = similarity
                relevant_chunks.append(chunk)
                max_similarity = max(max_similarity, similarity)
        
        return relevant_chunks, max_similarity
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms for better retrieval
        """
        # Add common variations
        expansions = {
            'importance': 'important significance value benefits advantages',
            'summarize': 'summary overview main topics key points',
            'explain': 'description explanation details',
            'what is': 'definition meaning concept',
            'how': 'process method approach way',
            'why': 'reason purpose rationale'
        }
        
        expanded = query.lower()
        for term, synonyms in expansions.items():
            if term in expanded:
                expanded = expanded + ' ' + synonyms
        
        return expanded
    
    def generate_answer(self, query: str, context_chunks: List[Dict], max_similarity: float) -> str:
        """
        Generate answer using retrieved context
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            max_similarity: Highest similarity score from retrieval
            
        Returns:
            Generated answer
        """
        # Define similarity threshold (0.0 to 1.0) - lowered significantly
        RELEVANCE_THRESHOLD = 0.05  # Very low threshold with exponential scoring
        
        # Check if query is related to the document
        if not context_chunks or max_similarity < RELEVANCE_THRESHOLD:
            # Query is out of scope - provide helpful explanation
            return self._generate_out_of_scope_response(query)
        
        # Combine context from all chunks
        context = "\n\n".join([f"[From Document Section {i+1}]:\n{chunk['text']}" 
                               for i, chunk in enumerate(context_chunks)])
        
        # Detect if this is a summary request
        is_summary = any(word in query.lower() for word in ['summarize', 'summary', 'overview', 'what is this about', 'main topics'])
        
        # Build prompt - adapt based on query type
        if is_summary:
            prompt = f"""You are an educational AI assistant. The user wants a summary of the document about Software Engineering.

DOCUMENT CONTENT:
{context}

USER'S REQUEST:
{query}

INSTRUCTIONS:
- Provide a clear, well-organized summary of the main topics and concepts in this Software Engineering document
- Identify key themes: process models, software engineering principles, methodologies, etc.
- Use bullet points or paragraphs as appropriate
- Be comprehensive but concise
- Focus on the actual educational content about software engineering

SUMMARY:"""
        else:
            prompt = f"""You are an educational AI assistant helping with Software Engineering concepts. Answer the question using the document content provided.

DOCUMENT CONTENT:
{context}

USER'S QUESTION:
{query}

INSTRUCTIONS:
- Answer the question clearly using the information from the document
- This is a Software Engineering document covering process models, methodologies, and practices
- Draw connections between concepts when relevant
- Provide thorough explanations with details from the document
- If making inferences, base them on the provided information
- Write in a helpful, educational tone suitable for students

ANSWER:"""
        
        try:
            if self.model_type == "gemini":
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.8,  # Increased for better synthesis
                        max_output_tokens=1024,
                    )
                )
                return response.text
            else:
                return "Model not supported for RAG yet"
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _generate_out_of_scope_response(self, query: str) -> str:
        """
        Generate a helpful response for out-of-scope queries
        
        Args:
            query: User's question
            
        Returns:
            Explanation of what the query is about and that it's not in the PDF
        """
        prompt = f"""The user asked: "{query}"

This question appears to be outside the scope of the uploaded document.

Please:
1. Briefly explain what the user is asking about (1-2 sentences)
2. Politely inform them that this topic is not covered in the uploaded PDF
3. Suggest they upload a relevant document or ask about topics in the current document

Keep the response helpful, concise, and professional."""
        
        try:
            if self.model_type == "gemini":
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.8,
                        max_output_tokens=256,
                    )
                )
                return response.text
            else:
                return f"Your question appears to be about '{query}', but this topic is not covered in the uploaded PDF. Please ask about content that is present in the document."
        except Exception as e:
            return "Your question doesn't seem to be related to the content in the uploaded PDF. Please ask about topics covered in the document."
    
    def chat(self, query: str) -> Dict[str, any]:
        """
        Main chat function - retrieves context and generates answer
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with answer, sources, and relevance info
        """
        # Retrieve relevant chunks with similarity scores (increased to 8 for better context)
        relevant_chunks, max_similarity = self.retrieve_relevant_chunks(query, top_k=8)
        
        # Debug: Print retrieved chunks
        print(f"\n🔍 Query: {query}")
        print(f"📊 Max Similarity: {max_similarity:.4f}")
        print(f"📚 Retrieved {len(relevant_chunks)} chunks:")
        for i, chunk in enumerate(relevant_chunks[:3], 1):  # Show first 3
            print(f"  Chunk {i} (sim: {chunk['similarity_score']:.4f}): {chunk['text'][:100]}...")
        
        # Generate answer (handles both in-scope and out-of-scope queries)
        answer = self.generate_answer(query, relevant_chunks, max_similarity)
        
        # Determine if query was in-scope
        is_in_scope = max_similarity >= 0.05  # Same threshold as in generate_answer
        
        # Return response with sources
        return {
            'answer': answer,
            'sources': [
                {
                    'chunk_id': chunk['id'],
                    'text': chunk['text'][:200] + "...",  # Preview
                    'word_count': chunk['word_count'],
                    'similarity': chunk['similarity_score']
                }
                for chunk in relevant_chunks
            ] if is_in_scope else [],  # Don't show sources for out-of-scope queries
            'num_sources': len(relevant_chunks) if is_in_scope else 0,
            'max_similarity': max_similarity,
            'is_in_scope': is_in_scope
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
