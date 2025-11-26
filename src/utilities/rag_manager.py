"""
RAG Manager for Chat with PDF
Handles vector store creation and similarity search using FAISS
Works with any document type
Uses lazy imports to handle dependency issues gracefully
"""

import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import faiss

load_dotenv()


class RAGManager:
    def __init__(self, model_type: str = "gemini", embedding_method: str = "auto"):
        """
        Initialize RAG Manager
        
        Args:
            model_type: "gemini" or "llama"
            embedding_method: "auto", "sentence-transformers", or "tfidf"
        """
        self.model_type = model_type
        self.embedding_method = embedding_method
        self.chunks = []
        self.model = None
        self.embeddings = None
        self.index = None
        self.document_title = None
        self.document_type = None
        self.embedding_model = None
        
        # Load embedding model (lazy import)
        print(f"📦 Loading embedding model...")
        self._load_embedding_model()
        print(f"✅ Embedding model loaded ({self.embedding_method})")
        
    def _load_embedding_model(self):
        """Load the appropriate embedding model with lazy imports"""
        
        # Auto mode: try sentence-transformers first, fall back to TF-IDF
        if self.embedding_method == "auto":
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_method = "sentence-transformers"
                self.embed_func = self._embed_with_sentence_transformers
                print("  → Using sentence-transformers")
                return
            except Exception as e:
                print(f"  ⚠️ sentence-transformers unavailable: {str(e)[:100]}")
                print("  → Falling back to TF-IDF")
                self.embedding_method = "tfidf"
                self._load_tfidf_model()
                return
        
        # Explicit sentence-transformers
        elif self.embedding_method == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embed_func = self._embed_with_sentence_transformers
            except Exception as e:
                raise ImportError(
                    f"Failed to load sentence-transformers: {e}\n"
                    "Try: pip install --upgrade torch transformers sentence-transformers\n"
                    "Or use embedding_method='tfidf' for a fallback option"
                )
        
        # Explicit TF-IDF
        elif self.embedding_method == "tfidf":
            self._load_tfidf_model()
        
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")
    
    def _load_tfidf_model(self):
        """Load TF-IDF based embedding (fallback option)"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn is required for TF-IDF embeddings.\n"
                "Install it with: pip install scikit-learn"
            )
        
        self.embedding_model = TfidfVectorizer(
            max_features=384,  # Match dimension of sentence-transformers
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.embed_func = self._embed_with_tfidf
        self._tfidf_fitted = False
        
    def _embed_with_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers"""
        return self.embedding_model.encode(texts, show_progress_bar=False)
    
    def _embed_with_tfidf(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using TF-IDF"""
        if not self._tfidf_fitted:
            # First time: fit and transform
            embeddings = self.embedding_model.fit_transform(texts).toarray()
            self._tfidf_fitted = True
        else:
            # Subsequent times: just transform
            embeddings = self.embedding_model.transform(texts).toarray()
        return embeddings.astype('float32')
        
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
    
    def index_documents(self, chunks: List[Dict], document_title: Optional[str] = None):
        """
        Index document chunks for retrieval using FAISS
        
        Args:
            chunks: List of text chunks with metadata
            document_title: Optional title of the document
        """
        self.chunks = chunks
        self.document_title = document_title or "the uploaded document"
        
        if not chunks:
            print("⚠️ No chunks to index")
            return
        
        # Detect document type
        self.document_type = self._detect_document_type(chunks)
        print(f"📄 Detected document type: {self.document_type}")
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        self.embeddings = self.embed_func(texts)
        
        # Ensure embeddings are 2D
        if len(self.embeddings.shape) == 1:
            self.embeddings = self.embeddings.reshape(1, -1)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"✅ Indexed {len(chunks)} chunks with FAISS (dimension: {dimension})")
    
    def _detect_document_type(self, chunks: List[Dict]) -> str:
        """
        Infer document type from content
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Document type as string
        """
        sample_size = min(10, len(chunks))
        combined_text = " ".join([chunk['text'] for chunk in chunks[:sample_size]]).lower()
        
        type_indicators = {
            'mathematics': ['theorem', 'equation', 'proof', 'formula', 'lemma', 'derivative', 'integral'],
            'programming': ['code', 'function', 'algorithm', 'class', 'variable', 'syntax', 'compile'],
            'science': ['experiment', 'hypothesis', 'research', 'data', 'methodology', 'results'],
            'business': ['revenue', 'market', 'strategy', 'customer', 'profit', 'investment'],
            'literature': ['chapter', 'character', 'plot', 'narrative', 'theme', 'author'],
            'legal': ['section', 'clause', 'agreement', 'party', 'hereby', 'pursuant'],
            'medical': ['patient', 'treatment', 'diagnosis', 'symptom', 'clinical', 'therapy'],
            'history': ['century', 'historical', 'period', 'war', 'civilization', 'ancient']
        }
        
        type_scores = {}
        for doc_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                type_scores[doc_type] = score
        
        if type_scores:
            detected_type = max(type_scores, key=type_scores.get)
            return detected_type if type_scores[detected_type] >= 2 else 'general'
        
        return 'general'
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 8) -> Tuple[List[Dict], float]:
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
        query_embedding = self.embed_func([expanded_query])
        
        # Ensure query embedding is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        max_similarity = 0.0
        
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                # Use exponential decay for better similarity scores
                similarity = float(np.exp(-distance / 10))
                chunk['similarity_score'] = similarity
                relevant_chunks.append(chunk)
                max_similarity = max(max_similarity, similarity)
        
        return relevant_chunks, max_similarity
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms for better retrieval"""
        expansions = {
            'importance': 'important significance value benefits advantages key critical essential',
            'summarize': 'summary overview main topics key points outline highlights',
            'explain': 'description explanation details clarify elaborate define',
            'what is': 'definition meaning concept description nature',
            'how': 'process method approach way procedure steps technique',
            'why': 'reason purpose rationale cause motivation explanation',
            'compare': 'difference comparison contrast versus similarities distinctions',
            'advantages': 'benefits pros strengths positive merits',
            'disadvantages': 'drawbacks cons weaknesses negative limitations',
            'features': 'characteristics properties attributes aspects elements',
            'types': 'kinds categories classifications varieties forms',
            'examples': 'instances samples cases illustrations demonstrations'
        }
        
        expanded = query.lower()
        for term, synonyms in expansions.items():
            if term in expanded:
                expanded = expanded + ' ' + synonyms
        
        return expanded
    
    def _get_confidence_level(self, max_similarity: float) -> str:
        """Determine confidence level based on similarity score"""
        if max_similarity > 0.3:
            return "high"
        elif max_similarity > 0.1:
            return "medium"
        else:
            return "low"
    
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
        RELEVANCE_THRESHOLD = 0.05
        
        if not context_chunks or max_similarity < RELEVANCE_THRESHOLD:
            return self._generate_out_of_scope_response(query)
        
        context = "\n\n".join([f"[Section {i+1}]:\n{chunk['text']}" 
                               for i, chunk in enumerate(context_chunks)])
        
        is_summary = any(word in query.lower() for word in [
            'summarize', 'summary', 'overview', 'what is this about', 
            'main topics', 'key points', 'highlights'
        ])
        
        doc_context = f"from {self.document_title}"
        if self.document_type != 'general':
            doc_context += f" (a {self.document_type} document)"
        
        if is_summary:
            prompt = f"""You are an educational AI assistant. The user wants a summary {doc_context}.

DOCUMENT CONTENT:
{context}

USER'S REQUEST:
{query}

INSTRUCTIONS:
- Provide a clear, well-organized summary of the main topics and concepts
- Identify key themes, important points, and central ideas
- Use bullet points or paragraphs as appropriate for clarity
- Be comprehensive but concise
- Focus on the actual content provided in the document
- Organize information logically

SUMMARY:"""
        else:
            prompt = f"""You are an educational AI assistant. Answer the question using content {doc_context}.

DOCUMENT CONTENT:
{context}

USER'S QUESTION:
{query}

INSTRUCTIONS:
- Answer the question clearly using information from the document
- Draw connections between concepts when relevant
- Provide thorough explanations with details from the provided content
- If making inferences, base them on the information given
- Write in a helpful, educational tone
- If the answer requires information not in the provided context, acknowledge this

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
    
    def _generate_out_of_scope_response(self, query: str) -> str:
        """Generate a helpful response for out-of-scope queries"""
        prompt = f"""The user asked: "{query}"

This question appears to be outside the scope of the uploaded document.

Please:
1. Briefly explain what the user is asking about (1-2 sentences)
2. Politely inform them that this topic doesn't appear to be covered in the uploaded document
3. Suggest they ask about topics that are in the current document, or upload a relevant document if they have one

Keep the response helpful, concise, and professional."""
        
        try:
            if self.model_type == "gemini":
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=256,
                    )
                )
                return response.text
            else:
                return f"Your question appears to be about '{query}', but this topic is not covered in {self.document_title}. Please ask about content that is present in the document."
        except Exception as e:
            return f"Your question doesn't seem to be related to the content in {self.document_title}. Please ask about topics covered in the document."
    
    def chat(self, query: str) -> Dict[str, any]:
        """
        Main chat function - retrieves context and generates answer
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with answer, sources, confidence, and relevance info
        """
        relevant_chunks, max_similarity = self.retrieve_relevant_chunks(query, top_k=8)
        
        print(f"\n🔍 Query: {query}")
        print(f"📊 Max Similarity: {max_similarity:.4f}")
        print(f"📚 Retrieved {len(relevant_chunks)} chunks")
        if relevant_chunks:
            print(f"Top 3 chunks:")
            for i, chunk in enumerate(relevant_chunks[:3], 1):
                print(f"  {i}. (sim: {chunk['similarity_score']:.4f}): {chunk['text'][:100]}...")
        
        answer = self.generate_answer(query, relevant_chunks, max_similarity)
        
        is_in_scope = max_similarity >= 0.05
        confidence = self._get_confidence_level(max_similarity)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'chunk_id': chunk['id'],
                    'text': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    'word_count': chunk['word_count'],
                    'similarity': chunk['similarity_score']
                }
                for chunk in relevant_chunks
            ] if is_in_scope else [],
            'num_sources': len(relevant_chunks) if is_in_scope else 0,
            'max_similarity': max_similarity,
            'confidence': confidence,
            'is_in_scope': is_in_scope,
            'document_title': self.document_title,
            'document_type': self.document_type,
            'embedding_method': self.embedding_method
        }
    
    def get_document_info(self) -> Dict[str, any]:
        """Get information about the indexed document"""
        return {
            'title': self.document_title,
            'type': self.document_type,
            'num_chunks': len(self.chunks),
            'indexed': self.index is not None,
            'embedding_method': self.embedding_method
        }

