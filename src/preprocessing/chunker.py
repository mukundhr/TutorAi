"""
Text chunking module for processing large documents
"""
from typing import List
import sys
sys.path.append('..')
import config

class TextChunker:
    def __init__(self, 
                 chunk_size: int = config.MAX_CHUNK_SIZE,
                 overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, sentences: List[str]) -> List[str]:
        """Chunk text by sentences with overlap"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep overlap sentences
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > self.overlap:
                    # Find sentences that fit in overlap
                    overlap_chunk = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.overlap:
                            overlap_chunk.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Filter out very small chunks
        chunks = [c for c in chunks if len(c) >= config.MIN_CHUNK_SIZE]
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk text by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return [c for c in chunks if len(c) >= config.MIN_CHUNK_SIZE]
