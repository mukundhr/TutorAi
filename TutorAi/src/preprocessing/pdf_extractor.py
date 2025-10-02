"""
PDF text extraction module
"""
import pdfplumber
import re
from typing import List, Dict

class PDFExtractor:
    def __init__(self):
        self.text = ""
        self.metadata = {}
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        extracted_text = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                self.metadata = {
                    "num_pages": len(pdf.pages),
                    "title": pdf.metadata.get('Title', 'Unknown')
                }
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Basic cleaning
                        text = self._clean_page_text(text)
                        extracted_text.append(text)
                
                self.text = "\n\n".join(extracted_text)
                return self.text
                
        except Exception as e:
            raise Exception(f"Error extracting PDF: {str(e)}")
    
    def _clean_page_text(self, text: str) -> str:
        """Clean individual page text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers (simple pattern)
        text = re.sub(r'\n\d+\n', '\n', text)
        return text.strip()
    
    def get_metadata(self) -> Dict:
        """Get PDF metadata"""
        return self.metadata