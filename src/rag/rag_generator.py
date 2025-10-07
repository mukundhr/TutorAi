"""
RAG-enhanced question generator
"""
from typing import List, Dict
from .retriever import DocumentRetriever
import re

class RAGQuestionGenerator:
    def __init__(self, ai_handler, retriever: DocumentRetriever):
        self.ai = ai_handler
        self.retriever = retriever
        
    def generate_saq_with_rag(self, 
                              chunk: str, 
                              num_questions: int = 3) -> List[Dict]:
        """Generate SAQ with retrieved context"""
        # Extract key concepts
        concepts = self.retriever.extract_key_concepts(chunk)
        
        # Retrieve additional context (only if we have strong concepts)
        additional_context = []
        if concepts and len(concepts) >= 2:
            # Use top 2 most important concepts
            additional_context = self.retriever.retrieve_for_question(
                ' '.join(concepts[:2]), 
                top_k=2
            )
        
        # Build prompt
        prompt = self._build_saq_prompt(chunk, additional_context, num_questions)
        system_prompt = "You are an expert educational content creator specializing in generating high-quality short answer questions."
        
        # Generate
        response = self.ai.generate(prompt=prompt, system_prompt=system_prompt)
        
        # Parse
        questions = self._parse_saq_response(response)
        return questions
    
    def generate_mcq_with_rag(self, 
                              chunk: str, 
                              num_questions: int = 3) -> List[Dict]:
        """Generate MCQ with RAG-enhanced distractors"""
        concepts = self.retriever.extract_key_concepts(chunk)
        
        # Retrieve context for better distractors (only with strong concepts)
        additional_context = []
        if concepts and len(concepts) >= 2:
            # Use top 2 concepts for focused retrieval
            additional_context = self.retriever.retrieve_for_question(
                ' '.join(concepts[:2]), 
                top_k=2
            )
        
        prompt = self._build_mcq_prompt(chunk, additional_context, num_questions)
        system_prompt = "You are an expert educational assessment designer creating challenging multiple-choice questions."
        
        response = self.ai.generate(prompt=prompt, system_prompt=system_prompt)
        questions = self._parse_mcq_response(response)
        
        return questions
    
    def _build_saq_prompt(self, main_chunk: str, context_chunks: List[str], num_questions: int) -> str:
        """Build RAG-enhanced SAQ prompt"""
        context_text = ""
        if context_chunks:
            context_text = "\n\nRelated Information (use ONLY if directly relevant):\n" + "\n---\n".join(context_chunks)
        
        prompt = f"""Based on the following lecture content, generate {num_questions} high-quality short answer questions.

Main Content (PRIMARY SOURCE):
{main_chunk}
{context_text}

Requirements:
- Focus PRIMARILY on the Main Content above
- Questions should require 2-4 sentence answers
- Test understanding of key concepts from the main content
- Only use Related Information if it directly enhances the question
- Be clear, specific, and avoid ambiguity
- Questions should be self-contained and answerable from the content

Format each question exactly as:
Q: [Your question here]
A: [Expected answer in 2-4 sentences]

---

Generate the questions now:"""
        
        return prompt
    
    def _build_mcq_prompt(self, main_chunk: str, context_chunks: List[str], num_questions: int) -> str:
        """Build RAG-enhanced MCQ prompt"""
        context_text = ""
        if context_chunks:
            context_text = "\n\nRelated Concepts (use ONLY for creating plausible distractors):\n" + "\n---\n".join(context_chunks)
        
        prompt = f"""Based on the following lecture content, generate {num_questions} high-quality multiple choice question(s) with 4 options each.

Main Content (PRIMARY SOURCE):
{main_chunk}
{context_text}

Requirements:
- Question and CORRECT answer must come from Main Content
- Create challenging but fair questions that test understanding
- Include 1 correct answer and 3 plausible distractors
- Distractors should be reasonable but clearly incorrect
- You may use Related Concepts to create realistic distractors
- All options should be similar in length and complexity
- Avoid trick questions or ambiguous wording

Format each question exactly as:
Q: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]
Explanation: [Brief explanation why the correct answer is right]

---

Generate the question(s) now:"""
        
        return prompt
    
    def _parse_saq_response(self, response: str) -> List[Dict]:
        """Parse SAQ response"""
        questions = []
        parts = re.split(r'(?=Q\d*[:)])', response)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            q_match = re.search(r'Q\d*[:)]\s*(.+?)(?=A\d*[:)]|$)', part, re.DOTALL)
            a_match = re.search(r'A\d*[:)]\s*(.+?)(?=Q\d*[:)]|$|---)', part, re.DOTALL)
            
            if q_match:
                question_text = q_match.group(1).strip()
                answer_text = a_match.group(1).strip() if a_match else "Answer not provided"
                
                questions.append({
                    'type': 'Short Answer',
                    'question': question_text,
                    'answer': answer_text,
                    'points': 5,
                    'rag_enhanced': True
                })
        
        return questions
    
    def _parse_mcq_response(self, response: str) -> List[Dict]:
        """Parse MCQ response"""
        questions = []
        parts = re.split(r'(?=Q\d*[:)])', response)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            try:
                q_match = re.search(r'Q\d*[:)]\s*(.+?)(?=A\))', part, re.DOTALL)
                if not q_match:
                    continue
                
                question_text = q_match.group(1).strip()
                
                options = {}
                for letter in ['A', 'B', 'C', 'D']:
                    pattern = f'{letter}\\)\\s*(.+?)(?=[A-D]\\)|Correct:|$)'
                    opt_match = re.search(pattern, part, re.DOTALL)
                    if opt_match:
                        options[letter] = opt_match.group(1).strip()
                
                correct_match = re.search(r'Correct:\s*([A-D])', part, re.IGNORECASE)
                correct_answer = correct_match.group(1).upper() if correct_match else 'A'
                
                exp_match = re.search(r'Explanation:\s*(.+?)(?=Q\d*[:)]|$|---)', part, re.DOTALL)
                explanation = exp_match.group(1).strip() if exp_match else ""
                
                if len(options) >= 4:
                    questions.append({
                        'type': 'Multiple Choice',
                        'question': question_text,
                        'options': options,
                        'correct_answer': correct_answer,
                        'explanation': explanation,
                        'points': 3,
                        'rag_enhanced': True
                    })
            
            except Exception as e:
                print(f"Error parsing MCQ: {str(e)}")
                continue
        
        return questions