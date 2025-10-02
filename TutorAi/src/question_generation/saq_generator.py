"""
Short Answer Question (SAQ) generator using Transformers
"""
import re
from typing import List, Dict
from .prompts import format_saq_prompt, SAQ_SYSTEM_PROMPT
from .transformers_handler import TransformersHandler

class SAQGenerator:
    def __init__(self, model_handler=None):
        self.model = None
        self.llama = None
        self.fallback_questions = []
        
        # Try to initialize transformers first
        try:
            self.model = TransformersHandler()
            print("[OK] SAQ Generator: Using Transformers model")
        except Exception as e:
            print(f"[ERROR] SAQ Generator: Transformers failed - {str(e)}")
            
            # Fallback to provided handler (llama) if available
            if model_handler:
                self.llama = model_handler
                print("[OK] SAQ Generator: Using LLaMA fallback")
            else:
                print("[WARNING] SAQ Generator: No models available, will use template questions")
                self._initialize_fallback_questions()
    
    def generate_questions(self, 
                          content: str, 
                          num_questions: int = 3) -> List[Dict]:
        """Generate short answer questions from content with fallback system"""
        
        # Try transformers first
        if self.model:
            try:
                print("Attempting question generation with Transformers...")
                response = self._generate_with_transformers(content, num_questions)
                print(f"[DEBUG] Transformers raw response: {response[:200]}...")
                questions = self._parse_response(response)
                if questions and len(questions) > 0:
                    print(f"[OK] Generated {len(questions)} questions with Transformers")
                    # Check if answers are generic
                    generic_count = sum(1 for q in questions if "refer to" in q.get('answer', '').lower())
                    if generic_count > 0:
                        print(f"[WARNING] {generic_count} questions have generic answers, trying fallback...")
                        raise Exception("Generic answers detected")
                    return questions
                else:
                    print("[WARNING] Transformers generated empty questions, trying fallback...")
            except Exception as e:
                print(f"[ERROR] Transformers generation failed: {str(e)}, trying fallback...")
        
        # Try LLaMA if available
        if self.llama:
            try:
                print("Attempting question generation with LLaMA...")
                prompt = format_saq_prompt(content, num_questions)
                response = self.llama.generate(
                    prompt=prompt,
                    system_prompt=SAQ_SYSTEM_PROMPT,
                    max_tokens=800,
                    temperature=0.7
                )
                questions = self._parse_response(response)
                if questions and len(questions) > 0:
                    print(f"[OK] Generated {len(questions)} questions with LLaMA")
                    return questions
                else:
                    print("[WARNING] LLaMA generated empty questions, using template fallback...")
            except Exception as e:
                print(f"[ERROR] LLaMA generation failed: {str(e)}, using template fallback...")
        
        # Ultimate fallback: template questions
        print("Using template questions as final fallback...")
        return self._generate_fallback_questions(content, num_questions)
    
    def _generate_with_transformers(self, content: str, num_questions: int) -> str:
        """Generate questions using transformers model"""
        
        # Create a focused prompt for question generation with actual content-based answers
        prompt = f"""Based on this educational content, create {num_questions} short answer questions with complete answers:

--- CONTENT ---
{content[:800]}
--- END CONTENT ---

Instructions:
1. Create questions that test understanding of the key concepts
2. Provide complete, factual answers based ONLY on the content above
3. Each answer should be 1-2 sentences with specific information from the text
4. Format exactly as shown below:

Q1: What is [specific concept from content]?
A1: [Complete answer with facts from the content]

Q2: How does [another concept] work?
A2: [Complete answer with facts from the content]

Begin:
Q1:"""
        
        response = self.model.generate(
            prompt=prompt,
            max_tokens=600,
            temperature=0.7,
            top_p=0.9
        )
        
        # Ensure we have the Q1: prefix for parsing
        if not response.startswith("Q"):
            response = "Q1:" + response
            
        return response
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse model response into structured questions with improved parsing"""
        questions = []
        
        # Clean up the response
        response = response.strip()
        
        # Split by question markers (more flexible patterns)
        parts = re.split(r'(?=Q\d*[:)]\s*)', response)
        
        for part in parts:
            part = part.strip()
            if not part or len(part) < 10:
                continue
            
            # Extract question with more flexible patterns
            q_match = re.search(r'Q\d*[:)]\s*(.+?)(?=\n\s*A\d*[:)]|A\d*[:)]|$)', part, re.DOTALL)
            # Extract answer with more flexible patterns
            a_match = re.search(r'A\d*[:)]\s*(.+?)(?=\n\s*Q\d*[:)]|Q\d*[:)]|$|---)', part, re.DOTALL)
            
            if q_match:
                question_text = q_match.group(1).strip()
                # Clean up question text
                question_text = re.sub(r'\n+', ' ', question_text).strip()
                
                # Get answer text
                if a_match:
                    answer_text = a_match.group(1).strip()
                    # Clean up answer text
                    answer_text = re.sub(r'\n+', ' ', answer_text).strip()
                    # Remove any trailing question markers
                    answer_text = re.sub(r'Q\d*[:)].+$', '', answer_text).strip()
                else:
                    answer_text = "Answer not available in the generated response"
                
                # Validate that we have meaningful content
                if len(question_text) > 5 and len(answer_text) > 5:
                    # Ensure question ends with question mark
                    if not question_text.endswith('?'):
                        question_text += '?'
                    
                    # Ensure answer ends with period
                    if not answer_text.endswith('.') and not answer_text.endswith('!'):
                        answer_text += '.'
                    
                    questions.append({
                        'type': 'Short Answer',
                        'question': question_text,
                        'answer': answer_text,
                        'points': 5
                    })
        
        return questions
    
    def generate_from_chunks(self, 
                            chunks: List[str], 
                            questions_per_chunk: int = 2) -> List[Dict]:
        """Generate questions from multiple text chunks"""
        all_questions = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            questions = self.generate_questions(chunk, questions_per_chunk)
            all_questions.extend(questions)
        
        return all_questions
    
    def validate_question(self, question: Dict) -> bool:
        """Validate if a question meets quality criteria"""
        q_text = question.get('question', '')
        a_text = question.get('answer', '')
        
        # Basic validation
        if len(q_text) < 10 or len(a_text) < 10:
            return False
        
        # Check if question ends with question mark
        if not q_text.strip().endswith('?'):
            return False
        
        # Check if answer is substantial
        if len(a_text.split()) < 5:
            return False
        
        return True
    
    def _initialize_fallback_questions(self):
        """Initialize template questions for when no models are available"""
        self.fallback_questions = [
            {
                'type': 'Short Answer',
                'question': 'What are the main concepts discussed in this content?',
                'answer': 'Please refer to the key points mentioned in the text.',
                'points': 5
            },
            {
                'type': 'Short Answer', 
                'question': 'Explain the significance of the topics covered.',
                'answer': 'The content covers important foundational concepts.',
                'points': 5
            },
            {
                'type': 'Short Answer',
                'question': 'What practical applications can be derived from this material?',
                'answer': 'Various applications can be identified based on the content.',
                'points': 5
            }
        ]
    
    def _generate_fallback_questions(self, content: str, num_questions: int) -> List[Dict]:
        """Generate questions with actual content-based answers when models fail"""
        questions = []
        
        # Extract meaningful sentences from content
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        
        # Extract key terms and their context
        import re
        words = content.split()
        key_terms = [word.strip('.,!?()') for word in words if len(word) > 6 and word.isalpha()]
        
        # Create more intelligent questions based on content
        for i in range(min(num_questions, 5)):
            if i < len(sentences) and sentences[i]:
                sentence = sentences[i][:200]  # Limit length
                
                # Extract key concepts from the sentence
                if i < len(key_terms):
                    term = key_terms[i]
                    
                    # Try to find sentences containing the term for better answers
                    answer_sentences = [s for s in sentences if term.lower() in s.lower()]
                    if answer_sentences:
                        answer = answer_sentences[0][:150].strip()
                        if not answer.endswith('.'):
                            answer += '.'
                    else:
                        answer = sentence[:100].strip()
                        if not answer.endswith('.'):
                            answer += '.'
                    
                    question = f"What is {term} and how is it defined in the context?"
                else:
                    # Create questions from content structure
                    if 'definition' in sentence.lower() or 'is' in sentence.lower():
                        # Extract the subject being defined
                        words_in_sent = sentence.split()
                        if len(words_in_sent) > 3:
                            subject = words_in_sent[0] if len(words_in_sent[0]) > 3 else words_in_sent[1]
                            question = f"How is {subject} characterized or defined?"
                            answer = sentence[:120].strip()
                            if not answer.endswith('.'):
                                answer += '.'
                        else:
                            question = "What are the key concepts discussed?"
                            answer = sentence[:100].strip() + '.'
                    else:
                        question = f"What information is provided about the main topic?"
                        answer = sentence[:100].strip()
                        if not answer.endswith('.'):
                            answer += '.'
            else:
                # Final fallback with generic but content-aware questions
                question = f"What are the main points covered in this section?"
                answer = "The content discusses various concepts and their relationships as outlined in the material."
            
            questions.append({
                'type': 'Short Answer',
                'question': question,
                'answer': answer,
                'points': 5
            })
        
        return questions
