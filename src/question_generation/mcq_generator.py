"""
Multiple Choice Question (MCQ) generator
"""
import re
from typing import List, Dict
from .prompts import format_mcq_prompt, MCQ_SYSTEM_PROMPT

class MCQGenerator:
    def __init__(self, llama_handler):
        self.llama = llama_handler
    
    def generate_questions(self, 
                          content: str, 
                          num_questions: int = 3,
                          timeout_seconds: int = 30) -> List[Dict]:
        """Generate multiple choice questions from content with timeout"""
        
        # Truncate very long content to prevent timeout
        if len(content) > 1500:
            content = content[:1500] + "..."
        
        prompt = format_mcq_prompt(content, num_questions)
        
        try:
            print(f"Generating {num_questions} MCQs from {len(content)} characters...")
            
            response = self.llama.generate(
                prompt=prompt,
                system_prompt=MCQ_SYSTEM_PROMPT,
                max_tokens=800,  # Reduced for faster generation
                temperature=0.8,  # Slightly higher for more variety
                stop=["</s>", "[INST]", "---"]  # Add stop sequences
            )
            
            questions = self._parse_response(response)
            print(f"[OK] Generated {len(questions)} MCQs successfully")
            return questions
            
        except Exception as e:
            print(f"[ERROR] Error generating MCQs: {str(e)}")
            return self._generate_fallback_mcq(content, num_questions)
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse LLaMA response into structured MCQs"""
        questions = []
        
        # Split by question markers
        parts = re.split(r'(?=Q\d*[:)])', response)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            try:
                # Extract question
                q_match = re.search(r'Q\d*[:)]\s*(.+?)(?=A\))', part, re.DOTALL)
                if not q_match:
                    continue
                
                question_text = q_match.group(1).strip()
                
                # Extract options
                options = {}
                for letter in ['A', 'B', 'C', 'D']:
                    pattern = f'{letter}\\)\\s*(.+?)(?=[A-D]\\)|Correct:|$)'
                    opt_match = re.search(pattern, part, re.DOTALL)
                    if opt_match:
                        options[letter] = opt_match.group(1).strip()
                
                # Extract correct answer
                correct_match = re.search(r'Correct:\s*([A-D])', part, re.IGNORECASE)
                correct_answer = correct_match.group(1).upper() if correct_match else 'A'
                
                # Extract explanation
                exp_match = re.search(r'Explanation:\s*(.+?)(?=Q\d*[:)]|$|---)', part, re.DOTALL)
                explanation = exp_match.group(1).strip() if exp_match else ""
                
                if len(options) >= 4:  # Ensure we have all options
                    questions.append({
                        'type': 'Multiple Choice',
                        'question': question_text,
                        'options': options,
                        'correct_answer': correct_answer,
                        'explanation': explanation,
                        'points': 3
                    })
            
            except Exception as e:
                print(f"Error parsing MCQ: {str(e)}")
                continue
        
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
        """Validate if an MCQ meets quality criteria"""
        q_text = question.get('question', '')
        options = question.get('options', {})
        correct = question.get('correct_answer', '')
        
        # Basic validation
        if len(q_text) < 10:
            return False
        
        # Check if we have 4 options
        if len(options) != 4:
            return False
        
        # Check if correct answer is valid
        if correct not in ['A', 'B', 'C', 'D']:
            return False
        
        # Check if all options have content
        for opt in options.values():
            if len(opt) < 3:
                return False
        
        return True
    
    def _generate_fallback_mcq(self, content: str, num_questions: int) -> List[Dict]:
        """Generate basic MCQ templates when model fails"""
        print("[WARNING] Using fallback MCQ generation")
        
        # Extract key terms for more relevant questions
        words = content.split()
        key_terms = [word for word in words if len(word) > 5 and word.isalpha()][:10]
        
        fallback_questions = []
        
        templates = [
            {
                "question": "What is the main topic discussed in this content?",
                "options": {
                    "A": "The primary subject matter",
                    "B": "A secondary topic",
                    "C": "An unrelated concept", 
                    "D": "Background information"
                },
                "correct_answer": "A",
                "explanation": "The main topic is discussed throughout the content."
            },
            {
                "question": "Which concept is most important in this material?",
                "options": {
                    "A": "A minor detail",
                    "B": "The central concept",
                    "C": "An example used",
                    "D": "A footnote reference"
                },
                "correct_answer": "B", 
                "explanation": "The central concept is the most important element."
            },
            {
                "question": "How would you categorize this information?",
                "options": {
                    "A": "Educational content",
                    "B": "Entertainment material",
                    "C": "Technical documentation",
                    "D": "Personal opinion"
                },
                "correct_answer": "A",
                "explanation": "This represents educational content for learning."
            }
        ]
        
        for i in range(min(num_questions, len(templates))):
            question = templates[i].copy()
            question['type'] = 'Multiple Choice'
            question['points'] = 10
            
            # Customize with key terms if available
            if i < len(key_terms):
                term = key_terms[i]
                question['question'] = f"What role does '{term}' play in this content?"
                question['options']['A'] = f"{term} is the main focus"
                question['options']['B'] = f"{term} is a supporting detail"
                question['options']['C'] = f"{term} is not mentioned"
                question['options']['D'] = f"{term} is briefly noted"
                question['explanation'] = f"The term '{term}' appears in the content and plays a specific role."
            
            fallback_questions.append(question)
        
        return fallback_questions
    
    def shuffle_options(self, question: Dict) -> Dict:
        """Shuffle MCQ options (useful for preventing pattern recognition)"""
        import random
        
        options = question['options']
        correct = question['correct_answer']
        correct_text = options[correct]
        
        # Get all option texts
        option_texts = list(options.values())
        random.shuffle(option_texts)
        
        # Reassign to letters
        new_options = {}
        letters = ['A', 'B', 'C', 'D']
        new_correct = None
        
        for i, text in enumerate(option_texts):
            new_options[letters[i]] = text
            if text == correct_text:
                new_correct = letters[i]
        
        question['options'] = new_options
        question['correct_answer'] = new_correct
        
        return question