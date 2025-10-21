"""
Unified LLM Manager for both MCQ and SAQ generation
Supports: Google Gemini 2.0 Flash and LLaMA 2 7B
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Gemini
import google.generativeai as genai

# LLaMA
from llama_cpp import Llama

load_dotenv()


class LLMManager:
    def __init__(self, model_type: str = "gemini"):
        """
        Initialize LLM Manager
        
        Args:
            model_type: "gemini" or "llama"
        """
        self.model_type = model_type
        self.model = None
        
    def load_model(self):
        """Load the selected model"""
        if self.model_type == "gemini":
            self._load_gemini()
        elif self.model_type == "llama":
            self._load_llama()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_gemini(self):
        """Load Google Gemini"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini 2.0 Flash loaded")
    
    def _load_llama(self):
        """Load LLaMA 2 7B"""
        model_path = os.getenv("LLAMA_MODEL_PATH", "./models/llama-2-7b-chat.Q4_K_M.gguf")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLaMA model not found at {model_path}")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # CPU threads
            n_gpu_layers=0  # Set to 35 if you have GPU
        )
        print("✅ LLaMA 2 7B loaded")
    
    def generate_questions(
        self,
        text: str,
        question_type: str,
        num_questions: int = 5,
        difficulty: str = "medium",
        temperature: float = 0.7
    ) -> List[Dict]:
        """
        Generate questions from text
        
        Args:
            text: Source text chunk
            question_type: "mcq" or "saq"
            num_questions: Number of questions to generate
            difficulty: "easy", "medium", or "hard"
            temperature: Sampling temperature (0.0-1.0)
        
        Returns:
            List of question dictionaries
        """
        prompt = self._build_prompt(text, question_type, num_questions, difficulty)
        
        if self.model_type == "gemini":
            response = self._generate_gemini(prompt, temperature)
        else:
            response = self._generate_llama(prompt, temperature)
        
        # Parse response into structured questions
        questions = self._parse_response(response, question_type)
        
        # Validate we got the requested number
        if len(questions) != num_questions:
            print(f"⚠️ Expected {num_questions} {question_type.upper()}s, got {len(questions)}")
            
            # If we got more, trim to exact number
            if len(questions) > num_questions:
                questions = questions[:num_questions]
                print(f"✂️ Trimmed to {num_questions} questions")
            
            # If we got fewer, warn user
            elif len(questions) < num_questions:
                print(f"⚠️ Only generated {len(questions)}/{num_questions} questions")
        
        return questions
    
    def _build_prompt(self, text: str, question_type: str, num_questions: int, difficulty: str) -> str:
        """Build prompt for question generation"""
        
        # Define difficulty guidelines
        difficulty_guide = {
            "easy": """
DIFFICULTY - EASY:
- Direct recall and basic comprehension
- Test definitions, terminology, and simple concepts
- Answer should be clearly stated in the text
- Minimal inference required""",
            "medium": """
DIFFICULTY - MEDIUM:
- Application and understanding of concepts
- Require connecting multiple ideas
- Some inference and analysis needed
- Test relationships between concepts""",
            "hard": """
DIFFICULTY - HARD:
- Critical thinking and deep analysis
- Synthesis of multiple concepts
- Require evaluation and complex reasoning
- Apply concepts to new situations
- May involve edge cases or exceptions"""
        }
        
        if question_type == "mcq":
            return f"""You are an expert educator creating multiple choice questions.

SOURCE TEXT:
{text}

TASK: Generate EXACTLY {num_questions} multiple choice questions at {difficulty.upper()} difficulty level.

{difficulty_guide[difficulty]}

REQUIREMENTS:
- Generate EXACTLY {num_questions} questions (no more, no less)
- Each question must have exactly 4 options (A, B, C, D)
- Only one correct answer
- Distractors should be plausible but incorrect
- Include a brief explanation for the correct answer
- Match the {difficulty} difficulty level specified above

FORMAT (use exactly this structure):
Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
ANSWER: [Correct letter]
EXPLANATION: [Why this is correct]

Q2: [Next question...]

Generate EXACTLY {num_questions} questions now:"""
        
        else:  # SAQ
            return f"""You are an expert educator creating short answer questions.

SOURCE TEXT:
{text}

TASK: Generate EXACTLY {num_questions} short answer questions at {difficulty.upper()} difficulty level.

{difficulty_guide[difficulty]}

REQUIREMENTS:
- Generate EXACTLY {num_questions} questions (no more, no less)
- Questions should require detailed explanations
- Test conceptual understanding and application
- Be specific and clear
- Include a comprehensive model answer (4-5 lines/sentences) for each question
- Answers should be thorough but concise
- Match the {difficulty} difficulty level specified above

FORMAT (use exactly this structure):
Q1: [Question text]
ANSWER: [Model answer - write 4-5 complete sentences explaining the concept thoroughly]

Q2: [Next question...]

Generate EXACTLY {num_questions} questions now:"""
    
    def _generate_gemini(self, prompt: str, temperature: float) -> str:
        """Generate using Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return ""
    
    def _generate_llama(self, prompt: str, temperature: float) -> str:
        """Generate using LLaMA"""
        try:
            response = self.model(
                prompt,
                max_tokens=2048,
                temperature=temperature,
                top_p=0.9,
                echo=False
            )
            return response['choices'][0]['text']
        except Exception as e:
            print(f"❌ LLaMA error: {e}")
            return ""
    
    def _parse_response(self, response: str, question_type: str) -> List[Dict]:
        """Parse LLM response into structured questions"""
        questions = []
        
        if not response:
            return questions
        
        # Split by question markers
        parts = response.split('\n\n')
        
        for part in parts:
            if not part.strip() or not part.startswith('Q'):
                continue
            
            try:
                if question_type == "mcq":
                    q = self._parse_mcq(part)
                else:
                    q = self._parse_saq(part)
                
                if q:
                    questions.append(q)
            except Exception as e:
                print(f"⚠️ Parse error: {e}")
                continue
        
        return questions
    
    def _parse_mcq(self, text: str) -> Optional[Dict]:
        """Parse MCQ from text"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if len(lines) < 6:  # Need Q, A, B, C, D, ANSWER minimum
            return None
        
        question = lines[0].split(':', 1)[1].strip() if ':' in lines[0] else lines[0]
        
        options = {}
        answer = ""
        explanation = ""
        
        for line in lines[1:]:
            if line.startswith(('A)', 'A.')):
                options['A'] = line[2:].strip()
            elif line.startswith(('B)', 'B.')):
                options['B'] = line[2:].strip()
            elif line.startswith(('C)', 'C.')):
                options['C'] = line[2:].strip()
            elif line.startswith(('D)', 'D.')):
                options['D'] = line[2:].strip()
            elif line.startswith('ANSWER:'):
                answer = line.split(':', 1)[1].strip()[0]  # Get first letter
            elif line.startswith('EXPLANATION:'):
                explanation = line.split(':', 1)[1].strip()
        
        if len(options) == 4 and answer:
            return {
                'question': question,
                'options': options,
                'correct_answer': answer,
                'explanation': explanation,
                'type': 'mcq'
            }
        
        return None
    
    def _parse_saq(self, text: str) -> Optional[Dict]:
        """Parse SAQ from text"""
        if 'ANSWER:' not in text:
            return None
        
        parts = text.split('ANSWER:', 1)
        question = parts[0].split(':', 1)[1].strip() if ':' in parts[0] else parts[0].strip()
        answer = parts[1].strip()
        
        return {
            'question': question,
            'answer': answer,
            'type': 'saq'
        }


# Example usage
if __name__ == "__main__":
    # Test with Gemini
    llm = LLMManager(model_type="gemini")
    llm.load_model()
    
    sample_text = """
    Photosynthesis is the process by which green plants convert light energy 
    into chemical energy. It occurs in the chloroplasts and requires sunlight, 
    water, and carbon dioxide. The main products are glucose and oxygen.
    """
    
    print("\n=== Testing MCQ Generation ===")
    mcqs = llm.generate_questions(sample_text, "mcq", num_questions=2)
    for q in mcqs:
        print(f"\nQ: {q['question']}")
        for opt, text in q['options'].items():
            print(f"  {opt}) {text}")
        print(f"Answer: {q['correct_answer']}")
    
    print("\n=== Testing SAQ Generation ===")
    saqs = llm.generate_questions(sample_text, "saq", num_questions=2)
    for q in saqs:
        print(f"\nQ: {q['question']}")
        print(f"A: {q['answer']}")