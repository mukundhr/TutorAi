"""
Unified LLM Manager for both MCQ and SAQ generation
Supports: Google Gemini 2.0 Flash and LLaMA 2 7B
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Gemini
import google.generativeai as genai

# LLaMA - make it optional
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except Exception as e:
    print(f"⚠️ LLaMA not available: {e}")
    LLAMA_AVAILABLE = False
    Llama = None

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
            if not LLAMA_AVAILABLE:
                raise RuntimeError(
                    "LLaMA is not available. Missing CUDA dependencies.\n\n"
                    "Please use Gemini instead, or install CPU-only version:\n"
                    "  pip uninstall llama-cpp-python -y\n"
                    "  pip install llama-cpp-python\n\n"
                    "For CUDA support, you need CUDA toolkit installed first."
                )
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
        
        print(f"📂 Loading LLaMA from: {model_path}")
        print("🎮 Attempting GPU acceleration...")
        
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=2048,  # Reduced for faster processing
                n_threads=8,
                n_gpu_layers=35,
                n_batch=256,  # Reduced from 512
                verbose=False,
                use_mlock=True  # Keep model in RAM
            )
            print("✅ LLaMA 2 7B loaded with GPU")
        except Exception as e:
            print(f"⚠️ GPU loading failed: {e}")
            print("🔄 Retrying with CPU only...")
            self.model = Llama(
                model_path=model_path,
                n_ctx=1536,  # Smaller context for CPU
                n_threads=6,  # Optimized thread count
                n_gpu_layers=0,
                n_batch=128,
                verbose=False,
                use_mlock=True
            )
            print("✅ LLaMA 2 7B loaded (CPU mode - will be slower)")
            print("⚠️ NOTE: CPU generation can take 1-3 minutes per batch")
        
        # Test generation
        print("🧪 Testing LLaMA generation...")
        import time
        start = time.time()
        test_result = self.model(
            "[INST] Say only: Test successful [/INST]",
            max_tokens=20,
            temperature=0.1
        )
        elapsed = time.time() - start
        test_output = test_result['choices'][0]['text'].strip()
        print(f"✅ LLaMA test output: '{test_output}' ({elapsed:.1f}s)")
    
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
        print(f"\n{'='*60}")
        print(f"🎯 Generating {num_questions} {question_type.upper()}s using {self.model_type.upper()}")
        print(f"{'='*60}")
        
        prompt = self._build_prompt(text, question_type, num_questions, difficulty)
        
        if self.model_type == "gemini":
            response = self._generate_gemini(prompt, temperature)
        else:
            response = self._generate_llama(prompt, temperature)
        
        # Check if we got a response
        if not response or len(response.strip()) < 50:
            print(f"\n❌ GENERATION FAILED!")
            print(f"   Model: {self.model_type.upper()}")
            print(f"   Response length: {len(response) if response else 0} chars")
            print(f"   Response content: '{response[:200] if response else 'EMPTY'}'")
            print(f"\n💡 Troubleshooting tips:")
            if self.model_type == "llama":
                print(f"   - LLaMA may need more time/resources")
                print(f"   - Try switching to Gemini (more reliable)")
                print(f"   - Check terminal for detailed error messages")
            else:
                print(f"   - Check your Gemini API key in .env file")
                print(f"   - Verify internet connection")
            return []
        
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
        
        print(f"✅ Successfully generated {len(questions)} {question_type.upper()}s\n")
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
            print(f"🔵 Calling Gemini with temperature={temperature}...")
            print(f"📝 Prompt length: {len(prompt)} characters")
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                )
            )
            
            result = response.text
            print(f"✅ Gemini response length: {len(result)} characters")
            
            if not result:
                print("⚠️ WARNING: Gemini returned empty response!")
            
            return result
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _generate_llama(self, prompt: str, temperature: float) -> str:
        """Generate using LLaMA"""
        import time
        try:
            print(f"🟡 Calling LLaMA with temperature={temperature}...")
            print(f"📝 Prompt length: {len(prompt)} characters")
            print(f"⏱️ This may take 30-90 seconds on CPU...")
            
            # LLaMA 2 Chat format
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            start_time = time.time()
            
            response = self.model(
                formatted_prompt,
                max_tokens=1024,  # Reduced from 2048 for faster generation
                temperature=temperature,
                top_p=0.9,
                echo=False,
                stop=["[INST]", "</s>", "Q6:", "Q7:"],  # Stop after requested questions
                repeat_penalty=1.1,
                top_k=40  # Add top_k for better quality/speed balance
            )
            
            elapsed = time.time() - start_time
            result = response['choices'][0]['text'].strip()
            print(f"✅ LLaMA response length: {len(result)} characters (took {elapsed:.1f}s)")
            print(f"📄 LLaMA response preview:\n{result[:300]}...")
            
            if not result or len(result.strip()) < 20:
                print("⚠️ WARNING: LLaMA returned empty or very short response!")
                print(f"Full response: '{result}'")
            
            return result
        except Exception as e:
            print(f"❌ LLaMA error: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return ""
    
    def _parse_response(self, response: str, question_type: str) -> List[Dict]:
        """Parse LLM response into structured questions"""
        questions = []
        
        if not response:
            print("❌ No response to parse!")
            return questions
        
        print(f"📄 Parsing {question_type} response ({len(response)} chars)...")
        print(f"First 500 chars: {response[:500]}...")
        
        # Try to split by question patterns
        import re
        
        # Look for Q1:, Q2:, etc. or just Q: patterns
        question_pattern = r'(?:^|\n)Q\d*[:.]\s*(.+?)(?=\nQ\d*[:.]|\n*$)'
        matches = re.findall(question_pattern, response, re.DOTALL | re.MULTILINE)
        
        if not matches:
            # Fallback: split by double newlines
            parts = response.split('\n\n')
            print(f"🔍 Fallback: Found {len(parts)} parts after splitting by \\n\\n")
        else:
            parts = matches
            print(f"🔍 Regex: Found {len(matches)} question blocks")
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            
            # Reconstruct question block if needed
            if not part.strip().startswith('Q'):
                part = f"Q{i+1}: {part}"
            
            try:
                if question_type == "mcq":
                    q = self._parse_mcq(part)
                else:
                    q = self._parse_saq(part)
                
                if q:
                    questions.append(q)
                    print(f"✅ Successfully parsed question {len(questions)}")
                else:
                    print(f"⚠️ Failed to parse part {i}: {part[:100]}...")
            except Exception as e:
                print(f"⚠️ Parse error on part {i}: {e}")
                continue
        
        print(f"📊 Total questions parsed: {len(questions)}")
        return questions
    
    def _parse_mcq(self, text: str) -> Optional[Dict]:
        """Parse MCQ from text"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        if len(lines) < 6:  # Need Q, A, B, C, D, ANSWER minimum
            print(f"⚠️ MCQ too short: only {len(lines)} lines")
            return None
        
        # Extract question - handle various formats
        question = ""
        if ':' in lines[0]:
            question = lines[0].split(':', 1)[1].strip()
        else:
            question = lines[0].replace('Q1', '').replace('Q2', '').replace('Q3', '').replace('Q4', '').replace('Q5', '').strip()
        
        options = {}
        answer = ""
        explanation = ""
        
        for line in lines[1:]:
            # More flexible option matching
            if line.upper().startswith(('A)', 'A.', 'A:', 'A -')):
                options['A'] = line[2:].strip() if len(line) > 2 else line[3:].strip()
            elif line.upper().startswith(('B)', 'B.', 'B:', 'B -')):
                options['B'] = line[2:].strip() if len(line) > 2 else line[3:].strip()
            elif line.upper().startswith(('C)', 'C.', 'C:', 'C -')):
                options['C'] = line[2:].strip() if len(line) > 2 else line[3:].strip()
            elif line.upper().startswith(('D)', 'D.', 'D:', 'D -')):
                options['D'] = line[2:].strip() if len(line) > 2 else line[3:].strip()
            elif 'ANSWER:' in line.upper() or 'CORRECT:' in line.upper():
                # Extract answer letter
                answer_part = line.split(':', 1)[1].strip() if ':' in line else line
                # Get first letter that's A, B, C, or D
                for char in answer_part.upper():
                    if char in ['A', 'B', 'C', 'D']:
                        answer = char
                        break
            elif 'EXPLANATION:' in line.upper():
                explanation = line.split(':', 1)[1].strip() if ':' in line else ""
        
        if len(options) == 4 and answer:
            return {
                'question': question,
                'options': options,
                'correct_answer': answer,
                'explanation': explanation,
                'type': 'mcq'
            }
        else:
            print(f"⚠️ MCQ incomplete: {len(options)} options, answer={answer}")
        
        return None
    
    def _parse_saq(self, text: str) -> Optional[Dict]:
        """Parse SAQ from text"""
        if 'ANSWER:' not in text.upper():
            print(f"⚠️ SAQ missing ANSWER: marker")
            return None
        
        # Case-insensitive split
        import re
        parts = re.split(r'ANSWER:\s*', text, flags=re.IGNORECASE, maxsplit=1)
        
        if len(parts) < 2:
            return None
        
        # Extract question - handle various formats
        question = parts[0]
        if ':' in question:
            question = question.split(':', 1)[1].strip()
        question = question.replace('Q1', '').replace('Q2', '').replace('Q3', '').replace('Q4', '').replace('Q5', '').strip()
        
        answer = parts[1].strip()
        
        if question and answer and len(answer) > 10:
            return {
                'question': question,
                'answer': answer,
                'type': 'saq'
            }
        else:
            print(f"⚠️ SAQ incomplete: Q len={len(question)}, A len={len(answer)}")
        
        return None


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