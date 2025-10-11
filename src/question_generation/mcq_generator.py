"""
Multiple Choice Question (MCQ) generator - FIXED VERSION
"""
import re
from typing import List, Dict
from .prompts import MCQ_SYSTEM_PROMPT

class MCQGenerator:
    def __init__(self, ai_handler):
        self.ai = ai_handler
    
    def generate_questions(self, 
                          content: str, 
                          num_questions: int = 5) -> List[Dict]:
        """Generate EXACTLY num_questions MCQs"""
        
        prompt = f"""Based on the following lecture content, you MUST generate EXACTLY {num_questions} multiple choice questions.

CRITICAL: Generate ALL {num_questions} questions. Each must have 4 options (A, B, C, D).

Lecture Content:
{content}

Instructions:
- Generate EXACTLY {num_questions} complete MCQs
- Each question must have 4 options: A, B, C, D
- Mark the correct answer
- Provide a brief explanation

Required Format:

Q1: [First question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]
Explanation: [Why this is correct]

Q2: [Second question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]
Explanation: [Why this is correct]

Continue until you have generated ALL {num_questions} questions.

Generate all {num_questions} questions now:"""
        
        try:
            response = self.ai.generate(
                prompt=prompt,
                system_prompt="You are an expert educator. Generate the EXACT number of MCQs requested. Complete ALL questions.",
                max_tokens=2500,  # MCQs need more tokens
                temperature=0.7
            )
            
            print(f"\n📝 MCQ RAW RESPONSE LENGTH: {len(response)} characters")
            
            questions = self._parse_response(response)
            print(f"✅ Parsed {len(questions)} MCQs from response")
            
            # Retry if needed
            if len(questions) < num_questions:
                print(f"⚠️ Only got {len(questions)}/{num_questions} MCQs, retrying...")
                remaining = num_questions - len(questions)
                
                retry_prompt = f"""Generate {remaining} MORE multiple choice questions from this content.

Content: {content}

Start from Q{len(questions)+1}.

Generate {remaining} more MCQs:"""
                
                retry_response = self.ai.generate(
                    prompt=retry_prompt,
                    max_tokens=2000,
                    temperature=0.7
                )
                
                retry_questions = self._parse_response(retry_response)
                questions.extend(retry_questions)
                print(f"✅ After retry: Total {len(questions)} MCQs")
            
            return questions[:num_questions]
            
        except Exception as e:
            print(f"❌ Error generating MCQs: {str(e)}")
            return []
    
    def _parse_response(self, response: str) -> List[Dict]:
        """Parse MCQ response"""
        questions = []
        
        # Split by Q markers
        parts = re.split(r'(?=Q\d+[:\)])', response)
        
        for part in parts:
            part = part.strip()
            if not part or len(part) < 50:
                continue
            
            try:
                # Extract question
                q_match = re.search(r'Q\d+[:\)]\s*(.+?)(?=\n*[A-D]\))', part, re.DOTALL)
                if not q_match:
                    continue
                
                question_text = q_match.group(1).strip()
                question_text = re.sub(r'\s+', ' ', question_text)
                
                # Extract options A, B, C, D
                options = {}
                for letter in ['A', 'B', 'C', 'D']:
                    pattern = f'{letter}\\)\s*(.+?)(?=\n*[A-D]\\)|Correct:|Explanation:|$)'
                    opt_match = re.search(pattern, part, re.DOTALL)
                    if opt_match:
                        opt_text = opt_match.group(1).strip()
                        opt_text = re.sub(r'\s+', ' ', opt_text)
                        options[letter] = opt_text
                
                # Must have all 4 options
                if len(options) != 4:
                    print(f"⚠️ Skipping MCQ - only found {len(options)} options")
                    continue
                
                # Extract correct answer
                correct_match = re.search(r'Correct[:\s]+([A-D])', part, re.IGNORECASE)
                correct_answer = correct_match.group(1).upper() if correct_match else 'A'
                
                # Extract explanation (optional)
                exp_match = re.search(r'Explanation[:\s]+(.+?)(?=Q\d+[:\)]|$)', part, re.DOTALL | re.IGNORECASE)
                explanation = exp_match.group(1).strip() if exp_match else ""
                explanation = re.sub(r'\s+', ' ', explanation)
                
                questions.append({
                    'type': 'Multiple Choice',
                    'question': question_text,
                    'options': options,
                    'correct_answer': correct_answer,
                    'explanation': explanation,
                    'points': 3
                })
                
            except Exception as e:
                print(f"⚠️ Error parsing MCQ part: {str(e)}")
                continue
        
        return questions
