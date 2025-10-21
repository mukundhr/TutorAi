"""
Short Answer Question (SAQ) generator
"""
import re
from typing import List, Dict
from .prompts import SAQ_SYSTEM_PROMPT

class SAQGenerator:
    def __init__(self, ai_handler):
        self.ai = ai_handler
    
    def generate_questions(self, 
                          content: str, 
                          num_questions: int = 5) -> List[Dict]:
        """
        Generate EXACTLY num_questions short answer questions
        
        Args:
            content: Text content to generate questions from
            num_questions: Exact number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        
        print(f"\n{'='*60}")
        print(f"🔄 GENERATING {num_questions} SHORT ANSWER QUESTIONS")
        print(f"{'='*60}")
        
        # Build enhanced prompt
        prompt = self._build_prompt(content, num_questions)
        
        # Enhanced system prompt
        system_prompt = """You are an expert educational content creator specializing in short answer questions.

CRITICAL RULES:
1. You MUST generate EXACTLY the number of questions requested
2. Each question MUST have both Q: and A: parts
3. Number them clearly (Q1/A1, Q2/A2, etc.)
4. Do NOT stop until you have completed ALL questions
5. Each answer should be 2-4 sentences long"""
        
        try:
            # Generate with high token limit
            print(f"📤 Sending prompt to AI...")
            response = self.ai.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=3000,  # Very high limit to generate all questions at once
                temperature=0.7
            )
            
            print(f"📥 Response received: {len(response)} characters")
            
            # Parse response
            questions = self._parse_response(response)
            
            print(f"✅ Successfully parsed {len(questions)} questions")
            
            # If we didn't get enough, try multiple retries
            retry_attempts = 0
            max_retries = 3  # Try up to 3 times
            
            while len(questions) < num_questions and retry_attempts < max_retries:
                retry_attempts += 1
                print(f"⚠️  Only got {len(questions)}/{num_questions} questions")
                print(f"🔄 Retry attempt {retry_attempts}/{max_retries} for {num_questions - len(questions)} more...")
                
                # Retry for remaining questions
                remaining = num_questions - len(questions)
                retry_questions = self._retry_generation(content, remaining, len(questions))
                
                if retry_questions:
                    questions.extend(retry_questions)
                    print(f"✅ After retry {retry_attempts}: Total {len(questions)} questions")
                else:
                    print(f"⚠️  Retry {retry_attempts} returned no questions")
                    break  # No point retrying if we got nothing
            
            # Return exactly what was requested (trim if we got too many)
            final_questions = questions[:num_questions]
            
            print(f"{'='*60}")
            print(f"✅ FINAL: Returning {len(final_questions)} SAQs")
            print(f"{'='*60}\n")
            
            return final_questions
            
        except Exception as e:
            print(f"❌ Error generating SAQs: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _build_prompt(self, content: str, num_questions: int) -> str:
        """Build an enhanced prompt that emphasizes exact count"""
        
        prompt = f"""Based on the following lecture content, you MUST generate EXACTLY {num_questions} short answer questions.

⚠️ CRITICAL REQUIREMENT: Generate ALL {num_questions} questions. Do NOT stop early!

Lecture Content:
{content}

Instructions:
1. Generate EXACTLY {num_questions} complete questions
2. Each question must have BOTH Q: and A: parts
3. Number them sequentially (Q1/A1, Q2/A2, Q3/A3, etc.)
4. Questions should test understanding, not just recall
5. Answers should be 2-4 sentences long
6. Make questions specific and clear

Required Format (FOLLOW THIS EXACTLY):

Q1: [Write your first question here - make it specific and test understanding]
A1: [Write a complete 2-4 sentence answer here that thoroughly addresses the question]

Q2: [Write your second question here - make it different from Q1]
A2: [Write a complete 2-4 sentence answer here]

Q3: [Write your third question here]
A3: [Write a complete 2-4 sentence answer here]

Q4: [Write your fourth question here]
A4: [Write a complete 2-4 sentence answer here]

Q5: [Write your fifth question here]
A5: [Write a complete 2-4 sentence answer here]

(Continue this pattern until you have written ALL {num_questions} questions)

NOW GENERATE ALL {num_questions} QUESTIONS:"""
        
        return prompt
    
    def _retry_generation(self, content: str, num_remaining: int, start_number: int) -> List[Dict]:
        """Retry generation for remaining questions"""
        
        retry_prompt = f"""You need to generate {num_remaining} MORE short answer questions from this content.

Content: {content}

IMPORTANT: Start numbering from Q{start_number + 1}.

Generate {num_remaining} additional questions now:

Q{start_number + 1}: [Question]
A{start_number + 1}: [Answer]

(Continue for all {num_remaining} questions)

Generate now:"""
        
        try:
            response = self.ai.generate(
                prompt=retry_prompt,
                system_prompt="Generate the exact number of questions requested. Complete ALL questions.",
                max_tokens=2000,  # Increased for retry
                temperature=0.7
            )
            
            return self._parse_response(response, offset=start_number)
            
        except Exception as e:
            print(f"❌ Retry failed: {e}")
            return []
    
    def _parse_response(self, response: str, offset: int = 0) -> List[Dict]:
        """
        Parse AI response into structured questions
        
        Args:
            response: Raw AI response text
            offset: Number offset for question numbering
            
        Returns:
            List of parsed questions
        """
        questions = []
        
        print(f"🔍 Parsing response (offset={offset})...")
        
        # Strategy 1: Match numbered Q/A pairs (most reliable)
        # Pattern: Q1: ... A1: ... Q2: ... A2: ...
        q_pattern = r'Q(\d+)[:\)]\s*(.+?)(?=\s*A\1[:\)]|\s*Q\d+[:\)]|$)'
        a_pattern = r'A(\d+)[:\)]\s*(.+?)(?=\s*Q\d+[:\)]|$)'
        
        # Find all Q matches
        q_matches = list(re.finditer(q_pattern, response, re.DOTALL | re.IGNORECASE))
        print(f"  Found {len(q_matches)} Q: markers")
        
        # Find all A matches  
        a_matches = list(re.finditer(a_pattern, response, re.DOTALL | re.IGNORECASE))
        print(f"  Found {len(a_matches)} A: markers")
        
        # Build dictionaries by number
        q_dict = {}
        for match in q_matches:
            num = int(match.group(1))
            text = match.group(2).strip()
            # Clean up text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            if len(text) > 10:  # Valid question
                q_dict[num] = text
        
        a_dict = {}
        for match in a_matches:
            num = int(match.group(1))
            text = match.group(2).strip()
            # Clean up text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            if len(text) > 15:  # Valid answer
                a_dict[num] = text
        
        # Combine Q&A pairs
        for num in sorted(q_dict.keys()):
            if num in a_dict:
                questions.append({
                    'type': 'Short Answer',
                    'question': q_dict[num],
                    'answer': a_dict[num],
                    'points': 5
                })
                print(f"  ✓ Paired Q{num} with A{num}")
        
        # Strategy 2: If Strategy 1 found nothing, try simpler split
        if not questions:
            print(f"  ⚠️  Strategy 1 failed, trying fallback parsing...")
            questions = self._fallback_parse(response)
        
        print(f"  📊 Total parsed: {len(questions)} questions")
        
        return questions
    
    def _fallback_parse(self, response: str) -> List[Dict]:
        """Fallback parsing strategy for less structured responses"""
        questions = []
        
        # Split by Q markers (less strict)
        parts = re.split(r'(?=Q\d*[:\)])', response, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if len(part) < 30:  # Too short to be a valid Q&A
                continue
            
            # Try to extract Q and A
            q_match = re.search(r'Q\d*[:\)]\s*(.+?)(?=A\d*[:\)]|$)', part, re.DOTALL | re.IGNORECASE)
            a_match = re.search(r'A\d*[:\)]\s*(.+?)(?=Q\d*[:\)]|$)', part, re.DOTALL | re.IGNORECASE)
            
            if q_match:
                q_text = q_match.group(1).strip()
                q_text = re.sub(r'\s+', ' ', q_text)
                
                a_text = ""
                if a_match:
                    a_text = a_match.group(1).strip()
                    a_text = re.sub(r'\s+', ' ', a_text)
                
                # Validate
                if len(q_text) > 10 and len(a_text) > 15:
                    questions.append({
                        'type': 'Short Answer',
                        'question': q_text,
                        'answer': a_text,
                        'points': 5
                    })
        
        return questions
    
    def generate_from_chunks(self, 
                            chunks: List[str], 
                            total_questions: int = 5) -> List[Dict]:
        """
        Generate questions from multiple chunks
        DEPRECATED: Better to combine chunks first
        """
        print(f"⚠️  WARNING: generate_from_chunks is deprecated")
        print(f"   Recommendation: Combine chunks and use generate_questions()")
        
        # Combine chunks
        combined = ' '.join(chunks[:5])
        if len(combined) > 3000:
            combined = combined[:3000]
        
        # Generate from combined content
        return self.generate_questions(combined, total_questions)
    
    def validate_question(self, question: Dict) -> bool:
        """Validate if a question meets quality criteria"""
        q_text = question.get('question', '')
        a_text = question.get('answer', '')
        
        # Must have both parts
        if not q_text or not a_text:
            return False
        
        # Minimum lengths
        if len(q_text) < 10 or len(a_text) < 15:
            return False
        
        # Question should end with ?
        if not q_text.strip().endswith('?'):
            return False
        
        # Answer should have at least 5 words
        if len(a_text.split()) < 5:
            return False
        
        return True