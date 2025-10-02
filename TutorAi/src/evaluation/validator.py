"""
Question validation module
"""
from typing import List, Dict, Tuple

class QuestionValidator:
    def __init__(self):
        self.validation_rules = {
            'min_question_length': 10,
            'max_question_length': 500,
            'min_answer_words': 5,
            'required_mcq_options': 4
        }
    
    def validate_saq(self, question: Dict) -> Tuple[bool, str]:
        """Validate a short answer question"""
        q_text = question.get('question', '').strip()
        answer = question.get('answer', '').strip()
        
        # Check question text
        if not q_text:
            return False, "Question text is empty"
        
        if len(q_text) < self.validation_rules['min_question_length']:
            return False, "Question text too short"
        
        if not q_text.endswith('?'):
            return False, "Question must end with a question mark"
        
        # Check answer
        if not answer:
            return False, "Answer is empty"
        
        answer_words = len(answer.split())
        if answer_words < self.validation_rules['min_answer_words']:
            return False, f"Answer too short (minimum {self.validation_rules['min_answer_words']} words)"
        
        return True, "Valid"
    
    def validate_mcq(self, question: Dict) -> Tuple[bool, str]:
        """Validate a multiple choice question"""
        q_text = question.get('question', '').strip()
        options = question.get('options', {})
        correct = question.get('correct_answer', '')
        
        # Check question text
        if not q_text:
            return False, "Question text is empty"
        
        if len(q_text) < self.validation_rules['min_question_length']:
            return False, "Question text too short"
        
        # Check options
        if len(options) != self.validation_rules['required_mcq_options']:
            return False, f"Must have exactly {self.validation_rules['required_mcq_options']} options"
        
        # Check if all options are present
        required_keys = {'A', 'B', 'C', 'D'}
        if set(options.keys()) != required_keys:
            return False, "Options must be labeled A, B, C, D"
        
        # Check if options have content
        for key, value in options.items():
            if not value or len(value.strip()) < 3:
                return False, f"Option {key} is too short or empty"
        
        # Check correct answer
        if correct not in required_keys:
            return False, "Correct answer must be A, B, C, or D"
        
        # Check for duplicate options
        option_values = [v.strip().lower() for v in options.values()]
        if len(option_values) != len(set(option_values)):
            return False, "Duplicate options detected"
        
        return True, "Valid"
    
    def validate_batch(self, questions: List[Dict]) -> Dict:
        """Validate a batch of questions"""
        results = {
            'total': len(questions),
            'valid': 0,
            'invalid': 0,
            'errors': []
        }
        
        for i, q in enumerate(questions):
            q_type = q.get('type', 'Unknown')
            
            if q_type == 'Short Answer':
                is_valid, message = self.validate_saq(q)
            elif q_type == 'Multiple Choice':
                is_valid, message = self.validate_mcq(q)
            else:
                is_valid = False
                message = f"Unknown question type: {q_type}"
            
            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
                results['errors'].append({
                    'index': i,
                    'type': q_type,
                    'error': message
                })
        
        return results
    
    def filter_valid_questions(self, questions: List[Dict]) -> List[Dict]:
        """Filter and return only valid questions"""
        valid_questions = []
        
        for q in questions:
            q_type = q.get('type', 'Unknown')
            
            if q_type == 'Short Answer':
                is_valid, _ = self.validate_saq(q)
            elif q_type == 'Multiple Choice':
                is_valid, _ = self.validate_mcq(q)
            else:
                is_valid = False
            
            if is_valid:
                valid_questions.append(q)
        
        return valid_questions