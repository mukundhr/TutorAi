"""
Evaluation metrics for generated questions
"""
import sys
sys.path.append('..')
import config
from typing import List, Dict

class QuestionEvaluator:
    def __init__(self):
        self.enable_bertscore = config.ENABLE_BERTSCORE
        self.enable_rouge = config.ENABLE_ROUGE
        
        # Lazy load heavy libraries
        self.bert_scorer = None
        self.rouge_scorer = None
    
    def _init_bert_scorer(self):
        """Initialize BERTScore (lazy loading)"""
        if self.enable_bertscore and self.bert_scorer is None:
            try:
                from bert_score import BERTScorer
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            except Exception as e:
                print(f"Warning: Could not load BERTScore: {e}")
                self.enable_bertscore = False
    
    def _init_rouge_scorer(self):
        """Initialize ROUGE scorer (lazy loading)"""
        if self.enable_rouge and self.rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
            except Exception as e:
                print(f"Warning: Could not load ROUGE: {e}")
                self.enable_rouge = False
    
    def evaluate_question_quality(self, questions: List[Dict]) -> Dict:
        """Evaluate overall quality of generated questions"""
        metrics = {
            'total_questions': len(questions),
            'valid_questions': 0,
            'avg_question_length': 0,
            'question_types': {},
            'quality_score': 0.0
        }
        
        if not questions:
            return metrics
        
        total_length = 0
        valid_count = 0
        
        for q in questions:
            q_type = q.get('type', 'Unknown')
            metrics['question_types'][q_type] = metrics['question_types'].get(q_type, 0) + 1
            
            q_text = q.get('question', '')
            total_length += len(q_text.split())
            
            # Basic quality checks
            if self._is_valid_question(q):
                valid_count += 1
        
        metrics['valid_questions'] = valid_count
        metrics['avg_question_length'] = total_length / len(questions)
        metrics['quality_score'] = (valid_count / len(questions)) * 100
        
        return metrics
    
    def _is_valid_question(self, question: Dict) -> bool:
        """Check if question meets basic validity criteria"""
        q_text = question.get('question', '')
        
        # Must have question text
        if not q_text or len(q_text) < 10:
            return False
        
        # For MCQ, check options
        if question.get('type') == 'Multiple Choice':
            options = question.get('options', {})
            correct = question.get('correct_answer', '')
            
            if len(options) != 4 or correct not in ['A', 'B', 'C', 'D']:
                return False
        
        # For SAQ, check answer
        if question.get('type') == 'Short Answer':
            answer = question.get('answer', '')
            if not answer or len(answer.split()) < 5:
                return False
        
        return True
    
    def calculate_diversity(self, questions: List[Dict]) -> float:
        """Calculate diversity score based on unique question patterns"""
        if not questions:
            return 0.0
        
        question_texts = [q.get('question', '').lower() for q in questions]
        
        # Calculate unique trigrams
        all_trigrams = set()
        for text in question_texts:
            words = text.split()
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            all_trigrams.update(trigrams)
        
        # Diversity = unique trigrams / total possible
        total_words = sum(len(text.split()) for text in question_texts)
        diversity_score = len(all_trigrams) / max(total_words, 1) * 100
        
        return min(diversity_score, 100.0)
    
    def evaluate_with_reference(self, 
                                generated: str, 
                                reference: str) -> Dict:
        """Evaluate generated text against reference (if available)"""
        scores = {}
        
        # ROUGE scores
        if self.enable_rouge:
            self._init_rouge_scorer()
            if self.rouge_scorer:
                rouge_scores = self.rouge_scorer.score(reference, generated)
                scores['rouge1'] = rouge_scores['rouge1'].fmeasure
                scores['rouge2'] = rouge_scores['rouge2'].fmeasure
                scores['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # BERTScore
        if self.enable_bertscore:
            self._init_bert_scorer()
            if self.bert_scorer:
                P, R, F1 = self.bert_scorer.score([generated], [reference])
                scores['bertscore_f1'] = F1.item()
        
        return scores
    
    def generate_report(self, questions: List[Dict]) -> str:
        """Generate a text report of evaluation metrics"""
        metrics = self.evaluate_question_quality(questions)
        diversity = self.calculate_diversity(questions)
        
        report = f"""
=== Question Generation Report ===

Total Questions Generated: {metrics['total_questions']}
Valid Questions: {metrics['valid_questions']}
Quality Score: {metrics['quality_score']:.2f}%
Diversity Score: {diversity:.2f}%
Average Question Length: {metrics['avg_question_length']:.1f} words

Question Type Distribution:
"""
        for q_type, count in metrics['question_types'].items():
            report += f"  - {q_type}: {count}\n"
        
        return report