"""
Prompt templates for LLaMA 2 question generation
"""

SAQ_SYSTEM_PROMPT = """You are an expert educational content creator specializing in generating high-quality short answer questions from academic lecture notes. Your questions should test deep understanding, not just memorization."""

SAQ_USER_PROMPT = """Based on the following lecture content, generate {num_questions} challenging short answer questions that test conceptual understanding and application.

Lecture Content:
{content}

Requirements:
- Questions should require 2-4 sentence answers
- Test understanding, not just recall
- Cover different aspects of the content
- Be clear and specific

Format each question exactly as:
Q: [Your question here]
A: [Expected answer in 2-4 sentences]

---

Generate the questions now:"""

MCQ_SYSTEM_PROMPT = """You are an expert educational assessment designer. You create challenging multiple-choice questions with plausible distractors that test genuine understanding."""

MCQ_USER_PROMPT = """Based on the following lecture content, generate {num_questions} multiple choice question(s) with 4 options each.

Lecture Content:
{content}

Requirements:
- Create challenging questions that test understanding
- Include 1 correct answer and 3 plausible distractors
- Distractors should represent common misconceptions or partial understanding
- All options should be similar in length and complexity

Format each question exactly as:
Q: [Your question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct: [A/B/C/D]
Explanation: [Brief explanation of why the answer is correct]

---

Generate the question(s) now:"""

DISTRACTOR_GENERATION_PROMPT = """Given this correct answer from a lecture, generate 3 plausible but incorrect alternatives (distractors).

Correct Answer: {correct_answer}
Context: {context}

Generate distractors that:
- Are plausible and realistic
- Represent common misconceptions
- Are similar in length to the correct answer
- Would trick someone with partial understanding

Distractors:
1. 
2. 
3. """

def format_saq_prompt(content: str, num_questions: int = 3) -> str:
    """Format short answer question prompt"""
    return SAQ_USER_PROMPT.format(content=content, num_questions=num_questions)

def format_mcq_prompt(content: str, num_questions: int = 3) -> str:
    """Format multiple choice question prompt"""
    return MCQ_USER_PROMPT.format(content=content, num_questions=num_questions)

def format_distractor_prompt(correct_answer: str, context: str) -> str:
    """Format distractor generation prompt"""
    return DISTRACTOR_GENERATION_PROMPT.format(
        correct_answer=correct_answer,
        context=context
    )