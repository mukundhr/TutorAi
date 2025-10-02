"""
Export handler for saving questions in various formats
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import sys
sys.path.append('..')
import config

class ExportHandler:
    def __init__(self, output_dir: Path = config.OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_json(self, questions: List[Dict], filename: str = None) -> str:
        """Export questions to JSON format"""
        if filename is None:
            filename = f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_questions': len(questions),
            'questions': questions
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def export_to_csv(self, questions: List[Dict], filename: str = None) -> str:
        """Export questions to CSV format"""
        if filename is None:
            filename = f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Determine fieldnames based on question types
            fieldnames = ['type', 'question', 'answer', 'correct_answer', 
                         'option_A', 'option_B', 'option_C', 'option_D', 
                         'explanation', 'points']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for q in questions:
                row = {
                    'type': q.get('type', ''),
                    'question': q.get('question', ''),
                    'points': q.get('points', 0)
                }
                
                if q.get('type') == 'Short Answer':
                    row['answer'] = q.get('answer', '')
                elif q.get('type') == 'Multiple Choice':
                    options = q.get('options', {})
                    row['option_A'] = options.get('A', '')
                    row['option_B'] = options.get('B', '')
                    row['option_C'] = options.get('C', '')
                    row['option_D'] = options.get('D', '')
                    row['correct_answer'] = q.get('correct_answer', '')
                    row['explanation'] = q.get('explanation', '')
                
                writer.writerow(row)
        
        return str(filepath)
    
    def export_to_txt(self, questions: List[Dict], filename: str = None) -> str:
        """Export questions to readable text format"""
        if filename is None:
            filename = f"questions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GENERATED QUESTIONS\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Questions: {len(questions)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, q in enumerate(questions, 1):
                f.write(f"Question {i}: [{q.get('type', 'Unknown')}]\n")
                f.write("-" * 80 + "\n")
                f.write(f"{q.get('question', '')}\n\n")
                
                if q.get('type') == 'Short Answer':
                    f.write(f"Expected Answer:\n{q.get('answer', '')}\n")
                    f.write(f"Points: {q.get('points', 0)}\n")
                
                elif q.get('type') == 'Multiple Choice':
                    options = q.get('options', {})
                    for letter in ['A', 'B', 'C', 'D']:
                        f.write(f"{letter}) {options.get(letter, '')}\n")
                    f.write(f"\nCorrect Answer: {q.get('correct_answer', '')}\n")
                    
                    explanation = q.get('explanation', '')
                    if explanation:
                        f.write(f"Explanation: {explanation}\n")
                    f.write(f"Points: {q.get('points', 0)}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        return str(filepath)
    
    def export_to_moodle_xml(self, questions: List[Dict], filename: str = None) -> str:
        """Export questions to Moodle XML format"""
        if filename is None:
            filename = f"questions_moodle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<quiz>\n')
            
            for q in questions:
                if q.get('type') == 'Short Answer':
                    f.write('  <question type="essay">\n')
                    f.write(f'    <name><text>Short Answer Question</text></name>\n')
                    f.write(f'    <questiontext format="html">\n')
                    f.write(f'      <text><![CDATA[{q.get("question", "")}]]></text>\n')
                    f.write(f'    </questiontext>\n')
                    f.write(f'    <defaultgrade>{q.get("points", 5)}</defaultgrade>\n')
                    f.write('  </question>\n')
                
                elif q.get('type') == 'Multiple Choice':
                    f.write('  <question type="multichoice">\n')
                    f.write(f'    <name><text>Multiple Choice Question</text></name>\n')
                    f.write(f'    <questiontext format="html">\n')
                    f.write(f'      <text><![CDATA[{q.get("question", "")}]]></text>\n')
                    f.write(f'    </questiontext>\n')
                    f.write(f'    <defaultgrade>{q.get("points", 3)}</defaultgrade>\n')
                    f.write('    <single>true</single>\n')
                    
                    options = q.get('options', {})
                    correct = q.get('correct_answer', '')
                    
                    for letter in ['A', 'B', 'C', 'D']:
                        fraction = '100' if letter == correct else '0'
                        f.write(f'    <answer fraction="{fraction}">\n')
                        f.write(f'      <text><![CDATA[{options.get(letter, "")}]]></text>\n')
                        f.write(f'    </answer>\n')
                    
                    f.write('  </question>\n')
            
            f.write('</quiz>\n')
        
        return str(filepath)