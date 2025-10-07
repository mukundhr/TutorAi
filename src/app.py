"""
AI Tutor - Main Streamlit Application
Automatic Question Generation from Lecture Notes
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from preprocessing.pdf_extractor import PDFExtractor
from preprocessing.text_cleaner import TextCleaner
from preprocessing.chunker import TextChunker
from question_generation.gemini_handler import GeminiHandler
from question_generation.llama_handler import LlamaHandler
from question_generation.saq_generator import SAQGenerator
from question_generation.mcq_generator import MCQGenerator
from rag.vector_store import VectorStore
from rag.retriever import DocumentRetriever
from rag.rag_generator import RAGQuestionGenerator
from evaluation.metrics import QuestionEvaluator
from evaluation.validator import QuestionValidator
from utilities.cache_manager import CacheManager
from utilities.export_handler import ExportHandler

import json
import time

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = None
if 'llama_model' not in st.session_state:
    st.session_state.llama_model = None
if 'transformers_model' not in st.session_state:
    st.session_state.transformers_model = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = config.DEFAULT_LLM
if 'enable_rag' not in st.session_state:
    st.session_state.enable_rag = True  # RAG enabled by default (refined version)
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_generator' not in st.session_state:
    st.session_state.rag_generator = None

def load_model():
    """Load models with caching"""
    selected_llm = st.session_state.selected_llm
    
    # Load the selected LLM for MCQ generation
    if selected_llm == "gemini":
        if st.session_state.gemini_model is None:
            with st.spinner("Loading Gemini model..."):
                try:
                    st.session_state.gemini_model = GeminiHandler()
                    st.success("[OK] Gemini model loaded successfully!")
                except Exception as e:
                    st.error(f"Gemini model failed to load: {str(e)}")
                    st.info("Please check your GEMINI_API_KEY in .env file or switch to local LLaMA model.")
                    return None
    
    elif selected_llm == "llama":
        if st.session_state.llama_model is None:
            with st.spinner("Loading LLaMA 2 7B model..."):
                try:
                    st.session_state.llama_model = LlamaHandler()
                    st.success("[OK] LLaMA model loaded successfully!")
                except Exception as e:
                    st.warning(f"LLaMA model failed to load: {str(e)}")
                    st.info("Will use transformers model for question generation instead.")
    
    # Always ensure we have a model for SAQ generation (Transformers)
    if st.session_state.transformers_model is None:
        with st.spinner("Loading Transformers model for SAQ generation..."):
            try:
                from question_generation.transformers_handler import TransformersHandler
                st.session_state.transformers_model = TransformersHandler()
                
                # Show which model was loaded
                model_info = st.session_state.transformers_model.get_model_info()
                st.success(f"[OK] Transformers model loaded: {model_info['loaded_model']}")
                
                # Show fallback attempts if any failed
                failed_attempts = [a for a in model_info['attempts'] if not a['success']]
                if failed_attempts:
                    st.info(f"Note: {len(failed_attempts)} alternative models were tried first")
                    
            except Exception as e:
                st.warning(f"Transformers model failed to load: {str(e)}")
                st.info("Will use template questions if no other models are available.")
    
    # Return the appropriate model for MCQ generation
    if selected_llm == "gemini":
        return st.session_state.gemini_model
    else:
        return st.session_state.llama_model or st.session_state.transformers_model

def main():
    # Header
    st.title("AI Tutor - Question Generation System")
    st.markdown("### Generate Short Answer and Multiple Choice Questions from Lecture Notes")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # LLM Model Selection
        st.subheader("🤖 LLM Selection")
        llm_option = st.radio(
            "Choose LLM for MCQ Generation:",
            options=["gemini", "llama"],
            format_func=lambda x: "Google Gemini (API)" if x == "gemini" else "Local LLaMA 2 7B",
            index=0 if st.session_state.selected_llm == "gemini" else 1,
            help="Gemini requires API key, LLaMA runs locally"
        )
        
        # Update selected LLM if changed
        if llm_option != st.session_state.selected_llm:
            st.session_state.selected_llm = llm_option
            # Clear the loaded models to force reload
            st.session_state.gemini_model = None
            st.session_state.llama_model = None
            st.info(f"Switched to {llm_option.upper()}. Click 'Load Models' to initialize.")
        
        st.markdown("---")
        
        # Model status
        st.subheader("Model Status")
        
        # Show status based on selected LLM
        if st.session_state.selected_llm == "gemini":
            if st.session_state.gemini_model is not None:
                st.success("[OK] Gemini Model Loaded")
                model_info = st.session_state.gemini_model.get_model_info()
                st.caption(f"Model: {model_info['model_name']}")
            else:
                st.info("⚬ Gemini Model (Not Loaded)")
        else:  # llama
            if st.session_state.llama_model is not None:
                st.success("[OK] LLaMA Model Loaded")
            else:
                st.info("⚬ LLaMA Model (Not Loaded)")
        
        # Transformers status (for SAQ)
        if st.session_state.transformers_model is not None:
            model_info = st.session_state.transformers_model.get_model_info()
            st.success(f"[OK] Transformers: {model_info['loaded_model']}")
            if model_info['total_attempts'] > 1:
                st.caption(f"Loaded after {model_info['total_attempts']} attempts")
        else:
            st.info("⚬ Transformers Model (SAQ)")
        
        # Show load button if selected model is not loaded
        selected_model_loaded = (
            (st.session_state.selected_llm == "gemini" and st.session_state.gemini_model is not None) or
            (st.session_state.selected_llm == "llama" and st.session_state.llama_model is not None)
        )
        transformers_loaded = st.session_state.transformers_model is not None
        
        if not selected_model_loaded or not transformers_loaded:
            if st.button("Load Models"):
                load_model()
                st.rerun()
        else:
            st.caption("[OK] Ready for question generation")
        
        st.markdown("---")
        
        # RAG Settings
        st.subheader("🔍 RAG Settings")
        enable_rag = st.checkbox(
            "Enable RAG (Retrieval-Augmented Generation)",
            value=st.session_state.enable_rag,
            help="Use semantic search to retrieve relevant context from the entire document for better question generation"
        )
        if enable_rag != st.session_state.enable_rag:
            st.session_state.enable_rag = enable_rag
        
        if enable_rag:
            st.success("✓ RAG Enabled - Enhanced context retrieval active")
        else:
            st.info("RAG Disabled - Using standard chunking")
        
        st.markdown("---")
        
        # Question generation settings
        st.subheader("Question Settings")
        num_saq = st.slider("Number of Short Answer Questions", 0, 20, config.DEFAULT_NUM_SAQ)
        num_mcq = st.slider("Number of Multiple Choice Questions", 0, 20, config.DEFAULT_NUM_MCQ)
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.1, 1.0, config.MODEL_TEMPERATURE, 0.1)
            chunk_size = st.slider("Chunk Size", 200, 1500, config.MAX_CHUNK_SIZE, 100)
            enable_cache = st.checkbox("Enable Caching", value=True)
        
        st.markdown("---")
        
        # Cache management
        st.subheader("Cache Management")
        cache_manager = CacheManager()
        cache_stats = cache_manager.get_cache_stats()
        st.info(f"Cached items: {cache_stats['num_cached_items']}")
        if st.button("Clear Cache"):
            cache_manager.clear()
            st.success("Cache cleared!")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📄 Upload & Generate", "📊 Results", "📈 Analytics"])
    
    with tab1:
        st.header("Upload Lecture Notes")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload lecture notes in PDF format"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            upload_path = config.UPLOAD_DIR / uploaded_file.name
            with open(upload_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"[OK] File uploaded: {uploaded_file.name}")
            
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")
            
            # Extract and preview text
            with st.expander("Preview Extracted Text"):
                try:
                    extractor = PDFExtractor()
                    raw_text = extractor.extract_text(str(upload_path))
                    metadata = extractor.get_metadata()
                    
                    with col2:
                        st.metric("Pages", metadata.get('num_pages', 'N/A'))
                    with col3:
                        st.metric("Characters", len(raw_text))
                    
                    st.text_area("Raw Text Preview", raw_text[:1000] + "...", height=200)
                except Exception as e:
                    st.error(f"Error extracting text: {str(e)}")
            
            # Generate questions button
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
                    if num_saq == 0 and num_mcq == 0:
                        st.warning("Please select at least one question type to generate!")
                    else:
                        generate_questions(
                            upload_path,
                            num_saq,
                            num_mcq,
                            temperature,
                            chunk_size,
                            enable_cache
                        )
    
    with tab2:
        st.header("Generated Questions")
        
        # Quick debug info (less verbose)
        if st.checkbox("🔍 Show Debug Info"):
            st.write(f"**Questions available:** {len(st.session_state.questions) if st.session_state.questions else 0}")
            st.write(f"**Processing complete:** {st.session_state.processing_complete}")
        

        
        # Display questions if they exist
        if st.session_state.questions and len(st.session_state.questions) > 0:
            # Add refresh button
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🔄 Refresh Display"):
                    st.rerun()
            with col2:
                st.write(f"**{len(st.session_state.questions)} questions generated**")
            
            st.markdown("---")
            
            # Separate SAQ and MCQ questions
            saq_questions = [q for q in st.session_state.questions if q.get('type') == 'Short Answer']
            mcq_questions = [q for q in st.session_state.questions if q.get('type') == 'Multiple Choice']
            
            # Display Short Answer Questions
            if saq_questions:
                st.subheader("📝 Short Answer Questions")
                for i, question in enumerate(saq_questions, 1):
                    # Check if question is valid
                    if not isinstance(question, dict):
                        st.error(f"Invalid SAQ format at position {i}")
                        continue
                    
                    st.markdown(f"### 📝 SAQ {i}")
                    question_text = question.get('question', 'No question text')
                    st.markdown(f"**Q{i}:** {question_text}")
                    
                    # Answer section
                    with st.expander("💡 Show Answer", expanded=False):
                        answer_text = question.get('answer', 'No answer provided')
                        st.markdown(f"**Answer:** {answer_text}")
                        points = question.get('points', 0)
                        st.caption(f"📊 Points: {points}")
                    
                    st.markdown("---")
            
            # Display Multiple Choice Questions  
            if mcq_questions:
                st.subheader("🎯 Multiple Choice Questions")
                for i, question in enumerate(mcq_questions, 1):
                    # Check if question is valid
                    if not isinstance(question, dict):
                        st.error(f"Invalid MCQ format at position {i}")
                        continue
                    
                    st.markdown(f"### 🎯 MCQ {i}")
                    question_text = question.get('question', 'No question text')
                    st.markdown(f"**Q{i}:** {question_text}")
                    

                    # MCQ Options
                    st.markdown("**Options:**")
                    options = question.get('options', {})
                    correct = question.get('correct_answer', '')
                    
                    # Display options with better formatting
                    for letter in ['A', 'B', 'C', 'D']:
                        option_text = options.get(letter, 'No option')
                        if letter == correct:
                            st.success(f"✅ **{letter})** {option_text} **(CORRECT)**")
                        else:
                            st.write(f"◯ **{letter})** {option_text}")
                    
                    with st.expander("💡 Show Explanation", expanded=False):
                        st.markdown(f"**Correct Answer:** {correct}")
                        explanation = question.get('explanation', 'No explanation provided')
                        st.markdown(f"**Explanation:** {explanation}")
                        points = question.get('points', 0)
                        st.caption(f"📊 Points: {points}")
                    
                    st.markdown("---")            # Export options
            st.subheader("Export Questions")
            export_handler = ExportHandler()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📥 Export as JSON"):
                    filepath = export_handler.export_to_json(st.session_state.questions)
                    st.success(f"Exported to: {filepath}")
            
            with col2:
                if st.button("📥 Export as CSV"):
                    filepath = export_handler.export_to_csv(st.session_state.questions)
                    st.success(f"Exported to: {filepath}")
            
            with col3:
                if st.button("📥 Export as TXT"):
                    filepath = export_handler.export_to_txt(st.session_state.questions)
                    st.success(f"Exported to: {filepath}")
            
            with col4:
                if st.button("📥 Export as Moodle XML"):
                    filepath = export_handler.export_to_moodle_xml(st.session_state.questions)
                    st.success(f"Exported to: {filepath}")
        
        else:
            st.info("📝 No questions generated yet.")
            st.markdown("""
            **To get started:**
            1. 📄 Upload a PDF file in the 'Upload & Generate' tab
            2. ⚙️ Configure your question settings in the sidebar  
            3. 🚀 Click 'Generate Questions'
            4. 🔄 Return to this tab to view your questions
            """)
            
            # Reset option if processing completed but no questions
            if st.session_state.processing_complete:
                st.warning("⚠️ Processing completed but questions not displayed.")
                if st.button("🔧 Reset Session"):
                    st.session_state.questions = []
                    st.session_state.processing_complete = False
                    st.rerun()
    
    with tab3:
        st.header("Quality Analytics")
        
        if st.session_state.questions:
            evaluator = QuestionEvaluator()
            
            # Generate evaluation metrics
            metrics = evaluator.evaluate_question_quality(st.session_state.questions)
            diversity = evaluator.calculate_diversity(st.session_state.questions)
            
            # Display main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Questions", metrics['total_questions'])
            with col2:
                st.metric("Valid Questions", metrics['valid_questions'])
            with col3:
                st.metric("Quality Score", f"{metrics['quality_score']:.1f}%")
            with col4:
                st.metric("Diversity Score", f"{diversity:.1f}%")
            
            # ROUGE and Quality Scores Section
            st.subheader("📊 Advanced Quality Metrics")
            
            # Calculate ROUGE scores between questions and answers
            rouge_scores = []
            bert_scores = []
            
            for question in st.session_state.questions:
                q_text = question.get('question', '')
                a_text = question.get('answer', '')
                
                if q_text and a_text and len(q_text) > 10 and len(a_text) > 10:
                    try:
                        scores = evaluator.evaluate_with_reference(q_text, a_text)
                        if scores:
                            rouge_scores.append(scores)
                    except:
                        pass
            
            if rouge_scores:
                # Display average ROUGE scores
                col1, col2, col3, col4 = st.columns(4)
                
                avg_rouge1 = sum(s.get('rouge1', 0) for s in rouge_scores) / len(rouge_scores)
                avg_rouge2 = sum(s.get('rouge2', 0) for s in rouge_scores) / len(rouge_scores)
                avg_rougeL = sum(s.get('rougeL', 0) for s in rouge_scores) / len(rouge_scores)
                avg_bert = sum(s.get('bertscore_f1', 0) for s in rouge_scores) / len(rouge_scores)
                
                with col1:
                    st.metric("ROUGE-1", f"{avg_rouge1:.3f}")
                with col2:
                    st.metric("ROUGE-2", f"{avg_rouge2:.3f}")
                with col3:
                    st.metric("ROUGE-L", f"{avg_rougeL:.3f}")
                with col4:
                    if avg_bert > 0:
                        st.metric("BERT Score", f"{avg_bert:.3f}")
                    else:
                        st.metric("BERT Score", "N/A")
                        
                st.info("📝 ROUGE scores measure overlap between questions and answers. Higher scores indicate better coherence.")
            else:
                st.warning("⚠️ Could not calculate ROUGE/BERT scores. Ensure ROUGE library is installed: pip install rouge-score")
            
            # Question type distribution
            st.subheader("Question Type Distribution")
            if metrics['question_types']:
                st.bar_chart(metrics['question_types'])
            
            # Detailed Quality Analysis
            with st.expander("📈 Detailed Quality Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Question Length Analysis")
                    lengths = [len(q.get('question', '').split()) for q in st.session_state.questions]
                    if lengths:
                        st.write(f"Average Length: {sum(lengths)/len(lengths):.1f} words")
                        st.write(f"Min Length: {min(lengths)} words")
                        st.write(f"Max Length: {max(lengths)} words")
                        
                        # Create histogram data
                        import pandas as pd
                        df = pd.DataFrame({'Length': lengths})
                        st.bar_chart(df['Length'].value_counts())
                
                with col2:
                    st.subheader("Answer Quality")
                    answer_lengths = [len(q.get('answer', '').split()) for q in st.session_state.questions if q.get('answer')]
                    if answer_lengths:
                        st.write(f"Average Answer Length: {sum(answer_lengths)/len(answer_lengths):.1f} words")
                        st.write(f"Questions with Answers: {len(answer_lengths)}/{len(st.session_state.questions)}")
                    
                    # Question validity breakdown
                    valid_questions = [q for q in st.session_state.questions if evaluator._is_valid_question(q)]
                    st.write(f"Validity Rate: {len(valid_questions)}/{len(st.session_state.questions)} ({len(valid_questions)/len(st.session_state.questions)*100:.1f}%)")
            
            # Detailed report
            with st.expander("📄 Full Report"):
                report = evaluator.generate_report(st.session_state.questions)
                st.code(report)
            
            # Feedback section
            st.subheader("Provide Feedback")
            st.markdown("Rate the quality of generated questions:")
            rating = st.slider("Overall Rating", 1, 5, 3)
            feedback = st.text_area("Additional Comments")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")
        
        else:
            st.info("Generate questions first to see analytics!")

def generate_questions(pdf_path, num_saq, num_mcq, temperature, chunk_size, enable_cache):
    """Main question generation pipeline"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load model
        status_text.text("Loading model...")
        progress_bar.progress(10)
        llama = load_model()
        
        # Step 2: Extract text
        status_text.text("Extracting text from PDF...")
        progress_bar.progress(20)
        extractor = PDFExtractor()
        raw_text = extractor.extract_text(str(pdf_path))
        
        # Step 3: Clean text
        status_text.text("Cleaning and preprocessing text...")
        progress_bar.progress(30)
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean(raw_text)
        sentences = cleaner.segment_sentences(cleaned_text)
        
        # Step 4: Chunk text
        status_text.text("Chunking text...")
        progress_bar.progress(40)
        chunker = TextChunker(chunk_size=chunk_size)
        chunks = chunker.chunk_by_sentences(sentences)
        
        st.info(f"Created {len(chunks)} text chunks for processing")
        
        # Step 4.5: Create vector store if RAG is enabled
        vector_store = None
        retriever = None
        if st.session_state.enable_rag:
            status_text.text("Building RAG vector store...")
            progress_bar.progress(45)
            
            try:
                vector_store = VectorStore()
                vector_store.create_index(chunks)
                retriever = DocumentRetriever(vector_store)
                st.session_state.vector_store = vector_store
                st.success("✓ RAG vector store created successfully!")
            except Exception as e:
                st.warning(f"RAG initialization failed: {str(e)}. Continuing without RAG.")
                st.session_state.enable_rag = False
        
        # Check cache
        cache_manager = CacheManager()
        cache_params = {
            'num_saq': num_saq,
            'num_mcq': num_mcq,
            'temperature': temperature,
            'chunk_size': chunk_size
        }
        
        cached_data = None
        if enable_cache:
            cached_data = cache_manager.get(cleaned_text[:500], cache_params)
        
        if cached_data:
            st.success("[OK] Using cached results!")
            all_questions = cached_data['questions']
            progress_bar.progress(100)
        else:
            # Step 5: Generate SAQs
            all_questions = []
            
            if num_saq > 0:
                status_text.text(f"Generating {num_saq} Short Answer Questions...")
                progress_bar.progress(50)
                
                # Check if RAG is enabled and use RAG generator
                if st.session_state.enable_rag and retriever and (st.session_state.gemini_model or st.session_state.llama_model):
                    st.info("Using RAG-enhanced SAQ generation")
                    llm_for_rag = st.session_state.gemini_model or st.session_state.llama_model
                    rag_gen = RAGQuestionGenerator(llm_for_rag, retriever)
                    use_rag = True
                else:
                    # Use transformers for SAQ generation
                    saq_generator = SAQGenerator()  # Will use transformers by default
                    use_rag = False
                
                # Process chunks dynamically until we have enough questions
                # Start with fewer chunks but process more if needed
                questions_per_chunk = 2  # Request 2 questions per chunk to be efficient
                max_chunks_available = len(chunks)
                chunks_processed = 0
                
                st.info(f"Generating {num_saq} SAQ questions...")
                
                # Process chunks until we have enough questions (with a safety limit)
                max_iterations = min(max_chunks_available, num_saq * 2)  # Safety limit
                
                for i in range(max_iterations):
                    if chunks_processed >= max_chunks_available:
                        st.warning(f"Processed all {max_chunks_available} chunks but only got {len(all_questions)} SAQ questions")
                        break
                        
                    if len(all_questions) >= num_saq:
                        break
                    
                    chunk = chunks[chunks_processed]
                    chunks_processed += 1
                    
                    if use_rag:
                        status_text.text(f"Processing SAQ chunk {chunks_processed}/{max_chunks_available} ({len(all_questions)}/{num_saq} questions) (RAG-Enhanced)...")
                    else:
                        status_text.text(f"Processing SAQ chunk {chunks_processed}/{max_chunks_available} ({len(all_questions)}/{num_saq} questions) (Transformers)...")
                    
                    try:
                        if use_rag:
                            saq_questions = rag_gen.generate_saq_with_rag(chunk, questions_per_chunk)
                        else:
                            saq_questions = saq_generator.generate_questions(chunk, questions_per_chunk)
                        
                        if saq_questions:
                            all_questions.extend(saq_questions)
                        else:
                            st.warning(f"No questions generated from chunk {chunks_processed}")
                            
                    except Exception as e:
                        st.warning(f"Error processing SAQ chunk {chunks_processed}: {str(e)}")
                        continue
                    
                    progress_bar.progress(50 + int((len(all_questions) / num_saq) * 20))
                
                # Trim to requested number
                all_questions = all_questions[:num_saq]
                
                st.success(f"✅ Generated {len(all_questions)} SAQ questions from {chunks_processed} chunks")
                
                if len(all_questions) < num_saq:
                    st.warning(f"⚠️ Only generated {len(all_questions)} out of {num_saq} requested SAQ questions. Try using a longer document or reduce the number of questions.")
            
            # Step 6: Generate MCQs
            if num_mcq > 0:
                selected_llm = st.session_state.selected_llm
                mcq_model = None
                
                if selected_llm == "gemini" and st.session_state.gemini_model:
                    mcq_model = st.session_state.gemini_model
                    model_name = "Gemini"
                elif selected_llm == "llama" and st.session_state.llama_model:
                    mcq_model = st.session_state.llama_model
                    model_name = "LLaMA"
                
                if mcq_model:
                    status_text.text(f"Generating {num_mcq} Multiple Choice Questions with {model_name}...")
                    progress_bar.progress(70)
                    
                    # Check if RAG is enabled
                    if st.session_state.enable_rag and retriever:
                        st.info("Using RAG-enhanced MCQ generation")
                        rag_gen_mcq = RAGQuestionGenerator(mcq_model, retriever)
                        use_rag_mcq = True
                    else:
                        mcq_generator = MCQGenerator(mcq_model)
                        use_rag_mcq = False
                    
                    # Process chunks dynamically to get the requested number
                    questions_per_chunk = 2  # Request 2 questions per chunk
                    max_chunks_available = len(chunks)
                    chunks_processed = 0
                    
                    st.info(f"Generating {num_mcq} MCQ questions using {model_name}...")
                    
                    mcq_questions = []
                    max_iterations = min(max_chunks_available, num_mcq * 2)  # Safety limit
                    
                    for i in range(max_iterations):
                        if chunks_processed >= max_chunks_available:
                            st.warning(f"Processed all {max_chunks_available} chunks but only got {len(mcq_questions)} MCQ questions")
                            break
                            
                        if len(mcq_questions) >= num_mcq:
                            break
                        
                        chunk = chunks[chunks_processed]
                        chunks_processed += 1
                        
                        if use_rag_mcq:
                            status_text.text(f"Processing MCQ chunk {chunks_processed}/{max_chunks_available} ({len(mcq_questions)}/{num_mcq} questions) ({model_name} + RAG)...")
                        else:
                            status_text.text(f"Processing MCQ chunk {chunks_processed}/{max_chunks_available} ({len(mcq_questions)}/{num_mcq} questions) ({model_name})...")
                        
                        try:
                            # Add timeout handling
                            if use_rag_mcq:
                                questions = rag_gen_mcq.generate_mcq_with_rag(chunk, questions_per_chunk)
                            else:
                                questions = mcq_generator.generate_questions(chunk, questions_per_chunk)
                            mcq_questions.extend(questions)
                            
                            # Stop early if we have enough questions
                            if len(mcq_questions) >= num_mcq:
                                st.info(f"Generated {len(mcq_questions)} questions, stopping early")
                                break
                                
                        except Exception as e:
                            st.warning(f"Error processing MCQ chunk {chunks_processed}: {str(e)}")
                            continue
                        
                        progress_bar.progress(70 + int((len(mcq_questions) / num_mcq) * 20))
                    
                    # Trim to requested number
                    mcq_questions = mcq_questions[:num_mcq]
                    all_questions.extend(mcq_questions)
                    
                    st.success(f"✅ Generated {len(mcq_questions)} MCQ questions from {chunks_processed} chunks using {model_name}")
                    
                    if len(mcq_questions) < num_mcq:
                        st.warning(f"⚠️ Only generated {len(mcq_questions)} out of {num_mcq} requested MCQ questions. Try using a longer document or reduce the number of questions.")
                else:
                    st.warning(f"MCQ generation requires {selected_llm.upper()} model. Please load the model first or only SAQ questions will be generated.")
                    progress_bar.progress(90)
            
            # Step 7: Validate questions
            status_text.text("Validating questions...")
            progress_bar.progress(90)
            
            validator = QuestionValidator()
            all_questions = validator.filter_valid_questions(all_questions)
            
            # Cache results
            if enable_cache and all_questions:
                cache_manager.set(cleaned_text[:500], cache_params, all_questions)
        
        # Step 8: Complete
        status_text.text("Complete!")
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state.questions = all_questions
        st.session_state.processing_complete = True
        
        # Show success message
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"[OK] Successfully generated {len(all_questions)} questions!")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error during generation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()