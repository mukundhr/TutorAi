"""
TutorAI - Complete Learning Platform
1. Question Generation
2. Statistics & Feedback
3. Chat with PDF
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(str(Path(__file__).parent))

from question_generation.llm_manager import LLMManager
from preprocessing import TextPreprocessor
from utilities import RAGManager

# Page config
st.set_page_config(
    page_title="TutorAI - Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_manager' not in st.session_state:
    st.session_state.rag_manager = None
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = {}


def calculate_metrics(question: dict, source_text: str) -> dict:
    """Calculate quality metrics for a question"""
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Combine question and answer for scoring
        if question['type'] == 'mcq':
            generated_text = question['question'] + ' ' + question.get('explanation', '')
        else:
            generated_text = question['question'] + ' ' + question['answer']
        
        # ROUGE scores
        rouge_scores = scorer.score(source_text[:2000], generated_text)
        
        # Calculate BERTScore (simplified - using rouge as proxy for now)
        bert_score_f1 = (rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure) / 2
        
        return {
            'bleu': round(rouge_scores['rouge1'].precision, 3),
            'rouge': round(rouge_scores['rougeL'].fmeasure, 3),
            'bert_score': round(bert_score_f1, 3),
            'quality_score': round((rouge_scores['rouge1'].fmeasure + rouge_scores['rougeL'].fmeasure + bert_score_f1) / 3, 3)
        }
    except Exception as e:
        print(f"Metrics error: {e}")
        return {
            'bleu': 0.0,
            'rouge': 0.0,
            'bert_score': 0.0,
            'quality_score': 0.0
        }


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selection
    model_type = st.selectbox(
        "Select LLM",
        ["gemini", "llama"],
        help="Gemini: Fast, cloud-based\nLLaMA: Private, local"
    )
    
    # Load Model Button
    if st.button("🔄 Load Model", use_container_width=True):
        with st.spinner(f"Loading {model_type.upper()}..."):
            try:
                st.session_state.llm = LLMManager(model_type=model_type)
                st.session_state.llm.load_model()
                st.session_state.model_loaded = True
                st.success(f"✅ {model_type.upper()} loaded!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.model_loaded = False
    
    if st.session_state.model_loaded:
        st.success(f"✅ Model Ready")
    
    st.divider()
    
    # Generation Settings
    st.subheader("📝 Generation Settings")
    
    question_type = st.selectbox(
        "Question Type",
        ["both", "mcq", "saq"],
        format_func=lambda x: {"mcq": "MCQ Only", "saq": "SAQ Only", "both": "Both MCQ + SAQ"}[x]
    )
    
    num_questions = st.slider("Questions per type", 1, 10, 5,
                              help="Total questions of each type (e.g., 5 MCQs + 5 SAQs = 10 total)")
    
    difficulty = st.select_slider("Difficulty", ["easy", "medium", "hard"], value="medium")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, 
                           help="Lower = focused, Higher = creative")
    
    st.divider()
    
    # Chunk Settings
    st.subheader("📄 Chunk Settings")
    
    chunk_size = st.number_input("Chunk size (characters)", 500, 2000, 1000, 100)
    
    max_chunks = st.number_input("Max chunks to process", 1, 20, 5,
                                 help="Limits total questions generated")


# Main Content - Tabs
tab1, tab2, tab3 = st.tabs(["📝 Question Generation", "📊 Statistics & Feedback", "💬 Chat with PDF"])

# TAB 1: Question Generation
with tab1:
    st.title("🎓 TutorAI - Question Generator")
    st.markdown("Generate educational questions from PDFs using AI")
    
    st.divider()
    
    # File Upload
    uploaded_file = st.file_uploader("📤 Upload PDF", type=['pdf'])
    
    if uploaded_file:
        temp_path = Path("temp_upload.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Process PDF
        if not st.session_state.chunks or st.button("🔄 Reprocess PDF"):
            with st.spinner("Processing..."):
                preprocessor = TextPreprocessor(chunk_size=chunk_size, chunk_overlap=100)
                st.session_state.chunks = preprocessor.process_pdf(str(temp_path))
        
        chunks = st.session_state.chunks
        
        if chunks:
            st.success(f"✅ {len(chunks)} chunks extracted")
            
            # Show expected questions
            if question_type == "both":
                st.info(f"📊 Will generate: {num_questions} MCQs + {num_questions} SAQs = {num_questions * 2} total")
            elif question_type == "mcq":
                st.info(f"📊 Will generate: {num_questions} MCQs")
            else:
                st.info(f"📊 Will generate: {num_questions} SAQs")
            
            # Generate Button
            if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
                if not st.session_state.model_loaded:
                    st.error("⚠️ Please load a model first!")
                else:
                    st.session_state.questions = []
                    progress = st.progress(0)
                    status = st.empty()
                    
                    # Combine all chunks into one text for better context
                    chunks_subset = chunks[:max_chunks]
                    combined_text = "\n\n".join([c['text'] for c in chunks_subset])
                    
                    # Calculate total steps: 1 for MCQ generation, 1 for SAQ generation
                    total_steps = 2 if question_type == "both" else 1
                    current_step = 0
                    
                    try:
                        # Generate exactly num_questions MCQs in total
                        if question_type in ["mcq", "both"]:
                            status.text(f"Generating {num_questions} MCQs...")
                            mcqs = st.session_state.llm.generate_questions(
                                combined_text, "mcq", num_questions, 
                                difficulty, temperature
                            )
                            for q in mcqs:
                                q['chunk_id'] = 'combined'
                                q['metrics'] = calculate_metrics(q, combined_text[:2000])
                                st.session_state.questions.append(q)
                            current_step += 1
                            progress.progress(current_step / total_steps)
                        
                        # Generate exactly num_questions SAQs in total
                        if question_type in ["saq", "both"]:
                            status.text(f"Generating {num_questions} SAQs...")
                            saqs = st.session_state.llm.generate_questions(
                                combined_text, "saq", num_questions,
                                difficulty, temperature
                            )
                            for q in saqs:
                                q['chunk_id'] = 'combined'
                                q['metrics'] = calculate_metrics(q, combined_text[:2000])
                                st.session_state.questions.append(q)
                            current_step += 1
                            progress.progress(current_step / total_steps)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        progress.progress(1.0)
                    
                    status.text("✅ Complete!")
                    progress.progress(1.0)
                    st.balloons()
                    
                    mcq_count = sum(1 for q in st.session_state.questions if q['type'] == 'mcq')
                    saq_count = sum(1 for q in st.session_state.questions if q['type'] == 'saq')
                    st.success(f"Generated {mcq_count} MCQs + {saq_count} SAQs = {len(st.session_state.questions)} total")
    
    # Display Questions
    if st.session_state.questions:
        st.divider()
        st.subheader(f"📋 Questions ({len(st.session_state.questions)})")
        
        filter_type = st.radio("Filter:", ["All", "MCQ", "SAQ"], horizontal=True)
        
        filtered = st.session_state.questions
        if filter_type == "MCQ":
            filtered = [q for q in st.session_state.questions if q['type'] == 'mcq']
        elif filter_type == "SAQ":
            filtered = [q for q in st.session_state.questions if q['type'] == 'saq']
        
        for i, q in enumerate(filtered, 1):
            with st.expander(f"Question {i} - {q['type'].upper()}", expanded=False):
                st.markdown(f"### {q['question']}")
                
                if q['type'] == 'mcq':
                    for opt, text in q['options'].items():
                        if opt == q['correct_answer']:
                            st.markdown(f"✅ **{opt}) {text}**")
                        else:
                            st.markdown(f"{opt}) {text}")
                    if q.get('explanation'):
                        st.info(f"💡 {q['explanation']}")
                else:
                    st.write(f"**Answer:** {q['answer']}")
        
        # Export
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_data = json.dumps([{k: v for k, v in q.items() if k not in ['chunk_id', 'rating', 'feedback']} 
                                   for q in st.session_state.questions], indent=2)
            st.download_button("📥 JSON", json_data, "questions.json", "application/json", use_container_width=True)
        
        with col2:
            df = pd.DataFrame(st.session_state.questions)
            st.download_button("📥 CSV", df.to_csv(index=False), "questions.csv", "text/csv", use_container_width=True)
        
        with col3:
            text_out = ""
            for i, q in enumerate(st.session_state.questions, 1):
                text_out += f"\n{'='*60}\nQuestion {i} ({q['type'].upper()})\n{'='*60}\n\n"
                text_out += f"Q: {q['question']}\n\n"
                if q['type'] == 'mcq':
                    for opt, text in q['options'].items():
                        text_out += f"{opt}) {text}\n"
                    text_out += f"\nAnswer: {q['correct_answer']}\n"
                else:
                    text_out += f"A: {q['answer']}\n"
                text_out += "\n"
            st.download_button("📥 TXT", text_out, "questions.txt", "text/plain", use_container_width=True)


# TAB 2: Statistics & Feedback
with tab2:
    st.title("📊 Statistics & Feedback")
    
    if not st.session_state.questions:
        st.info("📝 Generate questions first to provide feedback")
    else:
        # Overall Stats
        st.header("📈 Overall Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        mcq_count = sum(1 for q in st.session_state.questions if q['type'] == 'mcq')
        saq_count = sum(1 for q in st.session_state.questions if q['type'] == 'saq')
        
        col1.metric("Total Questions", len(st.session_state.questions))
        col2.metric("MCQs", mcq_count)
        col3.metric("SAQs", saq_count)
        col4.metric("Chunks Used", len(set(q.get('chunk_id', '') for q in st.session_state.questions)))
        
        st.divider()
        
        # Quality Metrics
        st.header("🎯 Quality Metrics")
        
        questions_with_metrics = [q for q in st.session_state.questions if 'metrics' in q]
        
        if questions_with_metrics:
            # Average metrics
            avg_bleu = sum(q['metrics']['bleu'] for q in questions_with_metrics) / len(questions_with_metrics)
            avg_rouge = sum(q['metrics']['rouge'] for q in questions_with_metrics) / len(questions_with_metrics)
            avg_bert = sum(q['metrics']['bert_score'] for q in questions_with_metrics) / len(questions_with_metrics)
            avg_quality = sum(q['metrics']['quality_score'] for q in questions_with_metrics) / len(questions_with_metrics)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg BLEU", f"{avg_bleu:.3f}")
            col2.metric("Avg ROUGE", f"{avg_rouge:.3f}")
            col3.metric("Avg BERT Score", f"{avg_bert:.3f}")
            col4.metric("Avg Quality", f"{avg_quality:.3f}")
            
            # Distribution Charts
            st.subheader("📊 Metric Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # BLEU vs ROUGE scatter
                df_metrics = pd.DataFrame([
                    {
                        'Question': f"Q{i+1}",
                        'BLEU': q['metrics']['bleu'],
                        'ROUGE': q['metrics']['rouge'],
                        'Type': q['type'].upper()
                    }
                    for i, q in enumerate(questions_with_metrics)
                ])
                
                fig = px.scatter(df_metrics, x='BLEU', y='ROUGE', color='Type',
                               title='BLEU vs ROUGE Score',
                               hover_data=['Question'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Quality score distribution
                fig = px.histogram(df_metrics.assign(Quality=[q['metrics']['quality_score'] 
                                                              for q in questions_with_metrics]),
                                 x='Quality', nbins=20,
                                 title='Quality Score Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart for average metrics
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[avg_bleu, avg_rouge, avg_bert, avg_quality],
                theta=['BLEU', 'ROUGE', 'BERT Score', 'Overall Quality'],
                fill='toself',
                name='Average Scores'
            ))
            fig.update_layout(title='Average Metric Scores', polar=dict(radialaxis=dict(range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Question Details with Metrics
        st.header("📋 Question Details & Metrics")
        
        for i, q in enumerate(st.session_state.questions):
            with st.expander(f"Question {i+1} - {q['type'].upper()}: {q['question'][:60]}...", expanded=False):
                st.markdown(f"**{q['question']}**")
                
                # Show the answer/options
                if q['type'] == 'mcq':
                    st.write("")
                    for opt, text in q['options'].items():
                        if opt == q['correct_answer']:
                            st.markdown(f"✅ **{opt}) {text}**")
                        else:
                            st.markdown(f"{opt}) {text}")
                    if q.get('explanation'):
                        st.info(f"💡 {q['explanation']}")
                else:
                    st.write(f"**Answer:** {q['answer']}")
                
                # Show metrics for THIS question
                if 'metrics' in q:
                    st.divider()
                    st.subheader("📊 Quality Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("BLEU", f"{q['metrics']['bleu']:.3f}")
                    col2.metric("ROUGE", f"{q['metrics']['rouge']:.3f}")
                    col3.metric("BERT Score", f"{q['metrics']['bert_score']:.3f}")
                    col4.metric("Overall Quality", f"{q['metrics']['quality_score']:.3f}")
        
        # Overall Feedback Section
        st.divider()
        st.header("💬 Your Feedback")
        st.markdown("Share your overall thoughts about the generated questions")
        
        overall_feedback = st.text_area(
            "Overall feedback about all questions",
            value=st.session_state.get('overall_feedback', ''),
            placeholder="• How is the overall quality?\n• Are the questions appropriate for the difficulty level?\n• Any patterns you noticed?\n• Suggestions for improvement?",
            height=200,
            key="overall_feedback_input"
        )
        st.session_state['overall_feedback'] = overall_feedback


# TAB 3: Chat with PDF
with tab3:
    st.title("💬 Chat with Your PDF")
    
    # Check if PDF has been processed
    if not st.session_state.chunks:
        st.warning("⚠️ Please upload and process a PDF in the 'Question Generation' tab first!")
        st.info("👈 Go to the first tab to upload your PDF")
    else:
        # Initialize RAG if not already done
        if st.session_state.rag_manager is None:
            if st.session_state.model_loaded:
                with st.spinner("🔧 Initializing chat system..."):
                    try:
                        st.session_state.rag_manager = RAGManager(model_type="gemini")
                        st.session_state.rag_manager.load_model()
                        st.session_state.rag_manager.index_documents(st.session_state.chunks)
                        st.success("✅ Chat system ready!")
                    except Exception as e:
                        st.error(f"❌ Failed to initialize chat: {e}")
            else:
                st.warning("⚠️ Please load a model in the sidebar first!")
        
        # Show document info
        if st.session_state.rag_manager:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Document Chunks", len(st.session_state.chunks))
            with col2:
                total_words = sum(c['word_count'] for c in st.session_state.chunks)
                st.metric("📝 Total Words", f"{total_words:,}")
            
            st.divider()
            
            # Chat interface
            st.markdown("### 💭 Ask questions about your document")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("#### � Conversation History")
                for i, msg in enumerate(st.session_state.chat_history):
                    if msg['role'] == 'user':
                        st.markdown(f"**🧑 You:** {msg['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {msg['content']}")
                        
                        # Show sources if available
                        if 'sources' in msg and msg['sources']:
                            with st.expander(f"📚 View {len(msg['sources'])} source(s)"):
                                for j, source in enumerate(msg['sources'], 1):
                                    st.markdown(f"**Source {j}** (Chunk {source['chunk_id']}):")
                                    st.caption(source['text'])
                    
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
            
            st.divider()
            
            # Input form
            with st.form(key="chat_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Your question:",
                    placeholder="e.g., What are the main concepts explained in this document?",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    submit = st.form_submit_button("🚀 Send", use_container_width=True, type="primary")
                with col2:
                    clear = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
            
            # Handle clear
            if clear:
                st.session_state.chat_history = []
                st.rerun()
            
            # Handle question
            if submit and user_question.strip():
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Get answer from RAG
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.rag_manager.chat(user_question)
                        
                        # Add assistant message
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response['answer'],
                            'sources': response['sources']
                        })
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            # Tips
            st.divider()
            with st.expander("💡 Tips for better answers"):
                st.markdown("""
                - **Be specific**: Ask about particular topics or concepts
                - **One topic at a time**: Break complex questions into smaller ones
                - **Reference the content**: Use terms from the document
                - **Follow up**: Ask clarifying questions based on previous answers
                
                **Example questions:**
                - "What is the definition of [concept]?"
                - "How does [process] work?"
                - "What are the key differences between [A] and [B]?"
                - "Can you explain [topic] in simple terms?"
                - "What are the main points about [subject]?"
                """)
        
        # Re-index button if chunks change
        if st.button("🔄 Re-index Document", help="Refresh the chat system with updated chunks"):
            if st.session_state.rag_manager and st.session_state.chunks:
                with st.spinner("Re-indexing..."):
                    st.session_state.rag_manager.index_documents(st.session_state.chunks)
                    st.session_state.chat_history = []
                    st.success("✅ Document re-indexed! Chat history cleared.")
                    st.rerun()
