import streamlit as st
import torch
import numpy as np
import os
import json
import plotly.graph_objects as go
import pandas as pd
from model.inference_optimized import OptimizedToxicityClassifier
import langid
from typing import List, Dict
import time
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container

# Set page configuration
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure paths
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "weights/toxic_classifier.onnx")
PYTORCH_MODEL_PATH = os.environ.get("PYTORCH_MODEL_PATH", "weights/toxic_classifier_xlm-roberta-large/pytorch_model.bin")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'ru': 'Russian',
    'tr': 'Turkish',
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese'
}

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Segoe UI', Tahoma, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .category-label {
        font-weight: 600;
    }
    .toxic-category {
        padding: 4px 12px;
        border-radius: 15px;
        background-color: rgba(255, 107, 107, 0.2);
        border: 1px solid rgba(255, 107, 107, 0.5);
        margin-right: 5px;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 5px;
    }
    .toxic-result {
        font-size: 1.8rem;
        font-weight: 600;
        padding: 5px 15px;
        border-radius: 10px;
        display: inline-block;
    }
    .model-info {
        border-left: 3px solid #4361ee;
        padding-left: 10px;
    }
    .progress-bar {
        width: 100%;
        height: 30px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load model at app start
@st.cache_resource
def load_classifier():
    try:
        if os.path.exists(ONNX_MODEL_PATH):
            classifier = OptimizedToxicityClassifier(onnx_path=ONNX_MODEL_PATH, device=DEVICE)
            st.session_state['model_type'] = 'ONNX'
            return classifier
        else:
            classifier = OptimizedToxicityClassifier(pytorch_path=PYTORCH_MODEL_PATH, device=DEVICE)
            st.session_state['model_type'] = 'PyTorch'
            return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Display model loading status
with st.spinner("Loading model..."):
    classifier = load_classifier()
    if classifier:
        st.session_state['model_loaded'] = True
        model_type = st.session_state.get('model_type', 'Unknown')
        st.success(f"✅ {model_type} model loaded successfully on {DEVICE}")
    else:
        st.session_state['model_loaded'] = False
        st.error("❌ Failed to load model. Please check logs.")

def detect_language(text: str) -> str:
    """Detect language of input text"""
    try:
        lang, _ = langid.classify(text)
        return lang if lang in SUPPORTED_LANGUAGES else 'en'
    except:
        return 'en'

def predict_toxicity(text: str, selected_language: str = "Auto-detect") -> Dict:
    """Predict toxicity of input text"""
    if not text or not text.strip():
        return {
            "error": "Please enter some text to analyze.",
            "results": None
        }
        
    if not st.session_state.get('model_loaded', False):
        return {
            "error": "Model not loaded. Please check logs.",
            "results": None
        }
    
    # Add a spinner while processing
    with st.spinner("Analyzing text..."):
        # Simulate a small delay to show the spinner (can be removed in production)
        time.sleep(0.5)
        
        # Detect language if auto-detect is selected
        if selected_language == "Auto-detect":
            lang_code = detect_language(text)
            detected = True
        else:
            # Convert from display name to code
            lang_code = next((code for code, name in SUPPORTED_LANGUAGES.items() 
                            if name == selected_language), 'en')
            detected = False
        
        # Run prediction
        try:
            results = classifier.predict([text], langs=[lang_code])[0]
            return {
                "results": results,
                "detected": detected,
                "lang_code": lang_code
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": f"Error processing text: {str(e)}",
                "results": None
            }

# Sidebar content
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/security-shield.png", width=80)
    st.markdown("<h1 class='main-title'>About</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🛡️ Multilingual Toxic Comment Classifier
    
    This app analyzes text for different types of toxicity across multiple languages.
    
    #### Supported languages:
    - English
    - Russian
    - Turkish
    - Spanish
    - French
    - Italian
    - Portuguese
    """)
    
    with st.expander("📊 Toxicity Categories", expanded=False):
        st.markdown("""
        This model classifies text into six toxicity categories:
        
        - **Toxic**: General toxicity
        - **Severe Toxic**: Extreme toxicity
        - **Obscene**: Obscene content
        - **Threat**: Threatening content
        - **Insult**: Insulting content
        - **Identity Hate**: Identity-based hate
        """)
    
    with st.expander("🔍 How it works", expanded=False):
        st.markdown("""
        This model uses XLM-RoBERTa with language-aware fine-tuning to detect toxicity across multiple languages.
        
        The classifier has been optimized for performance using ONNX Runtime for fast inference.
        
        **Technical Details:**
        - Architecture: XLM-RoBERTa Large
        - Languages: 7 supported languages
        - Max Sequence Length: 128 tokens
        """)
    
    st.divider()
    
    # Model information
    st.markdown("<div class='model-info'>", unsafe_allow_html=True)
    st.markdown(f"**Model type:** {st.session_state.get('model_type', 'Unknown')}")
    st.markdown(f"**Device:** {DEVICE}")
    st.markdown("</div>", unsafe_allow_html=True)

# Main app
colored_header(
    label="Multilingual Toxic Comment Classifier",
    description="Analyze text for toxic content in multiple languages",
    color_name="blue-70"
)

# Language selection and text input
col1, col2 = st.columns([3, 1])
with col1:
    # Display example buttons in a horizontal layout
    st.markdown("### Try with examples:")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("English Example"):
            st.session_state['text_input'] = "You are such an idiot, nobody likes your stupid content."
            st.session_state['selected_language'] = "Auto-detect"
    
    with example_col2:
        if st.button("Spanish Example"):
            st.session_state['text_input'] = "Eres un completo idiota y nadie te quiere."
            st.session_state['selected_language'] = "Auto-detect"
    
    with example_col3:
        if st.button("French Example"):
            st.session_state['text_input'] = "Tu es tellement stupide, personne n'aime ton contenu minable."
            st.session_state['selected_language'] = "Auto-detect"

with col2:
    # Language selection dropdown
    language_options = ["Auto-detect"] + list(SUPPORTED_LANGUAGES.values())
    selected_language = st.selectbox(
        "Select Language",
        language_options,
        index=0,
        key="selected_language"
    )

# Text input area
text_input = st.text_area(
    "Enter text to analyze",
    height=120,
    placeholder="Type or paste text here...",
    key="text_input",
    help="Enter text in any supported language to analyze for toxicity"
)

# Analyze button
analyze_button = st.button("📊 Analyze Text", type="primary", use_container_width=True)

# Process when button is clicked or text is submitted
if analyze_button or (text_input and 'last_analyzed' not in st.session_state or st.session_state.get('last_analyzed') != text_input):
    if text_input:
        st.session_state['last_analyzed'] = text_input
        prediction = predict_toxicity(text_input, selected_language)
        
        if "error" in prediction and prediction["error"]:
            st.error(prediction["error"])
        elif prediction["results"]:
            results = prediction["results"]
            
            # Two column layout for results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Overall result and language info
                is_toxic = results["is_toxic"]
                result_color = "#FF6B6B" if is_toxic else "#4CC9F0"
                result_text = "TOXIC" if is_toxic else "NON-TOXIC"
                
                st.markdown(f"""
                <h2>Analysis Result: <span class='toxic-result' style='background-color: {result_color}20; color: {result_color};'>{result_text}</span></h2>
                <h3>Language: {SUPPORTED_LANGUAGES.get(prediction["lang_code"], prediction["lang_code"])} {'(detected)' if prediction["detected"] else ''}</h3>
                """, unsafe_allow_html=True)
                
                # Display detected categories if toxic
                if is_toxic:
                    st.markdown("### Detected toxic categories:")
                    st.markdown("<div>", unsafe_allow_html=True)
                    for category in results["toxic_categories"]:
                        formatted_category = category.replace('_', ' ').title()
                        st.markdown(f"<span class='toxic-category'>{formatted_category}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Create a DataFrame for detailed results
                categories = []
                probabilities = []
                statuses = []
                
                for label, prob in results["probabilities"].items():
                    categories.append(label.replace('_', ' ').title())
                    probabilities.append(round(prob * 100, 1))
                    statuses.append("DETECTED" if prob >= 0.5 else "Not Detected")
                
                df = pd.DataFrame({
                    "Category": categories,
                    "Probability (%)": probabilities,
                    "Status": statuses
                })
                
                # Sort by probability
                df = df.sort_values(by="Probability (%)", ascending=False)
                
                # Display as a styled table
                st.markdown("### Detailed Results:")
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Category": st.column_config.TextColumn("Category"),
                        "Probability (%)": st.column_config.ProgressColumn(
                            "Probability (%)",
                            format="%f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "Status": st.column_config.TextColumn("Status"),
                    }
                )
            
            with col2:
                # Create a horizontal bar chart with Plotly
                fig = go.Figure()
                
                # Add bars with different colors based on toxicity
                for i, (cat, prob, status) in enumerate(zip(categories, probabilities, statuses)):
                    color = "#FF6B6B" if status == "DETECTED" else "#4CC9F0"
                    fig.add_trace(go.Bar(
                        y=[cat],
                        x=[prob],
                        orientation='h',
                        name=cat,
                        marker=dict(color=color),
                        text=[f"{prob}%"],
                        textposition='auto',
                        hoverinfo='text',
                        hovertext=[f"{cat}: {prob}%"]
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Toxicity Probabilities",
                    xaxis_title="Probability (%)",
                    yaxis_title="Category",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=30),
                    xaxis=dict(range=[0, 100]),
                    bargap=0.15,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a section for threshold information
                with st.expander("About Toxicity Thresholds"):
                    st.markdown("""
                    The model uses language-specific thresholds to determine if a text is toxic:
                    
                    - **Toxic**: 48-50%
                    - **Severe Toxic**: 45%
                    - **Obscene**: 47-50%
                    - **Threat**: 40-42%
                    - **Insult**: 46-50%
                    - **Identity Hate**: 40-42%
                    
                    These thresholds are optimized for best precision/recall balance based on evaluation data.
                    """)
    else:
        st.info("Please enter some text to analyze.")

# Bottom section for additional information
st.divider()
st.markdown("""
### How to use this tool
1. Enter text in the input box above
2. Select a language or use the auto-detect feature
3. Click "Analyze Text" to get results
4. Examine the detailed breakdown of toxicity categories and probabilities
""")

# Adding footer with credits
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 20px;">
    Powered by XLM-RoBERTa | Streamlit UI
</div>
""", unsafe_allow_html=True) 