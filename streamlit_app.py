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
import base64
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.let_it_rain import rain

# Set page configuration
st.set_page_config(
    page_title="Toxicity Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure paths
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "weights/toxic_classifier.onnx")
PYTORCH_MODEL_PATH = os.environ.get("PYTORCH_MODEL_PATH", "weights/toxic_classifier_xlm-roberta-large/pytorch_model.bin")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported languages with emoji flags
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'flag': '🇺🇸'},
    'ru': {'name': 'Russian', 'flag': '🇷🇺'},
    'tr': {'name': 'Turkish', 'flag': '🇹🇷'},
    'es': {'name': 'Spanish', 'flag': '🇪🇸'},
    'fr': {'name': 'French', 'flag': '🇫🇷'},
    'it': {'name': 'Italian', 'flag': '🇮🇹'},
    'pt': {'name': 'Portuguese', 'flag': '🇵🇹'}
}

# Language examples - toxic content examples for testing
LANGUAGE_EXAMPLES = {
    'en': "You are such an idiot, nobody likes your stupid content.",
    'ru': "Ты полный придурок, твой контент никому не нравится.",
    'tr': "Sen tam bir aptalsın, kimse senin aptalca içeriğini beğenmiyor.",
    'es': "Eres un completo idiota y nadie te quiere.",
    'fr': "Tu es tellement stupide, personne n'aime ton contenu minable.",
    'it': "Sei un tale idiota, a nessuno piace il tuo contenuto stupido.",
    'pt': "Você é um idiota completo, ninguém gosta do seu conteúdo estúpido."
}

# Theme colors - Fixed light theme
THEME = {
    "primary": "#4361EE",
    "secondary": "#3A0CA3",
    "background": "#FFFFFF",
    "surface": "#F8F9FA",
    "text": "#212529",
    "toxic": "#FF6B6B",
    "non_toxic": "#4CC9F0",
    "warning": "#F9C74F",
    "info": "#90BE6D"
}

# Custom CSS for better styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
        color: {THEME["text"]};
    }}
    
    .main-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, {THEME["primary"]} 0%, {THEME["secondary"]} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }}
    
    .subtitle {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        color: {THEME["text"]}aa;
        margin-bottom: 2rem;
    }}
    
    .category-label {{
        font-weight: 600;
    }}
    
    .toxic-category {{
        padding: 4px 12px;
        border-radius: 15px;
        background-color: {THEME["toxic"]}20;
        border: 1px solid {THEME["toxic"]}50;
        margin-right: 5px;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 5px;
        color: {THEME["toxic"]};
        transition: all 0.3s ease;
    }}
    
    .toxic-category:hover {{
        background-color: {THEME["toxic"]}40;
        transform: scale(1.05);
    }}
    
    .toxic-result {{
        font-size: 1.8rem;
        font-weight: 600;
        padding: 5px 15px;
        border-radius: 10px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    
    .toxic-result:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .model-info {{
        border-left: 3px solid {THEME["primary"]};
        padding-left: 10px;
        transition: all 0.3s ease;
    }}
    
    .model-info:hover {{
        border-left-width: 5px;
        background-color: {THEME["primary"]}10;
    }}
    
    .stButton button {{
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }}
    
    .example-btn {{
        background: linear-gradient(90deg, {THEME["primary"]}88 0%, {THEME["secondary"]}88 100%);
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-block;
        text-align: center;
    }}
    
    .example-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }}
    
    .stTextArea textarea {{
        border-radius: 8px;
        border: 1px solid {THEME["primary"]}66;
        padding: 8px;
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
    }}
    
    .stTextArea textarea:focus {{
        border-color: {THEME["primary"]};
        box-shadow: 0 0 0 2px {THEME["primary"]}33;
    }}
    
    div[data-testid="stExpander"] {{
        border-radius: 8px;
        border: 1px solid {THEME["text"]}22;
        overflow: hidden;
        transition: all 0.3s ease;
    }}
    
    div[data-testid="stExpander"]:hover {{
        border-color: {THEME["primary"]};
    }}
    
    div[data-testid="stVerticalBlock"] > div:has(div.stDataFrame) {{
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }}
    
    div[data-testid="stVerticalBlock"] > div:has(div.stDataFrame):hover {{
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }}
    
    div[data-testid="stTable"] {{
        border-radius: 8px;
        overflow: hidden;
    }}
    
    .threshold-bg {{
        padding: 15px;
        border-radius: 10px;
        background-color: {THEME["info"]}11;
        border-left: 3px solid {THEME["info"]};
    }}
    
    .usage-step {{
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 8px;
        background-color: {THEME["secondary"]}08;
        transition: all 0.3s ease;
    }}
    
    .usage-step:hover {{
        background-color: {THEME["secondary"]}15;
        transform: translateX(5px);
    }}
    
    .step-number {{
        font-size: 24px;
        font-weight: 700;
        color: {THEME["primary"]};
        margin-right: 15px;
        background-color: {THEME["primary"]}15;
        height: 40px;
        width: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
    }}
    
    .language-option {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .language-flag {{
        font-size: 1.2rem;
    }}
    
    @keyframes pulse {{
        0% {{
            box-shadow: 0 0 0 0 {THEME["primary"]}80;
        }}
        70% {{
            box-shadow: 0 0 0 10px {THEME["primary"]}00;
        }}
        100% {{
            box-shadow: 0 0 0 0 {THEME["primary"]}00;
        }}
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    .footer {{
        text-align: center;
        opacity: 0.7;
        padding: 20px;
        transition: all 0.3s ease;
    }}
    
    .footer:hover {{
        opacity: 1;
    }}
    
    /* Card styling */
    .info-card {{
        background-color: {THEME["surface"]};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border-left: 4px solid {THEME["primary"]};
        margin-bottom: 20px;
    }}
    
    .info-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }}
    
    .info-card h4 {{
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
        color: {THEME["primary"]};
    }}
    
    .example-container {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 10px;
        margin-bottom: 20px;
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
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
            # Get language code from the display name without flag
            selected_name = selected_language.split(' ')[1] if len(selected_language.split(' ')) > 1 else selected_language
            lang_code = next((code for code, info in SUPPORTED_LANGUAGES.items() 
                            if info['name'] == selected_name), 'en')
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

# Conditionally apply a different particle effect on successful analysis
def celebration_effect():
    if st.session_state.get('is_analysis_complete', False) and st.session_state.get('is_first_celebration', True):
        if not st.session_state.get('analysis_has_error', False):
            rain(emoji="🎉", font_size=40, falling_speed=5, animation_length=1)
            st.session_state['is_first_celebration'] = False

# Sidebar content
with st.sidebar:
    st.markdown("<h1 class='main-title'>Toxicity Analyzer</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🛡️ Multilingual Toxicity Analyzer
    
    This app analyzes text for different types of toxicity across multiple languages with high accuracy.
    """)
    
    # Create language cards with flags
    st.markdown("#### Supported Languages:")
    lang_cols = st.columns(2)
    
    for i, (code, info) in enumerate(SUPPORTED_LANGUAGES.items()):
        col_idx = i % 2
        with lang_cols[col_idx]:
            st.markdown(f"<div class='language-option'><span class='language-flag'>{info['flag']}</span> {info['name']}</div>", 
                      unsafe_allow_html=True)
    
    with st.expander("📊 Toxicity Categories", expanded=False):
        categories = {
            "Toxic": "General toxicity in language",
            "Severe Toxic": "Extreme forms of harmful content",
            "Obscene": "Content containing obscene language/references",
            "Threat": "Content implying threats or intimidation",
            "Insult": "Content aimed at insulting individuals/groups",
            "Identity Hate": "Content targeting specific identities"
        }
        
        for cat, desc in categories.items():
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <strong>{cat}:</strong> {desc}
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("🔍 How it works", expanded=False):
        st.markdown("""
        This model uses XLM-RoBERTa with language-aware fine-tuning to detect toxicity across multiple languages.
        
        The classifier has been optimized for performance using ONNX Runtime for fast inference.
        """)
        
        st.markdown("""
        <div class='info-card' style='background-color: #172d43;'>
            <h4>Technical Details</h4>
            <ul>
                <li>Architecture: XLM-RoBERTa Large</li>
                <li>Languages: 7 supported languages</li>
                <li>Max Sequence Length: 128 tokens</li>
                <li>Training Dataset: Multilingual toxicity data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Model information
    st.markdown("<div class='model-info'>", unsafe_allow_html=True)
    model_type = st.session_state.get('model_type', 'Loading...')
    st.markdown(f"**Model type:** {model_type}")
    st.markdown(f"**Device:** {DEVICE}")
    st.markdown("</div>", unsafe_allow_html=True)

# Display model loading status
if 'model_loaded' not in st.session_state:
    with st.spinner("🔄 Loading model..."):
        classifier = load_classifier()
        if classifier:
            st.session_state['model_loaded'] = True
            model_type = st.session_state.get('model_type', 'Unknown')
            st.success(f"✅ {model_type} model loaded successfully on {DEVICE}")
            # Trigger celebration only first time
            st.session_state['is_first_celebration'] = True
        else:
            st.session_state['model_loaded'] = False
            st.error("❌ Failed to load model. Please check logs.")
else:
    # Model already loaded, just get it from cache
    classifier = load_classifier()

# Main app
st.markdown("<h1 class='main-title'>Multilingual Toxicity Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle' style='color: white;'>Detect toxic content in multiple languages with state-of-the-art accuracy</p>", unsafe_allow_html=True)

# # Header card with instructions
# with stylable_container(
#     key="header_container",
#     css_styles="""
#         {
#             border-radius: 10px;
#             padding: 15px;
#             background-color: rgba(67, 97, 238, 0.1);
#             margin-bottom: 20px;
#         }
#     """
# ):
#     st.markdown("""
#     Enter text in any supported language to analyze it for toxic content. The model detects six different categories of toxicity 
#     with high accuracy and is optimized for cross-lingual performance.
#     """)

# Initialize session state for example selection if not present
if 'use_example' not in st.session_state:
    st.session_state['use_example'] = False
    st.session_state['example_text'] = ""
    st.session_state['detected_lang'] = "Auto-detect"

# Language selection dropdown (in a separate row)
language_options = ["Auto-detect"] + [f"{info['flag']} {info['name']}" for code, info in SUPPORTED_LANGUAGES.items()]
selected_language = st.selectbox(
    "Select Language",
    language_options,
    index=0,
    key="selected_language",
    help="Choose a specific language or use auto-detection"
)

# Display all examples in a grid layout
st.markdown("### Try with examples:")

# Create a 4x2 grid for all language examples
example_cols = st.columns(4)

# Order languages by putting the most common ones first
ordered_langs = ['en', 'es', 'fr', 'pt', 'it', 'ru', 'tr']

# Function to set example text
def set_example(lang_code):
    st.session_state['use_example'] = True
    st.session_state['example_text'] = LANGUAGE_EXAMPLES[lang_code]
    st.session_state['detected_lang'] = lang_code

for i, lang_code in enumerate(ordered_langs):
    info = SUPPORTED_LANGUAGES[lang_code]
    col_idx = i % 4
    
    with example_cols[col_idx]:
        if st.button(f"{info['flag']} {info['name']}", 
                   use_container_width=True, 
                   help=f"Try with a {info['name']} example of toxic content"):
            set_example(lang_code)

# Text input area with interactive styling
with stylable_container(
    key="text_input_container",
    css_styles="""
        {
            border-radius: 10px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
    """
):
    # Set the text input value from example if one was selected
    if st.session_state['use_example']:
        text_input = st.text_area(
            "Enter text to analyze",
            height=120,
            value=st.session_state['example_text'],
            key="text_input",
            help="Enter text in any supported language to analyze for toxicity"
        )
        # Reset the flag after using it
        st.session_state['use_example'] = False
    else:
        text_input = st.text_area(
            "Enter text to analyze",
            height=120,
            placeholder="Type or paste text here...",
            key="text_input",
            help="Enter text in any supported language to analyze for toxicity"
        )

# Analyze button with improved styling
analyze_button = st.button(
    "🔍 Analyze Text", 
    type="primary", 
    use_container_width=True,
    help="Click to analyze the entered text for toxicity"
)

# Process when button is clicked or text is submitted
if analyze_button or (text_input and 'last_analyzed' not in st.session_state or st.session_state.get('last_analyzed') != text_input):
    if text_input:
        st.session_state['last_analyzed'] = text_input
        prediction = predict_toxicity(text_input, selected_language)
        
        # Set analysis status flags for celebration effect
        st.session_state['is_analysis_complete'] = True
        st.session_state['analysis_has_error'] = "error" in prediction and prediction["error"]
        
        if "error" in prediction and prediction["error"]:
            st.error(prediction["error"])
        elif prediction["results"]:
            # Trigger celebration effect
            celebration_effect()
            
            results = prediction["results"]
            
            # Create metrics at the top
            metric_cols = st.columns(3)
            
            # Overall toxicity result
            is_toxic = results["is_toxic"]
            result_color = THEME["toxic"] if is_toxic else THEME["non_toxic"]
            result_text = "TOXIC" if is_toxic else "NON-TOXIC"
            
            # Language info
            lang_code = prediction["lang_code"]
            lang_info = SUPPORTED_LANGUAGES.get(lang_code, {"name": lang_code, "flag": "🌐"})
            
            # Count toxic categories
            toxic_count = len(results["toxic_categories"]) if is_toxic else 0
            
            with metric_cols[0]:
                st.metric("Analysis Result", result_text, delta=None)
            
            with metric_cols[1]:
                detected_text = " (detected)" if prediction["detected"] else ""
                st.metric("Language", f"{lang_info['flag']} {lang_info['name']}{detected_text}", delta=None)
            
            with metric_cols[2]:
                st.metric("Toxic Categories", f"{toxic_count}/6", delta=None)
                
            # Apply styling to metrics
            style_metric_cards(
                background_color=f"{result_color}20",
                border_left_color=result_color,
                border_color=f"{result_color}50", 
                box_shadow=True
            )
            
            st.divider()
            
            # Two column layout for results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Card with overall result and detected categories
                with stylable_container(
                    key="result_card",
                    css_styles=f"""
                        {{
                            border-radius: 10px;
                            padding: 20px;
                            background-color: {result_color}10;
                            border-left: 5px solid {result_color};
                            margin-bottom: 20px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        }}
                    """
                ):
                    # Overall result with animated highlight
                    st.markdown(f"""
                    <h2>Analysis Result: <span class='toxic-result pulse' style='background-color: {result_color}20; color: {result_color};'>{result_text}</span></h2>
                    <h3>Language: {lang_info['flag']} {lang_info['name']} {'(detected)' if prediction["detected"] else ''}</h3>
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
                            help="Percentage likelihood of toxicity category"
                        ),
                        "Status": st.column_config.TextColumn("Status", 
                                                             help="Whether the category was detected"),
                    }
                )
            
            with col2:
                # Create a horizontal bar chart with Plotly
                fig = go.Figure()
                
                # Sort data for the chart to match table sorting
                chart_data = sorted(zip(categories, probabilities, statuses), key=lambda x: x[1], reverse=True)
                chart_cats, chart_probs, chart_statuses = zip(*chart_data)
                
                # Add bars with different colors based on toxicity
                for i, (cat, prob, status) in enumerate(zip(chart_cats, chart_probs, chart_statuses)):
                    color = THEME["toxic"] if status == "DETECTED" else THEME["non_toxic"]
                    fig.add_trace(go.Bar(
                        y=[cat],
                        x=[prob],
                        orientation='h',
                        name=cat,
                        marker=dict(
                            color=color,
                            line=dict(
                                color=color + "dd",
                                width=1
                            )
                        ),
                        text=[f"{prob}%"],
                        textposition='auto',
                        hoverinfo='text',
                        hovertext=[f"{cat}: {prob}%"]
                    ))
                
                # Update layout
                fig.update_layout(
                    title={
                        'text': "Toxicity Probabilities",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(
                            size=18,
                            family="Poppins, sans-serif"
                        )
                    },
                    xaxis_title="Probability (%)",
                    yaxis_title="Category",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=30),
                    xaxis=dict(
                        range=[0, 100],
                        gridcolor=f"{THEME['text']}22",
                        zerolinecolor=f"{THEME['text']}33",
                    ),
                    yaxis=dict(
                        gridcolor=f"{THEME['text']}22",
                    ),
                    bargap=0.2,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(
                        family="Poppins, sans-serif",
                        color=THEME["text"]
                    ),
                    hoverlabel=dict(
                        font=dict(
                            family="Poppins, sans-serif",
                            size=14
                        ),
                        bordercolor=f"{THEME['text']}22",
                    )
                )
                
                # Add a light grid
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=f"{THEME['text']}11")
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                })
                
                # Add a section for threshold information with better styling
                with st.expander("About Toxicity Thresholds"):
                    st.markdown("""
                    <div class='threshold-bg'>
                    The model uses language-specific thresholds to determine if a text is toxic:
                    
                    - **Toxic**: 48-50%
                    - **Severe Toxic**: 45%
                    - **Obscene**: 47-50%
                    - **Threat**: 40-42%
                    - **Insult**: 46-50%
                    - **Identity Hate**: 40-42%
                    
                    These thresholds are optimized for best precision/recall balance based on evaluation data.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Small card with text info
                with stylable_container(
                    key="text_info_card",
                    css_styles=f"""
                        {{
                            border-radius: 8px;
                            padding: 15px;
                            background-color: {THEME["surface"]};
                            border: 1px solid {THEME["text"]}22;
                            margin-top: 20px;
                        }}
                    """
                ):
                    text_length = len(text_input)
                    word_count = len(text_input.split())
                    
                    st.markdown(f"""
                    <div style='display: flex; justify-content: space-between;'>
                        <div><strong>Text Length:</strong> {text_length} characters</div>
                        <div><strong>Word Count:</strong> {word_count} words</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Please enter some text to analyze.")

# Bottom section with improved styling for usage guide
st.divider()
colored_header(
    label="How to use this tool",
    description="Follow these steps to analyze text for toxicity",
    color_name="blue-70"
)

# Steps with more engaging design
st.markdown("""
<div class='usage-step'>
    <div class='step-number'>1</div>
    <div>Enter text in the input box above. You can type directly or paste from another source.</div>
</div>

<div class='usage-step'>
    <div class='step-number'>2</div>
    <div>Select a specific language or use the auto-detect feature if you're unsure.</div>
</div>

<div class='usage-step'>
    <div class='step-number'>3</div>
    <div>Click "Analyze Text" to get detailed toxicity analysis results.</div>
</div>

<div class='usage-step'>
    <div class='step-number'>4</div>
    <div>Examine the breakdown of toxicity categories, probabilities, and visualization.</div>
</div>
""", unsafe_allow_html=True)

# Adding footer with credits and improved styling
st.markdown("""
<div class='footer'>
    <div>Powered by XLM-RoBERTa | Enhanced Streamlit UI</div>
    <div style='font-size: 0.9rem; margin-top: 5px;'>v2.0.0 - Toxicity Analyzer</div>
</div>
""", unsafe_allow_html=True) 