# Fix for torch.classes watchdog errors
import sys
class ModuleProtector:
    def __init__(self, module_name):
        self.module_name = module_name
        self.original_module = sys.modules.get(module_name)
        
    def __enter__(self):
        if self.module_name in sys.modules:
            self.original_module = sys.modules[self.module_name]
            sys.modules[self.module_name] = None
            
    def __exit__(self, *args):
        if self.original_module is not None:
            sys.modules[self.module_name] = self.original_module

# Temporarily remove torch.classes from sys.modules to prevent Streamlit's file watcher from accessing it
with ModuleProtector('torch.classes'):
    import streamlit as st

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Multilingual Toxicity Analyzer",
    page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZC1wbHVzLWljb24gbHVjaWRlLXNoaWVsZC1wbHVzIj48cGF0aCBkPSJNMjAgMTNjMCA1LTMuNSA3LjUtNy42NiA4Ljk1YTEgMSAwIDAgMS0uNjctLjAxQzcuNSAyMC41IDQgMTggNCAxM1Y2YTEgMSAwIDAgMSAxLTFjMiAwIDQuNS0xLjIgNi4yNC0yLjcyYTEuMTcgMS4xNyAwIDAgMSAxLjUyIDBDMTQuNTEgMy44MSAxNyA1IDE5IDVhMSAxIDAgMCAxIDEgMXoiLz48cGF0aCBkPSJNOSAxMmg2Ii8+PHBhdGggZD0iTTEyIDl2NiIvPjwvc3ZnPg==",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import all other dependencies
import torch
import os
import plotly.graph_objects as go
import pandas as pd
from model.inference_optimized import OptimizedToxicityClassifier
import langid
from typing import List, Dict
import time
import psutil
import platform
try:
    import cpuinfo
except ImportError:
    cpuinfo = None
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.card import card
from streamlit_extras.metric_cards import style_metric_cards

# Configure paths
ONNX_MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "weights/toxic_classifier.onnx")
PYTORCH_MODEL_DIR = os.environ.get("PYTORCH_MODEL_DIR", "weights/toxic_classifier_xlm-roberta-large")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get GPU info if available
def get_gpu_info():
    if DEVICE == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
            cuda_version = torch.version.cuda
            
            memory_info = f"{gpu_memory_allocated:.1f}/{gpu_memory_total:.1f} GB"
            return f"{gpu_name} (CUDA {cuda_version}, Memory: {memory_info})"
        except Exception as e:
            return "CUDA device"
    return "CPU"

# Get CPU information
def get_cpu_info():
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        if cpu_freq:
            freq_info = f"{cpu_freq.current/1000:.2f} GHz"
        else:
            freq_info = "Unknown"
            
        # Try multiple methods to get CPU model name
        cpu_model = None
        
        # Method 1: Try reading from /proc/cpuinfo directly
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_model = line.split(':', 1)[1].strip()
                        break
        except:
            pass
            
        # Method 2: If Method 1 fails, try using platform.processor()
        if not cpu_model:
            cpu_model = platform.processor()
            
        # Method 3: If still no result, try using platform.machine()
        if not cpu_model or cpu_model == '':
            cpu_model = platform.machine()
            
        # Method 4: Final fallback to using psutil
        if not cpu_model or cpu_model == '':
            try:
                import cpuinfo
                cpu_model = cpuinfo.get_cpu_info()['brand_raw']
            except:
                pass
        
        # Clean up the model name
        if cpu_model:
            # Remove common unnecessary parts
            replacements = [
                '(R)', '(TM)', '(r)', '(tm)', 'CPU', '@', '  ', 'Processor'
            ]
            for r in replacements:
                cpu_model = cpu_model.replace(r, ' ')
            # Clean up extra spaces
            cpu_model = ' '.join(cpu_model.split())
            # Limit length
            if len(cpu_model) > 40:
                cpu_model = cpu_model[:37] + "..."
        else:
            cpu_model = "Unknown CPU"
        
        return {
            "name": cpu_model,
            "cores": cpu_count,
            "freq": freq_info,
            "usage": f"{cpu_percent:.1f}%"
        }
    except Exception as e:
        return {
            "name": "CPU",
            "cores": "Unknown",
            "freq": "Unknown",
            "usage": "Unknown"
        }

# Get RAM information
def get_ram_info():
    try:
        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024**3)  # Convert to GB
        ram_used = ram.used / (1024**3)    # Convert to GB
        ram_percent = ram.percent
        
        return {
            "total": f"{ram_total:.1f} GB",
            "used": f"{ram_used:.1f} GB",
            "percent": f"{ram_percent:.1f}%"
        }
    except Exception as e:
        return {
            "total": "Unknown",
            "used": "Unknown",
            "percent": "Unknown"
        }

# Update system resource information
def update_system_resources():
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    
    return {
        "cpu": cpu_info,
        "ram": ram_info
    }

# Initialize system information
GPU_INFO = get_gpu_info()
SYSTEM_INFO = update_system_resources()

# Add a function to update GPU memory info in real-time
def update_gpu_info():
    if DEVICE == "cuda":
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            return f"{gpu_memory_allocated:.1f}/{gpu_memory_total:.1f} GB"
        except:
            return "N/A"
    return "N/A"

# Helper function to convert hex to rgba
def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'

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

# Language examples - expanded with multiple examples per language, categorized as toxic or non-toxic
LANGUAGE_EXAMPLES = {
    'en': {
        'toxic': [
            "You are such an idiot, nobody likes your stupid content.",
            "Shut up you worthless piece of garbage. Everyone hates you.",
            "This is the most pathetic thing I've ever seen. Only losers would think this is good.",
            "Just kill yourself already, the world would be better without you."
        ],
        'non_toxic': [
            "I disagree with your opinion, but I appreciate your perspective.",
            "This content could use some improvement, but I see the effort you put into it.",
            "While I don't personally enjoy this type of content, others might find it valuable.",
            "Thank you for sharing your thoughts on this complex topic."
        ]
    },
    'ru': {
        'toxic': [
            "Ты полный придурок, твой контент никому не нравится.",
            "Заткнись, бесполезный кусок мусора. Все тебя ненавидят.",
            "Это самая жалкая вещь, которую я когда-либо видел. Только неудачники думают, что это хорошо.",
            "Почему бы тебе просто не исчезнуть нахрен? Никто не будет скучать по тебе."
        ],
        'non_toxic': [
            "Я не согласен с вашим мнением, но уважаю вашу точку зрения.",
            "Этот контент можно улучшить, но я вижу, сколько усилий вы в него вложили.",
            "Хотя мне лично не нравится такой контент, другие могут найти его полезным.",
            "Спасибо, что поделились своими мыслями на эту сложную тему."
        ]
    },
    'tr': {
        'toxic': [
            "Sen tam bir aptalsın, kimse senin aptalca içeriğini beğenmiyor.",
            "Kapa çeneni değersiz çöp parçası. Herkes senden nefret ediyor.",
            "Bu gördüğüm en acıklı şey. Sadece lanet olası kaybedenler bunun iyi olduğunu düşünür.",
            "Dünya sensiz daha iyi olurdu, kaybol git."
        ],
        'non_toxic': [
            "Fikrinize katılmıyorum ama bakış açınızı takdir ediyorum.",
            "Bu içerik biraz geliştirilebilir, ancak gösterdiğiniz çabayı görüyorum.",
            "Şahsen bu tür içerikten hoşlanmasam da, başkaları bunu değerli bulabilir.",
            "Bu karmaşık konu hakkındaki düşüncelerinizi paylaştığınız için teşekkür ederim."
        ]
    },
    'es': {
        'toxic': [
            "Eres un completo idiota y nadie te quiere.",
            "Cállate, pedazo de basura inútil. Todos te odian.",
            "Esto es lo más patético que he visto nunca. Solo los perdedores pensarían que esto es bueno.",
            "El mundo estaría mejor sin ti, deberías desaparecer, joder."
        ],
        'non_toxic': [
            "No estoy de acuerdo con tu opinión, pero aprecio tu perspectiva.",
            "Este contenido podría mejorarse, pero veo el esfuerzo que has puesto en él.",
            "Aunque personalmente no disfruto este tipo de contenido, otros podrían encontrarlo valioso.",
            "Gracias por compartir tus pensamientos sobre este tema tan complejo."
        ]
    },
    'fr': {
        'toxic': [
            "Tu es tellement stupide, personne n'aime ton contenu minable.",
            "Ferme-la, espèce de déchet inutile. Tout le monde te déteste.",
            "C'est la chose la plus pathétique que j'ai jamais vue. Seuls les loosers penseraient que c'est bien.",
            "Le monde serait meilleur sans toi, connard, va-t'en."
        ],
        'non_toxic': [
            "Je ne suis pas d'accord avec ton opinion, mais j'apprécie ta perspective.",
            "Ce contenu pourrait être amélioré, mais je vois l'effort que tu y as mis.",
            "Bien que personnellement je n'apprécie pas ce type de contenu, d'autres pourraient le trouver précieux.",
            "Merci d'avoir partagé tes réflexions sur ce sujet complexe."
        ]
    },
    'it': {
        'toxic': [
            "Sei un tale idiota, a nessuno piace il tuo contenuto stupido.",
            "Chiudi quella bocca, pezzo di spazzatura inutile. Tutti ti odiano.",
            "Questa è la cosa più patetica che abbia mai visto. Solo i perdenti penserebbero che sia buona.",
            "Il mondo sarebbe migliore senza di te, sparisci."
        ],
        'non_toxic': [
            "Non sono d'accordo con la tua opinione, ma apprezzo la tua prospettiva.",
            "Questo contenuto potrebbe essere migliorato, ma vedo lo sforzo che ci hai messo.",
            "Anche se personalmente non apprezzo questo tipo di contenuto, altri potrebbero trovarlo utile.",
            "Grazie per aver condiviso i tuoi pensieri su questo argomento complesso."
        ]
    },
    'pt': {
        'toxic': [
            "Você é um idiota completo, ninguém gosta do seu conteúdo estúpido.",
            "Cale a boca, seu pedaço de lixo inútil. Todos te odeiam.",
            "Isso é a coisa mais patética que eu já vi. Só perdedores pensariam que isso é bom.",
            "O mundo seria melhor sem você, desapareça."
        ],
        'non_toxic': [
            "Eu discordo da sua opinião, mas aprecio sua perspectiva.",
            "Este conteúdo poderia ser melhorado, mas vejo o esforço que você colocou nele.",
            "Embora eu pessoalmente não goste deste tipo de conteúdo, outros podem achá-lo valioso.",
            "Obrigado por compartilhar seus pensamentos sobre este tema complexo."
        ]
    }
}

# Theme colors - Light theme with black text
THEME = {
    "primary": "#2D3142",
    "background": "#FFFFFF",
    "surface": "#FFFFFF",
    "text": "#000000",  # Changed to pure black for maximum contrast
    "text_secondary": "#FFFFFF",  # For text that needs to be white
    "button": "#000000",  # Dark black for buttons
    "toxic": "#E53935",  # Darker red for better contrast
    "non_toxic": "#2E7D32",  # Darker green for better contrast
    "warning": "#F57C00",  # Darker orange for better contrast
    "info": "#1976D2",  # Darker blue for better contrast
    "sidebar_bg": "#FFFFFF",
    "card_bg": "white",
    "input_bg": "#F8F9FA"
}

# Custom CSS for better styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    :root, html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
        color: {THEME["text"]} !important;
        border: 1px solid {THEME["text"]} !important;
        overflow: hidden;
    }}
    
    svg, path{{
        color: {THEME["text"]};
    }}
    
    /* Heading font styling */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
        color: {THEME["text"]};
    }}
    
    .st-emotion{{
        background-color: {THEME["background"]};
    }}
    
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: {THEME["text"]};
    }}

    /* Examples section styling */
    .examples-section {{
        margin-top: 15px;
        color: {THEME["text"]};
    }}
    
    .example-button {{
        text-align: left;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-bottom: 5px;
        transition: all 0.2s ease;
        background-color: {THEME["input_bg"]};
        border-radius: 8px;
        color: {THEME["text"]};
    }}
    
    .example-button:hover {{
        transform: translateX(3px);
        background-color: {hex_to_rgba(THEME["primary"], 0.1)};
    }}
    
    /* Style tab content */
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 1rem;
        color: {THEME["text"]};
    }}
    
    /* Tab content styling */
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
        color: {THEME["text"]};
    }}
    
    /* Style expandable sections */
    div[data-testid="stExpander"] {{
        margin-bottom: 10px !important;
        background-color: {THEME["card_bg"]};
        border: 1px solid {hex_to_rgba(THEME["text"], 0.1)};
        color: {THEME["text"]} !important;
    }}
    
    div[data-testid="stExpander"] div[data-testid="stExpanderContent"] {{
        max-height: 300px;
        overflow-y: auto;
        padding: 5px 10px;
    }}

    /* Hardware info styles */
    .hardware-info {{
        background-color: {hex_to_rgba(THEME["primary"], 0.05)};
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid {THEME["primary"]};
        color: {THEME["text"]};
    }}
    
    .hardware-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        color: {THEME["primary"]};
    }}
    
    /* Override Streamlit's default background */
    .stApp {{
        background-color: {THEME["background"]};
    }}
    
    .st-emotion-cache-h4xjwg{{
        background-color: {THEME["background"]};
        color: {THEME["text"]};
    }}
    
    /* Code editor and text areas */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        background-color: {THEME["input_bg"]};
        color: {THEME["text"]};
        font-family: 'Space Grotesk', sans-serif;
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {THEME["sidebar_bg"]};
        color: {THEME["text"]};
    }}
    
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] {{
        color: {THEME["text"]};
        background-color: {THEME["background"]};
    }}
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stButton label {{
        color: {THEME["text"]} !important;
        background-color: {THEME["background"]} !important;
    }}
    
    section[data-testid="stSidebar"] h3 {{
        color: {THEME["text"]};
        background-color: {THEME["background"]};
    }}
    
    section[data-testid="stSidebar"] .main-title {{
        color: {THEME["text"]};
        -webkit-text-fill-color: {THEME["text"]};
        background-color: {THEME["background"]};
    }}
    
    section[data-testid="stSidebar"] h1 {{
        color: {THEME["text"]};
        background-color: {THEME["background"]};
    }}
    
    .main-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: {THEME["text"]};
        margin-bottom: 1rem;
        letter-spacing: -0.03em;
    }}
    
    .subtitle {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 400;
        color: {THEME["text"]};
        margin-bottom: 2rem;
    }}
    
    .category-label {{
        font-weight: 600;
    }}
    
    .toxic-category {{
        padding: 3px 8px;
        border-radius: 12px;
        background-color: {hex_to_rgba(THEME["toxic"], 0.13)};
        border: 1px solid {hex_to_rgba(THEME["toxic"], 0.31)};
        margin-right: 5px;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 5px;
        font-size: 0.9rem;
        color: {THEME["toxic"]};
        transition: all 0.3s ease;
    }}
    
    .toxic-result {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 8px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        letter-spacing: -0.02em;
    }}
    
    .model-info {{
        border-left: 3px solid {THEME["primary"]};
        padding-left: 10px;
        transition: all 0.3s ease;
    }}
    
    .stButton button {{
        font-family: 'Space Grotesk', sans-serif;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        background-color: {THEME["button"]} !important;
        color: {THEME["text_secondary"]} !important;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        background-color: {hex_to_rgba(THEME["button"], 0.9)} !important;
    }}
    

    
    /* Cards for metrics at top */
    div[data-testid="metric-container"] {{
        background-color: {THEME["card_bg"]};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid {hex_to_rgba(THEME["text"], 0.1)};
    }}
    
    /* Fix metric label colors - more specific selectors */
    div[data-testid="metric-container"] > div:first-child {{
        color: {THEME["text"]} !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }}
    
    div[data-testid="metric-container"] > div:first-child > label {{
        color: {THEME["text"]} !important;
    }}
    
    /* Target the specific label element */
    div[data-testid="stMetricLabel"] {{
        color: {THEME["text"]} !important;
        font-weight: 500 !important;
    }}
    
    div[data-testid="stMetricLabel"] > div {{
        color: {THEME["text"]} !important;
    }}
    
    /* Ensure metric value is also properly colored */
    div[data-testid="stMetricValue"] {{
        color: {THEME["text"]} !important;
        font-weight: 600 !important;
    }}
    
    /* Target all text within metric containers */
    div[data-testid="metric-container"] * {{
        color: {THEME["text"]} !important;
    }}
    
    /* Additional specific targeting for metric labels */
    [data-testid="metric-container-label"] {{
        color: {THEME["text"]} !important;
    }}
    
    [data-testid="metric-container-label-value"] {{
        color: {THEME["text"]} !important;
    }}
    
    /* Force black color on metric labels */
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] div[role="button"] {{
        color: {THEME["text"]} !important;
    }}
    
    /* Remove default Streamlit menu and footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Style dataframe */
    .stDataFrame {{
        background-color: {THEME["card_bg"]};
    }}
    
    .footer {{
        text-align: center;
        opacity: 0.7;
        padding: 20px;
        transition: all 0.3s ease;
        color: {THEME["text"]};
        font-family: 'Space Grotesk', sans-serif;
    }}
    
    .footer:hover {{
        opacity: 1;
    }}

    /* Fix dropdown text color */
    div[data-baseweb="select"] div[class*="valueContainer"] {{
        color: {THEME["text_secondary"]} !important;
    }}
    
    div[data-baseweb="select"] div[class*="placeholder"] {{
        color: {THEME["text_secondary"]} !important;
    }}
    
    div[data-baseweb="select"] div[class*="singleValue"] {{
        color: {THEME["text_secondary"]} !important;
    }}
    
    /* Fix button text color */
    .stButton > button {{
        background-color: {THEME["button"]} !important;
        color: {THEME["text_secondary"]} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        letter-spacing: 0.02em !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        background-color: {hex_to_rgba(THEME["button"], 0.9)} !important;
        color: {THEME["text_secondary"]} !important;
    }}
    
    /* Fix dropdown styling */
    div[data-baseweb="select"] {{
        background-color: {THEME["button"]} !important;
        border-color: {hex_to_rgba(THEME["button"], 0.5)} !important;
        border-radius: 8px !important;
    }}
    
    /* Fix dropdown options */
    div[data-baseweb="popover"] {{
        background-color: {THEME["text"]} !important;
        color: {THEME["text"]} !important;
    }}
    
    div[data-baseweb="popover"] div[role="option"] {{
        color: {THEME["text_secondary"]} !important;
    }}
    
    div[data-baseweb="popover"] div[role="option"]:hover {{
        background-color: {hex_to_rgba(THEME["button"], 0.7)} !important;
    }}
    
    /* Fix dropdown arrow color */
    div[data-baseweb="select"] svg {{
        color: {THEME["text_secondary"]} !important;
    }}
    
    /* Fix input label color */
    .stTextArea label {{
        color: {THEME["text"]} !important;
        font-weight: 500 !important;
    }}
    
    /* Fix selectbox label */
    .stSelectbox label {{
        color: {THEME["text"]} !important;
        font-weight: 500 !important;
    }}
    
    /* Ensure consistent black color */
    .stButton > button {{
        background-color: #000000 !important;
    }}
    
    div[data-baseweb="select"] {{
        background-color: #000000 !important;
    }}
    
    div[data-baseweb="popover"] {{
        background-color: #000000 !important;
    }}

    /* Style How to section */
    .usage-step {{
        background-color: {THEME["card_bg"]};
        border: 1px solid {hex_to_rgba(THEME["text"], 0.1)};
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }}

    .usage-step:hover {{
        transform: translateX(5px);
        border-color: {hex_to_rgba(THEME["button"], 0.3)};
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}

    .step-number {{
        background-color: {THEME["button"]};
        color: {THEME["text_secondary"]};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        min-width: 2.5rem;
        height: 2.5rem;
        border-radius: 30%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1.2rem;
    }}

    .usage-step div:last-child {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        color: {THEME["text"]}; 
        flex: 1;
        line-height: 1.4;
    }}

    /* Style the How to section header */
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}

    .colored-header {{
        margin: 2rem 0 1.5rem 0;
    }}

    .colored-header h1 {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: {THEME["text"]};
        margin-bottom: 0.5rem;
    }}

    .colored-header p {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        color: {hex_to_rgba(THEME["text"], 0.8)};
    }}

    /* Fix plotly chart axis labels */
    .js-plotly-plot .plotly .g-gtitle {{
        color: {THEME["text"]} !important;
    }}
    
    .js-plotly-plot .plotly .xtitle, 
    .js-plotly-plot .plotly .ytitle {{
        fill: {THEME["text"]} !important;
        color: {THEME["text"]} !important;
    }}
    
    .js-plotly-plot .plotly .xtick text, 
    .js-plotly-plot .plotly .ytick text {{
        fill: {THEME["text"]} !important;
        color: {THEME["text"]} !important;
    }}
    
    /* Fix tab text color */
    button[data-baseweb="tab"] {{
        color: {THEME["text"]} !important;
    }}
    
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {THEME["primary"]} !important;
    }}
    
    /* Fix expander arrow color */
    div[data-testid="stExpander"] svg {{
        color: {THEME["text"]} !important;
    }}
    
    /* Ensure plotly modebar buttons are visible */
    .modebar-btn path {{
        fill: {THEME["text"]} !important;
    }}
    
    /* Fix any remaining white text on white background issues */
    .element-container, .stMarkdown, .stText {{
        color: {THEME["text"]} !important;
    }}
    
    /* Ensure text inputs have black text */
    .stTextInput input, .stTextArea textarea {{
        color: {THEME["text"]} !important;
    }}
    
    /* Fix plotly legend text */
    .js-plotly-plot .plotly .legend text {{
        fill: {THEME["text"]} !important;
        color: {THEME["text"]} !important;
    }}

    /* Fix success message color */
    .stSuccess {{
        color: {THEME["text"]} !important;
    }}
    
    /* Fix success icon color */
    .stSuccess svg {{
        fill: {THEME["text"]} !important;
    }}
    
    /* Ensure all alert messages have proper text color */
    div[data-baseweb="notification"] {{
        color: {THEME["text"]} !important;
    }}
    
    /* Fix input area text color */
    textarea {{
        color: {THEME["text"]} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Custom CSS for metric labels - Add this near the top with the other CSS
st.markdown(f"""
<style>
/* Direct targeting of metric labels */
[data-testid="stMetricLabel"] {{
    color: {THEME["text"]} !important;
    font-weight: 500 !important;
}}

[data-testid="stMetricLabel"] span {{
    color: {THEME["text"]} !important;
    font-weight: 500 !important;
}}

/* Target the label content directly */
[data-testid="stMetricLabel"] div {{
    color: {THEME["text"]} !important;
}}

/* Target every element inside a metric label */
[data-testid="stMetricLabel"] * {{
    color: {THEME["text"]} !important;
}}

/* Style the value too */
[data-testid="stMetricValue"] {{
    color: {THEME["text"]} !important;
}}

/* Extremely specific selector to ensure it overrides everything */
div[data-testid="metric-container"] div[data-testid="stMetricLabel"] {{
    color: {THEME["text"]} !important;
}}
</style>
""", unsafe_allow_html=True)

# Load model at app start
@st.cache_resource
def load_classifier():
    try:
        if os.path.exists(ONNX_MODEL_PATH):
            classifier = OptimizedToxicityClassifier(onnx_path=ONNX_MODEL_PATH, device=DEVICE)
            st.session_state['model_type'] = 'Loaded'
            return classifier
        elif os.path.exists(PYTORCH_MODEL_DIR):
            classifier = OptimizedToxicityClassifier(pytorch_path=PYTORCH_MODEL_DIR, device=DEVICE)
            st.session_state['model_type'] = 'Loaded'
            return classifier
        else:
            st.error(f"❌ No model found at {ONNX_MODEL_PATH} or {PYTORCH_MODEL_DIR}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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
        # Record start time for inference metrics
        start_time = time.time()
        
        # Detect language if auto-detect is selected
        if selected_language == "Auto-detect":
            lang_detection_start = time.time()
            lang_code = detect_language(text)
            lang_detection_time = time.time() - lang_detection_start
            detected = True
        else:
            # Get language code from the display name without flag
            selected_name = selected_language.split(' ')[1] if len(selected_language.split(' ')) > 1 else selected_language
            lang_code = next((code for code, info in SUPPORTED_LANGUAGES.items() 
                            if info['name'] == selected_name), 'en')
            lang_detection_time = 0
            detected = False
        
        # Run prediction
        try:
            model_inference_start = time.time()
            results = classifier.predict([text], langs=[lang_code])[0]
            model_inference_time = time.time() - model_inference_start
            total_time = time.time() - start_time
            
            return {
                "results": results,
                "detected": detected,
                "lang_code": lang_code,
                "performance": {
                    "total_time": total_time,
                    "lang_detection_time": lang_detection_time,
                    "model_inference_time": model_inference_time
                }
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "error": f"Error processing text: {str(e)}",
                "results": None
            }

# Function to set example text
def set_example(lang_code, example_type, example_index=0):
    st.session_state['use_example'] = True
    # Get the example based on the language, type and index
    example = LANGUAGE_EXAMPLES[lang_code][example_type][example_index]
    st.session_state['example_text'] = example
    st.session_state['detected_lang'] = lang_code
    st.session_state['example_info'] = {
        'type': example_type,
        'lang': lang_code,
        'index': example_index
    }

# Initialize session state for example selection if not present
if 'use_example' not in st.session_state:
    st.session_state['use_example'] = False
    st.session_state['example_text'] = ""
    st.session_state['detected_lang'] = "Auto-detect"
    st.session_state['example_info'] = None

# Sidebar content
with st.sidebar:
    st.markdown("<h1 class='main-title'>Multilingual Toxicity Analyzer</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    #### This app analyzes text for different types of toxicity across multiple languages with high accuracy.
    """)
    
    # Create language cards with flags
    st.markdown("#### Supported Languages:")
    lang_cols = st.columns(2)
    
    for i, (code, info) in enumerate(SUPPORTED_LANGUAGES.items()):
        col_idx = i % 2
        with lang_cols[col_idx]:
            st.markdown(f"<div class='language-option'><span class='language-flag'>{info['flag']}</span> {info['name']}</div>", 
                      unsafe_allow_html=True)
    
    st.divider()
    
    # Language selection dropdown moved to sidebar
    st.markdown("### 🌐 Select Language")
    language_options = ["Auto-detect"] + [f"{info['flag']} {info['name']}" for code, info in SUPPORTED_LANGUAGES.items()]
    selected_language = st.selectbox(
        "Choose language or use auto-detect",
        language_options,
        index=0,
        key="selected_language",
        help="Choose a specific language or use auto-detection"
    )
    
    # Examples moved to sidebar
    st.markdown("### 📝 Try with examples:")
    
    # Create tabs for toxic and non-toxic examples
    example_tabs = st.tabs(["Toxic Examples", "Non-Toxic Examples"])
    
    # Order languages by putting the most common ones first
    ordered_langs = ['en', 'es', 'fr', 'pt', 'it', 'ru', 'tr']
    
    # Toxic examples tab
    with example_tabs[0]:
        st.markdown('<div class="examples-section">', unsafe_allow_html=True)
        for lang_code in ordered_langs:
            info = SUPPORTED_LANGUAGES[lang_code]
            with st.expander(f"{info['flag']} {info['name']} examples"):
                for i, example in enumerate(LANGUAGE_EXAMPLES[lang_code]['toxic']):
                    # Display a preview of the example
                    preview = example[:40] + "..." if len(example) > 40 else example
                    button_key = f"toxic_{lang_code}_{i}"
                    button_help = f"Try with this {info['name']} toxic example"
                    
                    # We can't directly apply CSS classes to Streamlit buttons, but we can wrap them
                    if st.button(f"Example {i+1}: {preview}", 
                            key=button_key,
                            use_container_width=True,
                            help=button_help):
                        set_example(lang_code, 'toxic', i)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Non-toxic examples tab
    with example_tabs[1]:
        st.markdown('<div class="examples-section">', unsafe_allow_html=True)
        for lang_code in ordered_langs:
            info = SUPPORTED_LANGUAGES[lang_code]
            with st.expander(f"{info['flag']} {info['name']} examples"):
                for i, example in enumerate(LANGUAGE_EXAMPLES[lang_code]['non_toxic']):
                    # Display a preview of the example
                    preview = example[:40] + "..." if len(example) > 40 else example
                    button_key = f"non_toxic_{lang_code}_{i}"
                    button_help = f"Try with this {info['name']} non-toxic example"
                    
                    if st.button(f"Example {i+1}: {preview}", 
                            key=button_key,
                            use_container_width=True,
                            help=button_help):
                        set_example(lang_code, 'non_toxic', i)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Model and Hardware information in the sidebar with improved layout
    st.markdown("### 💻 System Information", unsafe_allow_html=True)
    
    # Update system resources info
    current_sys_info = update_system_resources()
    
    # GPU section
    if DEVICE == "cuda":
        st.markdown("""
        <div class="hardware-info">
            <div class="hardware-title"><span class="icon">🎮</span> GPU</div>
            <div class="hardware-resource">
        """, unsafe_allow_html=True)
        
        gpu_name = GPU_INFO.split(" (")[0]
        st.markdown(f"<div class='hardware-stat'><span class='label'>Model:</span> <span class='value'>{gpu_name}</span></div>", unsafe_allow_html=True)
        
        cuda_version = "Unknown"
        if "CUDA" in GPU_INFO:
            cuda_version = GPU_INFO.split("CUDA ")[1].split(",")[0]
        st.markdown(f"<div class='hardware-stat'><span class='label'>CUDA:</span> <span class='value'>{cuda_version}</span></div>", unsafe_allow_html=True)
        
        current_gpu_memory = update_gpu_info()
        st.markdown(f"<div class='hardware-stat'><span class='label'>Memory:</span> <span class='value'>{current_gpu_memory}</span></div>", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # CPU section
    st.markdown("""
    <div class="hardware-info">
        <div class="hardware-title"><span class="icon">⚙️</span> CPU</div>
        <div class="hardware-resource">
    """, unsafe_allow_html=True)
    
    cpu_info = current_sys_info["cpu"]
    st.markdown(f"<div class='hardware-stat'><span class='label'>Model:</span> <span class='value'>{cpu_info['name']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hardware-stat'><span class='label'>Cores:</span> <span class='value'>{cpu_info['cores']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hardware-stat'><span class='label'>Frequency:</span> <span class='value'>{cpu_info['freq']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hardware-stat'><span class='label'>Usage:</span> <span class='value'>{cpu_info['usage']}</span></div>", unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # RAM section
    st.markdown("""
    <div class="hardware-info">
        <div class="hardware-title"><span class="icon">🧠</span> RAM</div>
        <div class="hardware-resource">
    """, unsafe_allow_html=True)
    
    ram_info = current_sys_info["ram"]
    st.markdown(f"<div class='hardware-stat'><span class='label'>Total:</span> <span class='value'>{ram_info['total']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hardware-stat'><span class='label'>Used:</span> <span class='value'>{ram_info['used']}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hardware-stat'><span class='label'>Usage:</span> <span class='value'>{ram_info['percent']}</span></div>", unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Toxicity Thresholds - Moved from results section to sidebar
    st.markdown("### ⚙️ Toxicity Thresholds")
    st.markdown("""
    <div class='threshold-bg'>
    The model uses language-specific thresholds to determine if a text is toxic:

    - **Toxic**: 60%
    - **Severe Toxic**: 54%
    - **Obscene**: 60%
    - **Threat**: 48%
    - **Insult**: 60%
    - **Identity Hate**: 50%

    These increased thresholds reduce false positives but may miss borderline toxic content.
    </div>
    """, unsafe_allow_html=True)

# Display model loading status
if 'model_loaded' not in st.session_state:
    with st.spinner("🔄 Loading model..."):
        classifier = load_classifier()
        if classifier:
            st.session_state['model_loaded'] = True
            st.success(f"✅ Model loaded successfully on {GPU_INFO}")
        else:
            st.session_state['model_loaded'] = False
            st.error("❌ Failed to load model. Please check logs.")
else:
    # Model already loaded, just get it from cache
    classifier = load_classifier()

# Main app
st.markdown("""
<h1 class='main-title'> 
    <svg xmlns="http://www.w3.org/2000/svg" style="padding-bottom: 10px;" width="45" height="45" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield-plus-icon lucide-shield-plus">
        <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
        <path d="M9 12h6"/>
        <path d="M12 9v6"/>
    </svg> 
    Multilingual Toxicity Analyzer
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<p class='subtitle'>Detect toxic content in multiple languages with state-of-the-art accuracy</p>
""", unsafe_allow_html=True)

# Text input area with interactive styling
with stylable_container(
    key="text_input_container",
    css_styles=f"""
        {{
            border-radius: 10px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            background-color: {THEME["card_bg"]};
            padding: 10px;
            margin-bottom: 15px;
        }}

        textarea {{
            caret-color: black !important;
            color: {THEME["text"]} !important;
        }}

        /* Ensure the text input cursor is visible */
        .stTextArea textarea {{
            caret-color: black !important;
        }}
    """
):
    # Get the current example text if it exists
    current_example = st.session_state.get('example_text', '')
    
    # Set the text input value, allowing for modifications
    text_input = st.text_area(
        "Enter text to analyze",
        height=80,
        value=current_example if st.session_state.get('use_example', False) else st.session_state.get('text_input', ''),
        key="text_input",
        help="Enter text in any supported language to analyze for toxicity"
    )
    
    # Check if the text has been modified from the example
    if st.session_state.get('use_example', False) and text_input != current_example:
        # Text was modified, clear example state
        st.session_state['use_example'] = False
        st.session_state['example_text'] = ""
        st.session_state['example_info'] = None

# Analyze button with improved styling in a more compact layout
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "Analyze Text", 
        type="primary", 
        use_container_width=True,
        help="Click to analyze the entered text for toxicity"
    )

# Process when button is clicked or text is submitted
if analyze_button or (text_input and 'last_analyzed' not in st.session_state or st.session_state.get('last_analyzed') != text_input):
    if text_input:
        st.session_state['last_analyzed'] = text_input
        
        # Get system resource info before prediction
        pre_prediction_resources = update_system_resources()
        
        # Make prediction
        prediction = predict_toxicity(text_input, selected_language)
        
        # Update resource usage after prediction
        post_prediction_resources = update_system_resources()
        
        # Calculate resource usage delta
        resource_delta = {
            "cpu_usage": float(post_prediction_resources["cpu"]["usage"].rstrip("%")) - float(pre_prediction_resources["cpu"]["usage"].rstrip("%")),
            "ram_usage": float(post_prediction_resources["ram"]["percent"].rstrip("%")) - float(pre_prediction_resources["ram"]["percent"].rstrip("%"))
        }
        
        # Update GPU memory info after prediction
        if DEVICE == "cuda":
            new_memory_info = update_gpu_info()
            # Note: Ideally we would update the displayed memory usage here,
            # but Streamlit doesn't support dynamic updates without a rerun,
            # so we'll just include memory info in our metrics
        
        # Set analysis status flags but remove celebration effect code
        st.session_state['is_analysis_complete'] = True
        st.session_state['analysis_has_error'] = "error" in prediction and prediction["error"]
        
        if "error" in prediction and prediction["error"]:
            st.error(prediction["error"])
        elif prediction["results"]:
            # Remove celebration effect call
            # celebration_effect()
            
            results = prediction["results"]
            performance = prediction.get("performance", {})
            
            # Overall toxicity result
            is_toxic = results["is_toxic"]
            result_color = THEME["toxic"] if is_toxic else THEME["non_toxic"]
            result_text = "TOXIC" if is_toxic else "NON-TOXIC"
            
            # Language info
            lang_code = prediction["lang_code"]
            lang_info = SUPPORTED_LANGUAGES.get(lang_code, {"name": lang_code, "flag": "🌐"})
            
            # Count toxic categories
            toxic_count = len(results["toxic_categories"]) if is_toxic else 0
            
            # Create data for visualization but don't display the table
            categories = []
            probabilities = []
            statuses = []
            
            # Use the same thresholds that are used in the inference model
            category_thresholds = {
                'toxic': 0.60,
                'severe_toxic': 0.54,
                'obscene': 0.60,
                'threat': 0.48,
                'insult': 0.60,
                'identity_hate': 0.50
            }
            
            for label, prob in results["probabilities"].items():
                categories.append(label.replace('_', ' ').title())
                probabilities.append(round(prob * 100, 1))
                threshold = category_thresholds.get(label, 0.5) * 100
                statuses.append("DETECTED" if prob * 100 >= threshold else "Not Detected")
            
            # Sort by probability for the chart
            chart_data = sorted(zip(categories, probabilities, statuses), key=lambda x: x[1], reverse=True)
            chart_cats, chart_probs, chart_statuses = zip(*chart_data)
            
            # Two column layout for results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Card with overall result and detected categories
                with stylable_container(
                    key="result_card",
                    css_styles=f"""
                        {{
                            border-radius: 10px;
                            padding: 10px 15px;
                            background-color: {THEME["card_bg"]};
                            border-left: 5px solid {result_color};
                            margin-bottom: 10px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                        }}
                    """
                ):
                    # Overall result with abbreviated display
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <h3 style="margin: 0; margin-right: 10px;">Analysis Result:</h3>
                        <span style='background-color: {hex_to_rgba(result_color, 0.13)}; color: {result_color}; font-family: "Space Grotesk", sans-serif; font-size: 1.1rem; font-weight: 700; padding: 2px 10px; border-radius: 6px;'>{result_text}</span>
                    </div>
                    <div style="margin: 5px 0; font-size: 0.95rem;">
                        <b>Language:</b> {lang_info['flag']} {lang_info['name']} {'(detected)' if prediction["detected"] else ''}
                    </div>
                    <div style="margin: 5px 0 12px 0; font-size: 0.95rem;">
                        <b>Toxic Categories:</b> {", ".join([f'<span class="toxic-category" style="padding: 2px 6px; font-size: 0.8rem; display: inline-block;">{category.replace("_", " ").title()}</span>' for category in results["toxic_categories"]]) if is_toxic and toxic_count > 0 else '<span style="color: #666; font-size: 0.9rem;">None</span>'}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add toxicity probability graph inside the result card
                    st.markdown("<h4 style='margin-top: 12px; margin-bottom: 8px;'>Toxicity Probabilities:</h4>", unsafe_allow_html=True)
                    
                    # Create a horizontal bar chart with Plotly
                    fig = go.Figure()
                    
                    # Add bars with different colors based on toxicity
                    for i, (cat, prob, status) in enumerate(zip(chart_cats, chart_probs, chart_statuses)):
                        color = THEME["toxic"] if status == "DETECTED" else THEME["non_toxic"]
                        border_color = hex_to_rgba(color, 0.85)  # Using rgba for border
                        
                        fig.add_trace(go.Bar(
                            y=[cat],
                            x=[prob],
                            orientation='h',
                            name=cat,
                            marker=dict(
                                color=color,
                                line=dict(
                                    color=border_color,
                                    width=2
                                )
                            ),
                            text=[f"{prob}%"],
                            textposition='outside',
                            textfont=dict(size=16, weight='bold'),  # Much larger, bold text
                            hoverinfo='text',
                            hovertext=[f"{cat}: {prob}%"]
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=None,
                        xaxis_title="Probability (%)",
                        yaxis_title=None,  # Remove y-axis title to save space
                        height=340,  # Significantly increased height
                        margin=dict(l=10, r=40, t=20, b=40),  # More margin space for labels
                        xaxis=dict(
                            range=[0, 115],  # Extended for outside labels
                            gridcolor=hex_to_rgba(THEME["text"], 0.15),
                            zerolinecolor=hex_to_rgba(THEME["text"], 0.3),
                            color=THEME["text"],
                            tickfont=dict(size=15),  # Larger tick font
                            title_font=dict(size=16, family="Space Grotesk, sans-serif")  # Larger axis title
                        ),
                        yaxis=dict(
                            gridcolor=hex_to_rgba(THEME["text"], 0.15),
                            color=THEME["text"],
                            tickfont=dict(size=15, family="Space Grotesk, sans-serif", weight='bold'),  # Larger, bold category names
                            automargin=True  # Auto-adjust margin to fit category names
                        ),
                        bargap=0.3,  # More space between bars
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(
                            family="Space Grotesk, sans-serif",
                            color=THEME["text"],
                            size=15  # Larger base font size
                        ),
                        showlegend=False
                    )
                    
                    # Grid lines
                    fig.update_xaxes(
                        showgrid=True, 
                        gridwidth=1.5,  # Slightly wider grid lines
                        gridcolor=hex_to_rgba(THEME["text"], 0.15),
                        dtick=20
                    )
                    
                    # Display the plot
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': False,
                        'displaylogo': False
                    })
            
            with col2:
                # Performance metrics card
                if performance:
                    with stylable_container(
                        key="performance_metrics_card",
                        css_styles=f"""
                            {{
                                border-radius: 10px;
                                padding: 20px;
                                background-color: {THEME["card_bg"]};
                                border-left: 3px solid {THEME["primary"]};
                                height: 100%;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                            }}
                        """
                    ):
                        st.markdown("<h3 style='margin-top: 0;'>Performance Metrics</h3>", unsafe_allow_html=True)
                        total_time = performance.get("total_time", 0)
                        inference_time = performance.get("model_inference_time", 0)
                        lang_detection_time = performance.get("lang_detection_time", 0)
                        
                        # Create tabs for different types of metrics
                        perf_tab1, perf_tab2 = st.tabs(["Time Metrics", "Resource Usage"])
                        
                        with perf_tab1:
                            time_cols = st.columns(1)
                            with time_cols[0]:
                                # Use custom HTML metrics instead of st.metric
                                total_time_val = f"{total_time:.3f}s"
                                inference_time_val = f"{inference_time:.3f}s"
                                lang_detection_time_val = f"{lang_detection_time:.3f}s"
                                
                                st.markdown(f"""
                                <div style="background-color: white; border-left: 3px solid {THEME["primary"]}; border: 1px solid {hex_to_rgba(THEME["primary"], 0.2)}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                    <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 0.85rem; margin-bottom: 3px;">
                                        Total Time
                                    </div>
                                    <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.2rem;">
                                        {total_time_val}
                                    </div>
                                </div>
                                
                                <div style="background-color: white; border-left: 3px solid {THEME["primary"]}; border: 1px solid {hex_to_rgba(THEME["primary"], 0.2)}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                    <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 0.85rem; margin-bottom: 3px;">
                                        Model Inference
                                    </div>
                                    <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.2rem;">
                                        {inference_time_val}
                                    </div>
                                </div>
                                
                                <div style="background-color: white; border-left: 3px solid {THEME["primary"]}; border: 1px solid {hex_to_rgba(THEME["primary"], 0.2)}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                    <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 0.85rem; margin-bottom: 3px;">
                                        Language Detection
                                    </div>
                                    <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.2rem;">
                                        {lang_detection_time_val}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with perf_tab2:
                            # Display system resource metrics with custom HTML
                            current_sys_info = update_system_resources()
                            
                            # Format delta: add + sign for positive values
                            cpu_usage = current_sys_info["cpu"]["usage"]
                            cpu_delta = f"{resource_delta['cpu_usage']:+.1f}%" if abs(resource_delta['cpu_usage']) > 0.1 else None
                            cpu_delta_display = f" ({cpu_delta})" if cpu_delta else ""
                            
                            ram_usage = current_sys_info["ram"]["percent"]
                            ram_delta = f"{resource_delta['ram_usage']:+.1f}%" if abs(resource_delta['ram_usage']) > 0.1 else None
                            ram_delta_display = f" ({ram_delta})" if ram_delta else ""
                            
                            if DEVICE == "cuda":
                                gpu_memory = update_gpu_info()
                                memory_display = f"GPU Memory: {gpu_memory}"
                            else:
                                memory_display = f"System RAM: {current_sys_info['ram']['used']} / {current_sys_info['ram']['total']}"
                            
                            st.markdown(f"""
                            <div style="background-color: white; border-left: 3px solid {THEME["primary"]}; border: 1px solid {hex_to_rgba(THEME["primary"], 0.2)}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 0.85rem; margin-bottom: 3px;">
                                    CPU Usage
                                </div>
                                <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.2rem;">
                                    {cpu_usage}<span style="font-size: 0.9rem; color: {THEME["primary"]};">{cpu_delta_display}</span>
                                </div>
                            </div>
                            
                            <div style="background-color: white; border-left: 3px solid {THEME["primary"]}; border: 1px solid {hex_to_rgba(THEME["primary"], 0.2)}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 0.85rem; margin-bottom: 3px;">
                                    RAM Usage
                                </div>
                                <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.2rem;">
                                    {ram_usage}<span style="font-size: 0.9rem; color: {THEME["primary"]};">{ram_delta_display}</span>
                                </div>
                            </div>
                            
                            <div style="background-color: white; border-left: 3px solid {THEME["primary"]}; border: 1px solid {hex_to_rgba(THEME["primary"], 0.2)}; border-radius: 10px; padding: 10px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                                <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 0.85rem; margin-bottom: 3px;">
                                    Memory
                                </div>
                                <div style="color: {THEME["text"]}; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.2rem;">
                                    {memory_display}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        pass  # Remove the info message

# Bottom section with improved styling for usage guide
st.divider()
colored_header(
    label="How to use this AI Model",
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
    <div>Select a specific language from the sidebar or use the auto-detect feature if you're unsure.</div>
</div>

<div class='usage-step'>
    <div class='step-number'>3</div>
    <div>Click "Analyze Text" to get detailed toxicity analysis results.</div>
</div>

<div class='usage-step'>
    <div class='step-number'>4</div>
    <div>Examine the breakdown of toxicity categories, probabilities, and visualization.</div>
</div>

<div class='usage-step'>
    <div class='step-number'>5</div>
    <div>Try different examples from the sidebar to see how the model performs with various languages.</div>
</div>
""", unsafe_allow_html=True)

# Adding footer with credits and improved styling
st.markdown("""
<div class='footer'>
    <div>Powered by XLM-RoBERTa | Streamlit UI</div>
    <div style='font-size: 0.9rem; margin-top: 5px;'>Made with ❤️ by Deeptanshu, Nauman, Sara and Soham</div>
</div>
""", unsafe_allow_html=True) 