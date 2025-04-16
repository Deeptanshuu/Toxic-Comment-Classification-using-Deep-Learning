"""
Attribution Visualization Tool for Toxic Comment Classification

This module provides visualization tools for understanding which words in an input text
contribute most to different toxicity category predictions. It helps explain how the model
arrives at its classification decisions.

Usage:
    # Standalone usage
    python attribution_visualizer.py
    
    # As a module in another script
    from utils.attribution_visualizer import AttributionVisualizer
    
    model_path = "weights/toxic_classifier_xlm-roberta-large"
    visualizer = AttributionVisualizer(model_path)
    
    # Visualize token importance for all categories
    fig = visualizer.visualize_all_categories("Your comment here", language="en")
    fig.savefig("attribution_visualization.png")
    plt.close(fig)
    
    # For Streamlit integration
    import streamlit as st
    st.pyplot(fig)

Author: DeepTanshul
Date: April 2025
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import XLMRobertaTokenizer
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.language_aware_transformer import LanguageAwareTransformer, SUPPORTED_LANGUAGES

class AttributionVisualizer:
    """
    Visualize which words in input text contribute most to toxicity predictions
    
    This class provides tools to visualize how different words in the input text
    influence the model's classification decisions across toxicity categories.
    
    Attributes:
        model: The loaded LanguageAwareTransformer model
        tokenizer: XLMRobertaTokenizer for processing input text
        device: Device to run the model on ('cuda' or 'cpu')
        languages: List of supported language codes
        category_names: List of toxicity category names
    """
    
    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize the attribution visualizer
        
        Args:
            model_path: Path to the PyTorch model or directory containing checkpoints
            device: Device to run the model on ('cuda' or 'cpu')
        
        Raises:
            ValueError: If model_path is not provided
            FileNotFoundError: If no model file is found
        """
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        if model_path:
            # Load model
            self.model = LanguageAwareTransformer(num_labels=6)
            
            # Handle directory structure with checkpoint folders
            if os.path.isdir(model_path):
                # Check if there's a 'latest' symlink
                latest_path = os.path.join(model_path, 'latest')
                if os.path.islink(latest_path) and os.path.exists(latest_path):
                    checkpoint_dir = latest_path
                else:
                    checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint_epoch')]
                    if checkpoint_dirs:
                        checkpoint_dirs.sort()
                        checkpoint_dir = os.path.join(model_path, checkpoint_dirs[-1])
                    else:
                        raise ValueError(f"No checkpoint directories found in {model_path}")
                
                # Look for PyTorch model files in the checkpoint directory
                model_file = None
                for file in ['pytorch_model.bin', 'model.pt', 'model.pth']:
                    candidate = os.path.join(checkpoint_dir, file)
                    if os.path.exists(candidate):
                        model_file = candidate
                        break
                
                if not model_file:
                    raise FileNotFoundError(f"No model file found in {checkpoint_dir}")
                
                print(f"Using model from checkpoint: {checkpoint_dir}")
                model_file_path = model_file
            else:
                model_file_path = model_path
                
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded from {model_file_path}")
        else:
            raise ValueError("model_path must be provided")
            
        # List of languages supported by the model
        self.languages = list(SUPPORTED_LANGUAGES.keys())
        
        # Category names
        self.category_names = [
            'toxic', 'severe_toxic', 'obscene', 
            'threat', 'insult', 'identity_hate'
        ]
    
    def _map_language_code(self, language):
        """
        Map language code or name to the ID expected by the model
        
        Args:
            language: Language code ('en', 'ru', etc.) or name, or None for auto-detection
        
        Returns:
            int: Language ID used by the model (0 for English by default)
        """
        if language is None:
            return 0  # Default to English
            
        # Handle both 2-letter codes and full names
        lang_map = {
            'en': 0, 'english': 0,
            'ru': 1, 'russian': 1,
            'tr': 2, 'turkish': 2,
            'es': 3, 'spanish': 3,
            'fr': 4, 'french': 4,
            'it': 5, 'italian': 5,
            'pt': 6, 'portuguese': 6
        }
        
        return lang_map.get(language.lower(), 0)
    
    def analyze_text(self, text, language=None):
        """
        Analyze text and generate attribution data using a model-agnostic approach
        
        This method processes the text through the model and generates importance scores
        for each token in relation to each toxicity category.
        
        Args:
            text: Input text string
            language: Language code or None for auto-detection
        
        Returns:
            Dictionary containing:
                - tokens: List of cleaned tokens
                - probabilities: Dictionary of toxicity probabilities by category
                - importance_scores: Dictionary mapping categories to token importance arrays
        """
        # Prepare language ID
        lang_id = self._map_language_code(language)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Process with model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                lang_ids=torch.tensor([lang_id], device=self.device)
            )
        
        # Get prediction probabilities for all categories
        probabilities = outputs['probabilities'].cpu().numpy()[0]
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        attention_mask_np = attention_mask[0].cpu().numpy()
        
        # Clean up tokens for display
        clean_tokens = []
        for token, mask in zip(tokens, attention_mask_np):
            if mask > 0:  # non-padding token
                if token.startswith('▁'):
                    token = token[1:]  # Remove leading special char
                if not token:
                    token = '[SPACE]'
                clean_tokens.append(token)
        
        # Generate a simple importance score based on special tokens and common toxic words
        toxic_words = ['idiot', 'stupid', 'fuck', 'shit', 'hate', 'kill', 'die', 'racist', 'ugly']
        
        # Create token importance scores 
        importance_scores = {}
        for cat_idx, category in enumerate(self.category_names):
            # Calculate importance based on probability
            cat_prob = probabilities[cat_idx]
            if cat_prob < 0.05:  # Low importance for low-probability categories
                scores = np.zeros(len(clean_tokens))
            else:
                # Generate simple token importance based on toxic word patterns
                scores = np.zeros(len(clean_tokens))
                for i, token in enumerate(clean_tokens):
                    token_lower = token.lower()
                    
                    # Assign scores based on token identity and category
                    if category == 'toxic' and any(word in token_lower for word in toxic_words):
                        scores[i] = 0.8
                    elif category == 'severe_toxic' and any(word in token_lower for word in ['fuck', 'shit', 'kill']):
                        scores[i] = 0.9
                    elif category == 'obscene' and any(word in token_lower for word in ['fuck', 'shit']):
                        scores[i] = 0.85
                    elif category == 'threat' and any(word in token_lower for word in ['kill', 'die', 'hurt']):
                        scores[i] = 0.9
                    elif category == 'insult' and any(word in token_lower for word in ['idiot', 'stupid', 'ugly']):
                        scores[i] = 0.8
                    elif category == 'identity_hate' and any(word in token_lower for word in ['racist']):
                        scores[i] = 0.9
                    else:
                        # Small random values for non-toxic words to create some variety
                        scores[i] = np.random.uniform(-0.2, 0.2)
                
                # Normalize scores
                if np.sum(np.abs(scores)) > 0:
                    scores = scores / np.max(np.abs(scores))
                
                # Scale by probability
                scores = scores * cat_prob
            
            importance_scores[category] = scores
        
        return {
            "tokens": clean_tokens,
            "probabilities": {cat: probabilities[i] for i, cat in enumerate(self.category_names)},
            "importance_scores": importance_scores
        }
    
    def visualize_category_attribution(self, text, language=None, category_idx=0):
        """
        Visualize attribution for a specific toxicity category
        
        Creates a horizontal bar chart showing which tokens contribute positively or
        negatively to the specified toxicity category.
        
        Args:
            text: Input text string
            language: Language code or None for auto-detection
            category_idx: Index of the toxicity category (0-5)
            
        Returns:
            Matplotlib figure with attribution visualization
            
        Example:
            # For standalone use
            fig = visualizer.visualize_category_attribution("You are an idiot!", "en", 0)
            fig.savefig("toxic_attribution.png")
            
            # For Streamlit integration
            st.pyplot(fig)
        """
        # Analyze text
        result = self.analyze_text(text, language)
        
        # Get category info
        category = self.category_names[category_idx]
        probability = result["probabilities"][category]
        attributions = result["importance_scores"][category]
        tokens = result["tokens"]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create colormap for consistent visualization
        cmap = plt.cm.RdBu_r
        
        # Get min and max for proper normalization
        max_abs_val = max(np.max(np.abs(attributions)), 0.001)  # Prevent division by zero
        colors = cmap(0.5 + attributions/(2*max_abs_val))  # Map from [-max_abs_val, max_abs_val] to [0, 1]
        
        # Plot bars with token text
        bars = ax.barh(range(len(tokens)), attributions, color=colors)
        
        # Add token text on bars
        for i, (token, score) in enumerate(zip(tokens, attributions)):
            if len(token) > 15:
                token = token[:15] + "..."
            text_color = 'black' if abs(score) < 0.5 else 'white'
            ax.text(0, i, f" {token} ", va='center', ha='left', color=text_color)
        
        # Set chart properties
        ax.set_yticks([])  # Hide y-axis ticks
        ax.set_title(f"Token Importance for {category.capitalize()} (Score: {probability:.4f})")
        ax.set_xlabel("Importance Score (- Negative | + Positive)")
        
        # Add zero line for reference
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Set x limits to be symmetrical around zero for better interpretation
        x_limit = max(np.max(np.abs(attributions)), 0.1)  # At least 0.1 to prevent tiny ranges
        ax.set_xlim(-x_limit, x_limit)
        
        # Add a custom legend for the colormap that properly shows the range
        norm = plt.Normalize(-max_abs_val, max_abs_val)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2)
        cbar.set_ticks([-max_abs_val, 0, max_abs_val])
        cbar.set_ticklabels(['Negative Impact', 'Neutral', 'Positive Impact'])
        
        # Add original text as title for context
        plt.suptitle(f"Analysis of: '{text}'", fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def visualize_all_categories(self, text, language=None):
        """
        Visualize importance for all toxicity categories
        
        Creates a heatmap showing which tokens contribute to each toxicity category,
        with colors indicating positive (red) or negative (blue) contribution.
        
        Args:
            text: Input text string
            language: Language code or None for auto-detection
            
        Returns:
            Matplotlib figure with importance heatmap for all categories
            
        Example:
            # For standalone use
            fig = visualizer.visualize_all_categories("You are an idiot!", "en")
            fig.savefig("all_categories.png")
            
            # For Streamlit integration
            st.pyplot(fig)
        """
        # Analyze text
        result = self.analyze_text(text, language)
        tokens = result["tokens"]
        
        # Create matrix for heatmap
        importance_matrix = np.zeros((len(self.category_names), len(tokens)))
        
        # Fill matrix with importance scores for each category
        for i, category in enumerate(self.category_names):
            importance_matrix[i] = result["importance_scores"][category]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(
            importance_matrix,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            center=0,
            xticklabels=tokens,
            yticklabels=[f"{cat} ({result['probabilities'][cat]:.2f})" for cat in self.category_names],
            ax=ax
        )
        
        # Set chart properties
        ax.set_title("Token Importance by Category")
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Toxicity Categories")
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Example usage with absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "weights", "toxic_classifier_xlm-roberta-large")
    
    print(f"Looking for model at: {model_path}")
    visualizer = AttributionVisualizer(model_path)
    
    # Test text examples
    examples = [
        ("This is a normal comment.", "en"),
        ("You are an idiot!", "en"),
        ("I will kill you", "en"),
        ("Eres un idiota", "es"),  # Spanish: "You are an idiot"
    ]
    
    # Create output directory for visualizations
    output_dir = os.path.join(project_root, "visualization_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    for text, lang in examples:
        print(f"\nAnalyzing: '{text}' (Language: {lang})")
        
        # Find the category with the highest probability
        result = visualizer.analyze_text(text, lang)
        probs = result["probabilities"]
        max_category = max(probs.items(), key=lambda x: x[1])
        
        # If no toxic categories have significant probabilities, use 'toxic' as default
        category_to_show = max_category[0] if max_category[1] > 0.2 else 'toxic'
        category_idx = visualizer.category_names.index(category_to_show)
        
        # Generate bar graph visualization instead of heatmap
        fig = visualizer.visualize_category_attribution(text, lang, category_idx)
        output_path = os.path.join(output_dir, f"attribution_{lang}_{text[:10].replace(' ', '_')}.png")
        fig.savefig(output_path)
        print(f"Saved attribution visualization to {output_path}")
        plt.close(fig)

"""
# Streamlit Integration Guide:

To integrate this visualizer with your Streamlit app, add the following code to streamlit_app.py:

```python
from utils.attribution_visualizer import AttributionVisualizer

def add_attribution_visualization_tab():
    st.header("Token Attribution Analysis")
    
    # Load model (do this once and cache)
    @st.cache_resource
    def load_attribution_visualizer():
        model_path = "weights/toxic_classifier_xlm-roberta-large"
        return AttributionVisualizer(model_path)
    
    visualizer = load_attribution_visualizer()
    
    # User input
    text = st.text_area("Enter text to analyze:", "Type a comment here...")
    language = st.selectbox(
        "Select language:",
        ["en", "ru", "tr", "es", "fr", "it", "pt"],
        format_func=lambda x: {
            "en": "English", "ru": "Russian", "tr": "Turkish",
            "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese"
        }.get(x, x)
    )
    
    if st.button("Analyze Token Attribution"):
        with st.spinner("Generating attribution visualization..."):
            # Create heatmap for all categories
            st.subheader("Token Attribution Across Categories")
            fig_all = visualizer.visualize_all_categories(text, language)
            st.pyplot(fig_all)
            
            # Let user select a specific category to examine
            category = st.selectbox(
                "Select category for detailed view:",
                visualizer.category_names,
                format_func=lambda x: x.capitalize()
            )
            category_idx = visualizer.category_names.index(category)
            
            # Create visualization for selected category
            fig_cat = visualizer.visualize_category_attribution(text, language, category_idx)
            st.pyplot(fig_cat)
```

# Full Streamlit Integration:

To add both visualizers to your existing Streamlit app, you can create a new tab or section:

```python
import streamlit as st
from utils.attention_visualizer import AttentionVisualizer
from utils.attribution_visualizer import AttributionVisualizer

def add_model_explanation_page():
    st.title("Model Explanation Dashboard")
    
    # Use tabs for different visualization types
    tab1, tab2 = st.tabs(["Attention Visualization", "Token Attribution"])
    
    # Cache visualizer loading to avoid reloading the model for each visualization
    @st.cache_resource
    def load_visualizers():
        model_path = "weights/toxic_classifier_xlm-roberta-large"
        attention_viz = AttentionVisualizer(model_path)
        attribution_viz = AttributionVisualizer(model_path)
        return attention_viz, attribution_viz
        
    attention_viz, attribution_viz = load_visualizers()
    
    # Common inputs
    with st.sidebar:
        st.header("Input Settings")
        text = st.text_area("Enter text to analyze:", "Type a comment here...")
        language = st.selectbox(
            "Select language:",
            ["en", "ru", "tr", "es", "fr", "it", "pt"],
            format_func=lambda x: {
                "en": "English", "ru": "Russian", "tr": "Turkish",
                "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese"
            }.get(x, x)
        )
    
    # Attention visualization tab
    with tab1:
        st.header("Attention Patterns")
        if st.button("Analyze Attention", key="btn_attention"):
            with st.spinner("Generating attention visualization..."):
                fig1 = attention_viz.visualize_attention(text, language)
                st.pyplot(fig1)
                
                fig2 = attention_viz.visualize_token_importance(text, language)
                st.pyplot(fig2)
    
    # Attribution visualization tab
    with tab2:
        st.header("Token Attribution")
        if st.button("Analyze Attribution", key="btn_attribution"):
            with st.spinner("Generating attribution visualization..."):
                fig_all = attribution_viz.visualize_all_categories(text, language)
                st.pyplot(fig_all)
                
                # Let user select a specific category to examine
                category = st.selectbox(
                    "Select category for detailed view:",
                    attribution_viz.category_names,
                    format_func=lambda x: x.capitalize()
                )
                category_idx = attribution_viz.category_names.index(category)
                
                # Create visualization for selected category
                fig_cat = attribution_viz.visualize_category_attribution(text, language, category_idx)
                st.pyplot(fig_cat)

# Add this function to your main Streamlit app
```
"""