"""
Attention Visualization Tool for Toxic Comment Classification

This module provides visualization tools for understanding how the toxicity classification model uses attention
mechanisms to make predictions. It visualizes which parts of the input text the model focuses on.

Usage:
    # Standalone usage
    python attention_visualizer.py
    
    # As a module in another script
    from utils.attention_visualizer import AttentionVisualizer
    
    model_path = "weights/toxic_classifier_xlm-roberta-large"
    visualizer = AttentionVisualizer(model_path)
    
    # Visualize a single comment
    fig = visualizer.visualize_attention("Your comment here", language="en")
    fig.savefig("attention_visualization.png")
    plt.close(fig)
    
    # For Streamlit integration
    import streamlit as st
    st.pyplot(fig)

Author: DeepTanshul
Date: April 2025
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import XLMRobertaTokenizer
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.language_aware_transformer import LanguageAwareTransformer, SUPPORTED_LANGUAGES

class AttentionVisualizer:
    """
    Visualize attention weights for the Toxicity Classifier model
    
    This class provides tools to visualize how the model pays attention to different
    parts of the input text when making toxicity classifications.
    
    Attributes:
        model: The loaded LanguageAwareTransformer model
        tokenizer: XLMRobertaTokenizer for processing input text
        device: Device to run the model on ('cuda' or 'cpu')
        languages: List of supported language codes
        category_names: List of toxicity category names
    """
    
    def __init__(self, model_path=None, device='cuda'):
        """
        Initialize the attention visualizer
        
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
                    # If no 'latest' symlink, look for checkpoint dirs
                    checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint_epoch')]
                    if checkpoint_dirs:
                        checkpoint_dirs.sort()  # Sort to get the latest by name
                        checkpoint_dir = os.path.join(model_path, checkpoint_dirs[-1])
                    else:
                        raise ValueError(f"No checkpoint directories found in {model_path}")
                
                # Look for PyTorch model files in the checkpoint directory
                model_file = None
                potential_files = ['pytorch_model.bin', 'model.pt', 'model.pth']
                for file in potential_files:
                    candidate = os.path.join(checkpoint_dir, file)
                    if os.path.exists(candidate):
                        model_file = candidate
                        break
                
                if not model_file:
                    raise FileNotFoundError(f"No model file found in {checkpoint_dir}")
                
                print(f"Using model from checkpoint: {checkpoint_dir}")
                model_file_path = model_file
            else:
                # If model_path is a direct file path
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
    
    def get_model_output(self, text, language=None, get_attention=True):
        """
        Get model output including attention weights
        
        Args:
            text: Input text string
            language: Language code ('en', 'ru', etc.) or None for auto-detection
            get_attention: Whether to collect attention weights
        
        Returns:
            Dictionary with model outputs including:
                - tokens: List of tokens from the tokenizer
                - token_ids: Tensor of token IDs
                - attention_mask: Mask for non-padding tokens
                - probabilities: Toxicity probabilities for each category
                - logits: Raw model outputs before sigmoid
                - attention_weights: Attention weights tensor or None
        """
        # Prepare language ID
        lang_id = self._map_language_code(language)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run model with hooks to capture attention weights
        attention_weights = []
        
        def get_attention_hook(module, input, output):
            # Capture attention weights from the forward pass
            # For q_proj, just store the input tensors for now
            attention_weights.append(input[0].detach())
            
        # Register forward hooks to capture attention
        hooks = []
        if get_attention:
            hooks.append(self.model.q_proj.register_forward_hook(get_attention_hook))
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                lang_ids=torch.tensor([lang_id], device=self.device)
            )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get token information
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        token_ids = inputs["input_ids"][0].cpu().numpy()
        attention_mask = inputs["attention_mask"][0].cpu().numpy()
        
        # Calculate an approximation of attention from the captured weights
        # This is an approximation since we're not capturing the full attention mechanism
        if get_attention and attention_weights:
            # Take the captured hidden states and calculate a simplified attention matrix
            hidden_states = attention_weights[0]
            seq_len = hidden_states.size(1)
            
            # Calculate dot-product attention as a simple approximation
            attention_approx = torch.matmul(
                hidden_states[0], hidden_states[0].transpose(-1, -2)
            )
            
            # Normalize the attention scores
            attention_approx = torch.nn.functional.softmax(attention_approx / (hidden_states.size(-1) ** 0.5), dim=-1)
        else:
            attention_approx = None
        
        # Format results
        result = {
            "tokens": tokens,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "probabilities": outputs['probabilities'].cpu().numpy()[0],
            "logits": outputs['logits'].cpu().numpy()[0],
            "attention_weights": attention_approx
        }
        
        return result

    def clean_tokens(self, tokens, attention_mask):
        """
        Clean tokens by removing special tokens, padding tokens, and formatting
        
        Args:
            tokens: List of tokens from the tokenizer
            attention_mask: Attention mask indicating valid tokens
            
        Returns:
            List of cleaned tokens
        """
        clean_tokens = []
        special_tokens = ['<s>', '</s>', '<pad>', '<mask>']
        
        for token, mask_val in zip(tokens, attention_mask):
            # Skip padding tokens
            if mask_val <= 0:
                continue
                
            # Skip special tokens
            if token in special_tokens:
                continue
                
            # Clean up token for display
            if token.startswith('▁'):
                token = token[1:]  # Remove leading special char
            if not token:
                token = '[SPACE]'
                
            clean_tokens.append(token)
            
        return clean_tokens
        
    def visualize_attention(self, text, language=None, layer_idx=0):
        """
        Visualize attention weights for a specific layer
        
        This function creates a visualization with:
        1. Attention heatmap showing how tokens relate to each other
        2. Category scores panel showing toxicity probabilities
        
        Args:
            text: Input text string
            language: Language code or None for auto-detection
            layer_idx: Index of the layer to visualize (only 0 for our custom model)
            
        Returns:
            Matplotlib figure with attention heatmap visualization
            
        Example:
            # For standalone use
            fig = visualizer.visualize_attention("You are an idiot!", "en")
            fig.savefig("attention.png")
            
            # For Streamlit integration
            st.pyplot(fig)
        """
        result = self.get_model_output(text, language)
        
        # Process tokens - remove special tokens
        tokens = self.clean_tokens(result["tokens"], result["attention_mask"])
        
        # Prepare attention matrix for visualization
        if result["attention_weights"] is not None:
            # Get attention for non-padding and non-special tokens only
            attention_mask = result["attention_mask"]
            
            # Filter out special tokens
            special_token_indices = []
            for i, token in enumerate(result["tokens"]):
                if token in ['<s>', '</s>', '<pad>', '<mask>']:
                    special_token_indices.append(i)
            
            # Get valid indices
            valid_indices = [i for i, mask_val in enumerate(attention_mask) 
                           if mask_val > 0 and i not in special_token_indices]
            valid_len = len(valid_indices)
            
            if valid_len == 0:
                print("No valid tokens found after filtering special tokens")
                return None
            
            # Extract the attention for valid tokens only
            attn_weights = result["attention_weights"]
            filtered_attn = torch.zeros((valid_len, valid_len), device=attn_weights.device)
            
            for i, src_idx in enumerate(valid_indices):
                for j, tgt_idx in enumerate(valid_indices):
                    filtered_attn[i, j] = attn_weights[src_idx, tgt_idx]
            
            # Move to CPU for numpy conversion
            attn_matrix = filtered_attn.cpu().numpy()
            
            # Create visualization with just the heatmap and scores panel
            fig = plt.figure(figsize=(14, 8))
            
            # Create grid for heatmap and score panel
            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])
            
            # Plot attention heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            heatmap = sns.heatmap(
                attn_matrix,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis',
                ax=ax1
            )
            ax1.set_title(f"Attention Matrix")
            ax1.set_xlabel("Token")
            ax1.set_ylabel("Token")
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add category probabilities in a separate panel
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.axis('off')  # Hide axis
            
            # Format probability text
            ax2.text(0.5, 0.98, "Category Scores", fontsize=12, weight='bold', ha='center', va='top')
            
            # Add each category score with color coding
            y_pos = 0.85  # Starting y position
            for i, cat in enumerate(self.category_names):
                prob = result['probabilities'][i]
                
                # Add color coding based on probability
                if prob > 0.75:
                    color = 'darkred'
                    weight = 'bold'
                elif prob > 0.5:
                    color = 'red'
                    weight = 'bold'
                elif prob > 0.25:
                    color = 'darkorange'
                    weight = 'normal'
                else:
                    color = 'black'
                    weight = 'normal'
                
                # Add category name and formatted probability
                ax2.text(0.1, y_pos - i*0.1, f"{cat}:", fontsize=11, va='center')
                ax2.text(0.7, y_pos - i*0.1, f"{prob:.3f}", fontsize=11, va='center', 
                         color=color, weight=weight)
            
            # Add original text as title
            plt.suptitle(f"Analysis of: '{text}'", fontsize=12)
            
            plt.tight_layout()
            return fig
        else:
            print("Attention weights not available")
            return None

# Example usage when run as a script
if __name__ == "__main__":
    # Use absolute path to ensure it works from any directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "weights", "toxic_classifier_xlm-roberta-large")
    
    print(f"Looking for model at: {model_path}")
    visualizer = AttentionVisualizer(model_path)
    
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
        fig = visualizer.visualize_attention(text, lang)
        output_path = os.path.join(output_dir, f"attention_{lang}_{text[:10].replace(' ', '_')}.png")
        fig.savefig(output_path)
        print(f"Saved attention visualization to {output_path}")
        plt.close(fig)