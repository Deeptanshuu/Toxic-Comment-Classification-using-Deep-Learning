import gradio as gr
import torch
import numpy as np
import os
import json
from model.inference_optimized import OptimizedToxicityClassifier
import matplotlib.pyplot as plt
from typing import List, Dict
import langid
import pandas as pd

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

# Initialize classifier
try:
    if os.path.exists(ONNX_MODEL_PATH):
        classifier = OptimizedToxicityClassifier(onnx_path=ONNX_MODEL_PATH, device=DEVICE)
        print(f"Loaded ONNX model from {ONNX_MODEL_PATH}")
    else:
        classifier = OptimizedToxicityClassifier(pytorch_path=PYTORCH_MODEL_PATH, device=DEVICE)
        print(f"Loaded PyTorch model from {PYTORCH_MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    classifier = None

def detect_language(text: str) -> str:
    """Detect language of input text"""
    try:
        lang, _ = langid.classify(text)
        return lang if lang in SUPPORTED_LANGUAGES else 'en'
    except:
        return 'en'

def predict_toxicity(text: str, selected_language: str = None) -> Dict:
    """Predict toxicity of input text"""
    if not text or not text.strip():
        return {
            "error": "Please enter some text to analyze.",
            "html_result": "<div class='error'>Please enter some text to analyze.</div>"
        }
        
    if classifier is None:
        return {
            "error": "Model not loaded. Please check logs.",
            "html_result": "<div class='error'>Model not loaded. Please check logs.</div>"
        }
    
    # Detect language if not specified
    if not selected_language or selected_language == "Auto-detect":
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
        
        # Format probabilities for display
        probs = results["probabilities"]
        sorted_categories = sorted(
            [(label, probs[label]) for label in probs], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [label.replace('_', ' ').title() for label, _ in sorted_categories]
        values = [prob * 100 for _, prob in sorted_categories]
        colors = ['#ff6b6b' if val >= 50 else '#74c0fc' for val in values]
        
        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Toxicity Analysis')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Annotate values
        for i, v in enumerate(values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
            
        # Create HTML result
        lang_display = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
        overall_result = "TOXIC" if results["is_toxic"] else "NON-TOXIC"
        result_color = "#ff6b6b" if results["is_toxic"] else "#66d9e8"
        
        html_result = f"""
        <div style='margin-bottom: 20px;'>
            <h2>Analysis Result: <span style='color: {result_color};'>{overall_result}</span></h2>
            <h3>Language: {lang_display} {'(detected)' if detected else ''}</h3>
        </div>
        <div style='margin-bottom: 10px;'>
            <table width='100%' style='border-collapse: collapse;'>
                <tr style='background-color: #e9ecef; font-weight: bold;'>
                    <th style='padding: 8px; text-align: left; border: 1px solid #dee2e6;'>Category</th>
                    <th style='padding: 8px; text-align: right; border: 1px solid #dee2e6;'>Probability</th>
                    <th style='padding: 8px; text-align: center; border: 1px solid #dee2e6;'>Status</th>
                </tr>
        """
        
        # Add rows for each toxicity category
        for label, prob in sorted_categories:
            formatted_label = label.replace('_', ' ').title()
            status = "DETECTED" if prob >= 0.5 else "Not Detected"
            status_color = "#ff6b6b" if prob >= 0.5 else "#66d9e8"
            prob_percent = f"{prob * 100:.1f}%"
            
            html_result += f"""
                <tr>
                    <td style='padding: 8px; border: 1px solid #dee2e6;'>{formatted_label}</td>
                    <td style='padding: 8px; text-align: right; border: 1px solid #dee2e6;'>{prob_percent}</td>
                    <td style='padding: 8px; text-align: center; border: 1px solid #dee2e6; color: {status_color}; font-weight: bold;'>{status}</td>
                </tr>
            """
        
        html_result += "</table></div>"
        
        # Add detected categories if toxic
        if results["is_toxic"]:
            toxic_categories = [cat.replace('_', ' ').title() for cat in results["toxic_categories"]]
            categories_list = ", ".join(toxic_categories)
            html_result += f"""
            <div style='margin-top: 10px;'>
                <p><strong>Detected toxic categories:</strong> {categories_list}</p>
            </div>
            """
        
        return {
            "prediction": results,
            "html_result": html_result,
            "fig": fig
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Error processing text: {str(e)}",
            "html_result": f"<div class='error'>Error processing text: {str(e)}</div>"
        }

def create_app():
    """Create and configure the Gradio interface"""
    # Create language dropdown options
    language_options = ["Auto-detect"] + list(SUPPORTED_LANGUAGES.values())
    
    # Define the interface
    with gr.Blocks(css="""
        .error { color: #ff6b6b; font-weight: bold; padding: 10px; border: 1px solid #ff6b6b; }
        .container { margin: 0 auto; max-width: 900px; }
        .gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .example-text { font-style: italic; color: #666; }
    """) as app:
        gr.Markdown("""
        # Multilingual Toxic Comment Classifier
        This app analyzes text for different types of toxicity across multiple languages. 
        Enter your text, select a language (or let it auto-detect), and click 'Analyze'.
        
        Supported languages: English, Russian, Turkish, Spanish, French, Italian, Portuguese
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Type or paste text here...",
                    lines=5
                )
                lang_dropdown = gr.Dropdown(
                    choices=language_options,
                    value="Auto-detect",
                    label="Language"
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### Example texts:")
                with gr.Accordion("English example"):
                    en_example_btn = gr.Button("Use English example")
                with gr.Accordion("Spanish example"):
                    es_example_btn = gr.Button("Use Spanish example")
                with gr.Accordion("French example"):
                    fr_example_btn = gr.Button("Use French example")
        
        # Examples
        en_example_text = "You are such an idiot, nobody likes your stupid content."
        es_example_text = "Eres un completo idiota y nadie te quiere."
        fr_example_text = "Tu es tellement stupide, personne n'aime ton contenu minable."
        
        en_example_btn.click(
            lambda: en_example_text, 
            outputs=text_input
        )
        es_example_btn.click(
            lambda: es_example_text, 
            outputs=text_input
        )
        fr_example_btn.click(
            lambda: fr_example_text, 
            outputs=text_input
        )
        
        # Output components
        result_html = gr.HTML(label="Analysis Result")
        plot_output = gr.Plot(label="Toxicity Probabilities")
        
        # Set up event handling
        analyze_btn.click(
            predict_toxicity,
            inputs=[text_input, lang_dropdown],
            outputs=[result_html, plot_output]
        )
        
        # Also analyze on pressing Enter in the text box
        text_input.submit(
            predict_toxicity,
            inputs=[text_input, lang_dropdown],
            outputs=[result_html, plot_output]
        )
        
        gr.Markdown("""
        ### About this model
        This model classifies text into six toxicity categories:
        - **Toxic**: General toxicity
        - **Severe Toxic**: Extreme toxicity
        - **Obscene**: Obscene content
        - **Threat**: Threatening content
        - **Insult**: Insulting content
        - **Identity Hate**: Identity-based hate
        
        Built using XLM-RoBERTa with language-aware fine-tuning.
        """)
    
    return app

# Launch the app when script is run directly
if __name__ == "__main__":
    # Create and launch the app
    app = create_app()
    app.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=7860,       # Default Gradio port
        share=True              # Generate public link
    ) 