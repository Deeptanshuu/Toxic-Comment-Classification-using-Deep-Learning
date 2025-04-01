import torch
import onnxruntime as ort
from transformers import XLMRobertaTokenizer
import numpy as np
import os

class OptimizedToxicityClassifier:
    """High-performance toxicity classifier for production"""
    
    def __init__(self, onnx_path=None, pytorch_path=None, device='cuda'):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        
        # Language mapping
        self.lang_map = {
            'en': 0, 'ru': 1, 'tr': 2, 'es': 3, 
            'fr': 4, 'it': 5, 'pt': 6
        }
        
        # Label names
        self.label_names = [
            'toxic', 'severe_toxic', 'obscene', 
            'threat', 'insult', 'identity_hate'
        ]
        
        # Load ONNX model if path provided
        if onnx_path and os.path.exists(onnx_path):
            # Use ONNX Runtime for inference
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers() \
                else ['CPUExecutionProvider']
                
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            self.use_onnx = True
            print(f"Loaded ONNX model from {onnx_path}")
        # Fall back to PyTorch if ONNX not available
        elif pytorch_path:
            from model.language_aware_transformer import LanguageAwareTransformer
            
            # Handle directory structure with checkpoint folders and 'latest' symlink
            if os.path.isdir(pytorch_path):
                # Check if there's a 'latest' symlink
                latest_path = os.path.join(pytorch_path, 'latest')
                if os.path.islink(latest_path) and os.path.exists(latest_path):
                    checkpoint_dir = latest_path
                else:
                    # If no 'latest' symlink, look for checkpoint dirs and use the most recent one
                    checkpoint_dirs = [d for d in os.listdir(pytorch_path) if d.startswith('checkpoint_epoch')]
                    if checkpoint_dirs:
                        checkpoint_dirs.sort()  # Sort to get the latest by name
                        checkpoint_dir = os.path.join(pytorch_path, checkpoint_dirs[-1])
                    else:
                        raise ValueError(f"No checkpoint directories found in {pytorch_path}")
                
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
                model_path = model_file
            else:
                # If pytorch_path is a direct file path
                model_path = pytorch_path
                
            self.model = LanguageAwareTransformer(num_labels=6)
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            self.use_onnx = False
            self.device = device
            print(f"Loaded PyTorch model from {model_path}")
        else:
            raise ValueError("Either onnx_path or pytorch_path must be provided")
    
    def predict(self, texts, langs=None, batch_size=8):
        """
        Predict toxicity for a list of texts
        
        Args:
            texts: List of text strings
            langs: List of language codes (e.g., 'en', 'fr')
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with toxicity predictions
        """
        results = []
        
        # Auto-detect or default language if not provided
        if langs is None:
            langs = ['en'] * len(texts)
        
        # Convert language codes to IDs
        lang_ids = [self.lang_map.get(lang, 0) for lang in langs]
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_langs = lang_ids[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Get predictions
            if self.use_onnx:
                # ONNX inference
                ort_inputs = {
                    'input_ids': inputs['input_ids'].numpy(),
                    'attention_mask': inputs['attention_mask'].numpy(),
                    'lang_ids': np.array(batch_langs, dtype=np.int64)
                }
                ort_outputs = self.session.run(None, ort_inputs)
                probabilities = 1 / (1 + np.exp(-ort_outputs[0]))  # sigmoid
            else:
                # PyTorch inference
                with torch.no_grad():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    lang_tensor = torch.tensor(batch_langs, dtype=torch.long, device=self.device)
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        lang_ids=lang_tensor,
                        mode='inference'
                    )
                    probabilities = outputs['probabilities'].cpu().numpy()
            
            # Format results
            for j, (text, lang, probs) in enumerate(zip(batch_texts, langs[i:i+batch_size], probabilities)):
                # Apply optimal thresholds per language
                lang_thresholds = {
                    'en': [0.58, 0.54, 0.56, 0.50, 0.55, 0.48],  # Increased by ~20% from original values
                    'default': [0.60, 0.54, 0.60, 0.48, 0.60, 0.50]  # Increased by ~20% from original values
                }
                
                thresholds = lang_thresholds.get(lang, lang_thresholds['default'])
                is_toxic = (probs >= np.array(thresholds)).astype(bool)
                
                result = {
                    'text': text,
                    'language': lang,
                    'probabilities': {
                        label: float(prob) for label, prob in zip(self.label_names, probs)
                    },
                    'is_toxic': bool(is_toxic.any()),
                    'toxic_categories': [
                        self.label_names[k] for k in range(len(is_toxic)) if is_toxic[k]
                    ]
                }
                results.append(result)
        
        return results 