import torch
import onnxruntime as ort
from transformers import XLMRobertaTokenizer
import numpy as np

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
        if onnx_path:
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
            self.model = LanguageAwareTransformer(num_labels=6)
            self.model.load_state_dict(torch.load(pytorch_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            self.use_onnx = False
            self.device = device
            print(f"Loaded PyTorch model from {pytorch_path}")
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
                    'en': [0.48, 0.45, 0.47, 0.42, 0.46, 0.40],  # Tuned based on analysis
                    'default': [0.50, 0.45, 0.50, 0.40, 0.50, 0.42]
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