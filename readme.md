# Toxic Comment Classification using Deep Learning

A multilingual toxic comment classification system using language-aware transformers and advanced deep learning techniques.

## 🏗️ Architecture Overview

### Core Components

1. **LanguageAwareTransformer**
   - Base: XLM-RoBERTa Large
   - Custom language-aware attention mechanism
   - Gating mechanism for feature fusion
   - Language-specific dropout rates
   - Support for 7 languages with English fallback

2. **ToxicDataset**
   - Efficient caching system
   - Language ID mapping
   - Memory pinning for CUDA optimization
   - Automatic handling of missing values

3. **Training System**
   - Mixed precision training (BF16/FP16)
   - Gradient accumulation
   - Language-aware loss weighting
   - Distributed training support
   - Automatic threshold optimization

### Key Features

- **Language Awareness**
  - Language-specific embeddings
  - Dynamic dropout rates per language
  - Language-aware attention mechanism
  - Automatic fallback to English for unsupported languages

- **Performance Optimization**
  - Gradient checkpointing
  - Memory-efficient attention
  - Automatic mixed precision
  - Caching system for processed data
  - CUDA optimization with memory pinning

- **Training Features**
  - Weighted focal loss with language awareness
  - Dynamic threshold optimization
  - Early stopping with patience
  - Gradient flow monitoring
  - Comprehensive metric tracking

## 📊 Data Processing

### Input Format
```python
{
    'comment_text': str,  # The text to classify
    'lang': str,          # Language code (en, ru, tr, es, fr, it, pt)
    'toxic': int,         # Binary labels for each category
    'severe_toxic': int,
    'obscene': int,
    'threat': int,
    'insult': int,
    'identity_hate': int
}
```

### Language Support
- Primary: en, ru, tr, es, fr, it, pt
- Default fallback: en (English)
- Language ID mapping: {en: 0, ru: 1, tr: 2, es: 3, fr: 4, it: 5, pt: 6}

## 🚀 Model Architecture

### Base Model
- XLM-RoBERTa Large
- Hidden size: 1024
- Attention heads: 16
- Max sequence length: 128

### Custom Components

1. **Language-Aware Classifier**
```python
- Input: Hidden states [batch_size, hidden_size]
- Language embeddings: [batch_size, 64]
- Projection: hidden_size + 64 -> 512
- Output: 6 toxicity predictions
```

2. **Language-Aware Attention**
```python
- Input: Hidden states + Language embeddings
- Scaled dot product attention
- Gating mechanism for feature fusion
- Memory-efficient implementation
```

## 🛠️ Training Configuration

### Hyperparameters
```python
{
    "batch_size": 32,
    "grad_accum_steps": 2,
    "epochs": 4,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "label_smoothing": 0.01,
    "model_dropout": 0.1,
    "freeze_layers": 2
}
```

### Optimization
- Optimizer: AdamW
- Learning rate scheduler: Cosine with warmup
- Mixed precision: BF16/FP16
- Gradient clipping: 1.0
- Gradient accumulation steps: 2

## 📈 Metrics and Monitoring

### Training Metrics
- Loss (per language)
- AUC-ROC (macro)
- Precision, Recall, F1
- Language-specific metrics
- Gradient norms
- Memory usage

### Validation Metrics
- AUC-ROC (per class and language)
- Optimal thresholds per language
- Critical class performance (threat, identity_hate)
- Distribution shift monitoring

## 🔧 Usage

### Training
```bash
python model/train.py
```

### Inference
```python
from model.predict import predict_toxicity

results = predict_toxicity(
    text="Your text here",
    model=model,
    tokenizer=tokenizer,
    config=config
)
```

## 🔍 Code Structure

```
model/
├── language_aware_transformer.py  # Core model architecture
├── train.py                      # Training loop and utilities
├── predict.py                    # Inference utilities
├── evaluation/
│   ├── evaluate.py              # Evaluation functions
│   └── threshold_optimizer.py    # Dynamic threshold optimization
├── data/
│   └── sampler.py               # Custom sampling strategies
└── training_config.py           # Configuration management
```

## 🤖 AI/ML Specific Notes

1. **Tensor Shapes**
   - Input IDs: [batch_size, seq_len]
   - Attention Mask: [batch_size, seq_len]
   - Language IDs: [batch_size]
   - Hidden States: [batch_size, seq_len, hidden_size]
   - Language Embeddings: [batch_size, embed_dim]

2. **Critical Components**
   - Language ID handling in forward pass
   - Attention mask shape management
   - Memory-efficient attention implementation
   - Gradient flow in language-aware components

3. **Performance Considerations**
   - Cache management for processed data
   - Memory pinning for GPU transfers
   - Gradient accumulation for large batches
   - Language-specific dropout rates

4. **Error Handling**
   - Language ID validation
   - Shape compatibility checks
   - Gradient norm monitoring
   - Device placement verification

## 📝 Notes for AI Systems

1. When modifying the code:
   - Maintain language ID handling in forward pass
   - Preserve attention mask shape management
   - Keep device consistency checks
   - Handle BatchEncoding security in PyTorch 2.6+

2. Key attention points:
   - Language ID tensor shape and type
   - Attention mask broadcasting
   - Memory-efficient attention implementation
   - Gradient flow through language-aware components

3. Common pitfalls:
   - Incorrect attention mask shapes
   - Language ID type mismatches
   - Memory leaks in caching
   - Device inconsistencies
