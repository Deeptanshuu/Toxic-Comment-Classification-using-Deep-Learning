# language_aware_transformer.py
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Tuple, Optional
import logging
import os
import json
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_score, recall_score

logger = logging.getLogger(__name__)

class LanguageAwareClassifier(nn.Module):
    def __init__(self, hidden_size=1024, num_labels=6):
        super().__init__()
        self.lang_embed = nn.Embedding(10, 64)  # 10 languages
        
        # Language-specific dropout rates
        self.lang_dropout_rates = {
            'ru': 0.45,
            'tr': 0.45,
            'default': 0.4
        }
        
        # Base classifier layers
        self.classifier = nn.ModuleDict({
            'projection': nn.Linear(hidden_size + 64, 512),
            'activation': nn.Tanh(),
            'norm': nn.LayerNorm(512),
            'dropout': nn.Dropout(0.4),  # Default dropout
            'output': nn.Linear(512, num_labels)
        })
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.classifier.values():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, lang_ids):
        # Get language embeddings
        lang_emb = self.lang_embed(lang_ids)
        
        # Concatenate features with language embeddings
        combined = torch.cat([x, lang_emb], dim=1)
        
        # Apply classifier layers with language-specific dropout
        x = self.classifier['projection'](combined)
        x = self.classifier['activation'](x)
        x = self.classifier['norm'](x)
        
        # Apply language-specific dropout
        if self.training:
            dropout_rates = torch.tensor([
                self.lang_dropout_rates.get(str(lang_id.item()), self.lang_dropout_rates['default'])
                for lang_id in lang_ids
            ], device=x.device)
            
            # Create dropout mask for each sample
            mask = torch.rand_like(x) > dropout_rates.unsqueeze(1)
            x = x * mask * (1.0 / (1.0 - dropout_rates.unsqueeze(1)))
        else:
            x = self.classifier['dropout'](x)
        
        return self.classifier['output'](x)

class LanguageAwareTransformer(nn.Module):
    def __init__(
        self, 
        num_labels: int = 6,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        model_name: str = "xlm-roberta-large",
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Add language metric tracking
        self.lang_metrics = {
            'train': defaultdict(lambda: defaultdict(float)),  # {lang: {metric: value}}
            'val': defaultdict(lambda: defaultdict(float))
        }
        
        # Track samples per language for proper averaging
        self.lang_samples = {
            'train': defaultdict(int),
            'val': defaultdict(int)
        }
        
        # Critical class indices for specific metric tracking
        self.critical_indices = {
            'threat': 3,      # Index of threat class
            'identity_hate': 5 # Index of identity_hate class
        }
        
        # Validate input parameters
        if num_labels <= 0:
            raise ValueError(f"Invalid num_labels: {num_labels}")
        if not 0 <= dropout < 1:
            raise ValueError(f"Invalid dropout: {dropout}")
        
        # Load pretrained model with original config
        try:
            self.base_model = XLMRobertaModel.from_pretrained(model_name)
            self.config = self.base_model.config
            logger.info(f"Initialized base model: {model_name}")
            
            # Store original dimensions
            self.original_hidden_size = self.config.hidden_size
            
        except Exception as e:
            logger.error(f"Error initializing base model: {str(e)}")
            raise
        
        # Project to custom hidden size if different from original
        self.needs_projection = hidden_size != self.original_hidden_size
        if self.needs_projection:
            self.dim_projection = nn.Sequential(
                nn.Linear(self.original_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            )
        
        # Language-aware Attention with proper dimension handling
        self.lang_attention = nn.MultiheadAttention(
            embed_dim=hidden_size if self.needs_projection else self.original_hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating Mechanism with proper dimensions
        gate_input_size = hidden_size if self.needs_projection else self.original_hidden_size
        self.gate_layer = nn.Sequential(
            nn.Linear(gate_input_size * 2, gate_input_size),
            nn.LayerNorm(gate_input_size)
        )
        
        # Feature Processing
        self.feature_projection = nn.Linear(
            hidden_size if self.needs_projection else self.original_hidden_size,
            hidden_size if self.needs_projection else self.original_hidden_size
        )
        self.layer_norm = nn.LayerNorm(
            hidden_size if self.needs_projection else self.original_hidden_size
        )
        
        # Activation and Regularization
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Replace output layer with Language-Aware Classifier
        final_hidden_size = hidden_size if self.needs_projection else self.original_hidden_size
        self.pre_output = nn.LayerNorm(final_hidden_size)
        self.output = LanguageAwareClassifier(
            hidden_size=final_hidden_size,
            num_labels=num_labels
        )
        
        # Language-specific thresholds (learnable)
        self.thresholds = nn.Parameter(torch.ones(num_labels) * 0.5)
        
        # Initialize weights
        self._init_weights()
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
        logger.info(f"Model initialized with {'custom' if self.needs_projection else 'original'} hidden size: {final_hidden_size}")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
        self.base_model.gradient_checkpointing_disable()
    
    def _init_weights(self):
        """Initialize the weights of the custom layers"""
        def _init_layer(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Use Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
        # Apply to custom layers
        _init_layer(self.gate_layer[0])  # Linear layer in gate
        _init_layer(self.feature_projection)
        _init_layer(self.layer_norm)
        _init_layer(self.pre_output)
        _init_layer(self.output)

    def _calculate_language_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lang_ids: torch.Tensor,
        loss: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> dict:
        """Calculate detailed per-language metrics"""
        try:
            # Get probabilities
            probs = torch.sigmoid(logits)
            
            # Convert to numpy for metric calculation
            probs_np = probs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Get unique languages in batch
            unique_langs = torch.unique(lang_ids).cpu().tolist()
            
            metrics = {}
            for lang_id in unique_langs:
                # Get mask for this language
                lang_mask = lang_ids == lang_id
                if not lang_mask.any():
                    continue
                
                # Update sample count
                self.lang_samples[mode][lang_id] += lang_mask.sum().item()
                
                # Calculate metrics for this language
                lang_metrics = {}
                
                # Loss if provided
                if loss is not None:
                    lang_loss = loss[lang_mask].mean().item()
                    lang_metrics['loss'] = lang_loss
                    self.lang_metrics[mode][lang_id]['loss'] = (
                        self.lang_metrics[mode][lang_id]['loss'] * 0.9 + 
                        lang_loss * 0.1  # EMA update
                    )
                
                # AUC for validation
                if mode == 'val':
                    try:
                        lang_auc = roc_auc_score(
                            labels_np[lang_mask],
                            probs_np[lang_mask],
                            average='macro'
                        )
                        lang_metrics['auc'] = lang_auc
                        self.lang_metrics[mode][lang_id]['auc'] = (
                            self.lang_metrics[mode][lang_id]['auc'] * 0.9 + 
                            lang_auc * 0.1  # EMA update
                        )
                    except ValueError:
                        pass  # Skip if only one class present
                
                # Critical class metrics
                for class_name, idx in self.critical_indices.items():
                    # Get class-specific predictions and labels
                    class_preds = (probs_np[lang_mask, idx] > 0.5).astype(int)
                    class_labels = labels_np[lang_mask, idx]
                    
                    # Calculate metrics
                    try:
                        precision = precision_score(class_labels, class_preds)
                        recall = recall_score(class_labels, class_preds)
                        
                        # Store metrics
                        if class_name == 'threat':
                            lang_metrics['threat_recall'] = recall
                            self.lang_metrics[mode][lang_id]['threat_recall'] = (
                                self.lang_metrics[mode][lang_id]['threat_recall'] * 0.9 + 
                                recall * 0.1
                            )
                        elif class_name == 'identity_hate':
                            lang_metrics['identity_precision'] = precision
                            self.lang_metrics[mode][lang_id]['identity_precision'] = (
                                self.lang_metrics[mode][lang_id]['identity_precision'] * 0.9 + 
                                precision * 0.1
                            )
                    except:
                        pass  # Skip if no positive samples
                
                metrics[str(lang_id)] = lang_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating language metrics: {str(e)}")
            return {}

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        mode: str = 'train'  # Add mode parameter
    ) -> dict:
        """Forward pass with language-specific metric tracking"""
        try:
            # Input validation
            if input_ids is None or attention_mask is None:
                raise ValueError("input_ids and attention_mask must not be None")
            if input_ids.dim() != 2 or attention_mask.dim() != 2:
                raise ValueError(f"Expected 2D tensors, got input_ids: {input_ids.dim()}D, attention_mask: {attention_mask.dim()}D")
            if input_ids.shape != attention_mask.shape:
                raise ValueError(f"Shape mismatch: input_ids {input_ids.shape} != attention_mask {attention_mask.shape}")
            if labels is not None and labels.shape[0] != input_ids.shape[0]:
                raise ValueError(f"Batch size mismatch: labels {labels.shape[0]} != input_ids {input_ids.shape[0]}")
            
            # Default lang_ids to English (0) if not provided
            if lang_ids is None:
                lang_ids = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            
            # Use automatic mixed precision for better memory efficiency
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
            
            with torch.autocast(device_type=device_type, dtype=dtype):
                # Base Model with gradient checkpointing if enabled
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    
                    base_output = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.base_model),
                        input_ids,
                        attention_mask,
                        use_reentrant=False  # More memory efficient
                    )
                else:
                    base_output = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                embeddings = base_output.last_hidden_state
                
                # Project dimensions if needed (inplace where possible)
                if self.needs_projection:
                    embeddings = self.dim_projection(embeddings)
                    if isinstance(embeddings, tuple):
                        embeddings = embeddings[0]
                
                # Memory-efficient attention using scaled dot product
                attention_mask_expanded = ~attention_mask.bool()
                if attention_mask_expanded.dim() == 2:
                    attention_mask_expanded = attention_mask_expanded.unsqueeze(1)
                
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    embeddings,  # query
                    embeddings,  # key
                    embeddings,  # value
                    attn_mask=attention_mask_expanded,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
                
                # Feature Processing with inplace operations
                attended_features = torch.nn.functional.gelu(attn_output)
                attended_features = self.feature_projection(attended_features)
                
                # Gating Mechanism with inplace operations
                combined = torch.cat([embeddings, attended_features], dim=-1)
                gate = torch.sigmoid_(self.gate_layer(combined)[0])  # Inplace sigmoid
                
                # Gated feature combination using efficient inplace ops
                features = torch.addcmul(
                    embeddings.mul_(gate),  # embeddings * gate (inplace)
                    attended_features,
                    1 - gate,  # Compute (1-gate) without new allocation
                    value=1.0
                )
                
                # Apply layer norm and dropout (inplace where possible)
                features = self.layer_norm(features)
                if self.training:
                    features = torch.nn.functional.dropout(
                        features,
                        p=self.dropout.p,
                        training=True,
                        inplace=True
                    )
                
                # Pool sequence dimension (use [CLS] token)
                pooled = features[:, 0]
                
                # Output projection with layer normalization and language-aware classification
                pooled = self.pre_output(pooled)
                logits = self.output(pooled, lang_ids)
                
                # Calculate probabilities (inplace)
                probabilities = torch.sigmoid(logits)
                
                # Calculate loss if labels provided
                loss = None
                if labels is not None:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
            
            # Ensure all outputs are on the same device
            device = pooled.device
            outputs = {
                'loss': loss,
                'logits': logits,
                'probabilities': probabilities,
                'thresholds': self.thresholds.to(device),
                'hidden_states': base_output.hidden_states,
                'attention_weights': attn_output,
                'gate_values': gate.mean(dim=1)  # For monitoring gate behavior
            }
            
            # Clean up any unnecessary tensors
            del base_output, embeddings, attn_output, attended_features, combined
            torch.cuda.empty_cache()
            
            # Calculate language-specific metrics if labels provided
            if labels is not None:
                lang_metrics = self._calculate_language_metrics(
                    logits,
                    labels,
                    lang_ids,
                    loss,
                    mode=mode
                )
                outputs['lang_metrics'] = lang_metrics
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get the attention weights for interpretation"""
        try:
            return self.lang_attention.get_attention_weights()
        except Exception as e:
            logger.error(f"Error getting attention weights: {str(e)}")
            return None
    
    def save_pretrained(self, save_path: str):
        """Save model to the specified path"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save model state dict
            model_path = os.path.join(save_path, 'pytorch_model.bin')
            torch.save(self.state_dict(), model_path)
            
            # Save config
            config_dict = {
                'num_labels': self.output.classifier['output'].out_features,
                'hidden_size': self.config.hidden_size,
                'num_attention_heads': self.config.num_attention_heads,
                'model_name': self.config.name_or_path,
                'dropout': self.dropout.p
            }
            config_path = os.path.join(save_path, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            logger.warning("No parameters found in model")
            return torch.device('cpu')

    def get_language_metrics(self, mode: str = 'train') -> dict:
        """Get accumulated language metrics"""
        try:
            metrics = {}
            for lang_id, lang_metrics in self.lang_metrics[mode].items():
                # Only include languages with samples
                if self.lang_samples[mode][lang_id] > 0:
                    metrics[str(lang_id)] = {
                        name: value 
                        for name, value in lang_metrics.items()
                    }
            return metrics
        except Exception as e:
            logger.error(f"Error getting language metrics: {str(e)}")
            return {}

    def reset_language_metrics(self, mode: str = 'train'):
        """Reset accumulated language metrics"""
        try:
            self.lang_metrics[mode].clear()
            self.lang_samples[mode].clear()
        except Exception as e:
            logger.error(f"Error resetting language metrics: {str(e)}") 