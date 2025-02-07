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
        
        # Language-specific dropout rates with numeric keys
        self.lang_dropout_rates = {
            1: 0.45,  # Russian
            2: 0.45,  # Turkish
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
        # Ensure lang_ids is a tensor of integers
        if not isinstance(lang_ids, torch.Tensor):
            lang_ids = torch.tensor(lang_ids, dtype=torch.long, device=x.device)
        elif lang_ids.dtype != torch.long:
            lang_ids = lang_ids.long()
        
        # Ensure lang_ids is on the correct device
        lang_ids = lang_ids.to(x.device)
        
        # Clamp language IDs to valid range [0, 9]
        lang_ids = torch.clamp(lang_ids, min=0, max=9)
        
        # Get language embeddings
        lang_emb = self.lang_embed(lang_ids)
        
        # Concatenate features with language embeddings
        combined = torch.cat([x, lang_emb], dim=1)
        
        # Apply classifier layers with language-specific dropout
        x = self.classifier['projection'](combined)
        x = self.classifier['activation'](x)
        x = self.classifier['norm'](x)
        
        # Apply language-specific dropout during training
        if self.training:
            dropout_rates = torch.tensor([
                self.lang_dropout_rates.get(
                    int(lang_id.item()) if isinstance(lang_id.item(), (int, float)) else 'default',
                    self.lang_dropout_rates['default']
                )
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
            # Get probabilities and convert to float32
            probs = torch.sigmoid(logits)
            
            # Convert to numpy with float32 for metric calculation
            probs_np = probs.detach().cpu().to(torch.float32).numpy()
            labels_np = labels.detach().cpu().to(torch.float32).numpy()
            
            # Get unique languages in batch
            unique_langs = torch.unique(lang_ids).cpu().tolist()
            
            metrics = {}
            for lang_id in unique_langs:
                # Get mask for this language
                lang_mask = (lang_ids == lang_id).cpu().numpy()
                if not lang_mask.any():
                    continue
                
                # Update sample count
                self.lang_samples[mode][lang_id] += lang_mask.sum()
                
                # Calculate metrics for this language
                lang_metrics = {}
                
                # Loss if provided
                if loss is not None and not isinstance(loss, float):
                    # Handle per-sample loss
                    if loss.dim() > 0:  # If loss is a tensor with dimensions
                        lang_loss = loss.detach().cpu()[lang_mask].mean().item()
                    else:  # If loss is a scalar tensor
                        lang_loss = loss.detach().cpu().item()
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
                    except ValueError as e:
                        logger.warning(f"Could not calculate AUC for language {lang_id}: {str(e)}")
                        pass  # Skip if only one class present
                
                # Critical class metrics
                for class_name, idx in self.critical_indices.items():
                    try:
                        # Get class-specific predictions and labels
                        class_preds = (probs_np[lang_mask, idx] > 0.5).astype(int)
                        class_labels = labels_np[lang_mask, idx]
                        
                        # Skip if no samples for this class
                        if len(class_labels) == 0:
                            continue
                        
                        # Calculate metrics
                        precision = precision_score(class_labels, class_preds, zero_division=0)
                        recall = recall_score(class_labels, class_preds, zero_division=0)
                        
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
                    except Exception as e:
                        logger.warning(f"Could not calculate metrics for {class_name} in language {lang_id}: {str(e)}")
                        continue
                
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
        mode: str = 'train'
    ) -> dict:
        """Forward pass with language-specific metric tracking"""
        try:
            # Input validation and device setup
            device = input_ids.device
            batch_size = input_ids.size(0)
            
            # Handle language IDs with proper error handling
            if lang_ids is None:
                logger.warning("No language IDs provided, defaulting to English (0)")
                lang_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif not isinstance(lang_ids, torch.Tensor):
                try:
                    lang_ids = torch.tensor(lang_ids, dtype=torch.long, device=device)
                except Exception as e:
                    logger.error(f"Failed to convert lang_ids to tensor: {str(e)}")
                    lang_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            elif lang_ids.dtype != torch.long:
                lang_ids = lang_ids.long()
            
            # Ensure lang_ids is on correct device and shape
            lang_ids = lang_ids.to(device)
            if lang_ids.dim() > 1:
                lang_ids = lang_ids.squeeze()
            if lang_ids.dim() == 0:
                lang_ids = lang_ids.unsqueeze(0)
            
            # Clamp language IDs to valid range [0, 9]
            lang_ids = torch.clamp(lang_ids, min=0, max=9)
            
            # Ensure attention_mask is 2D [batch_size, seq_len]
            if attention_mask.dim() == 4:  # If shape is [batch_size, 1, seq_len, seq_len]
                attention_mask = attention_mask.squeeze(1).squeeze(1)
            elif attention_mask.dim() == 3:  # If shape is [batch_size, 1, seq_len]
                attention_mask = attention_mask.squeeze(1)
            
            # Convert attention mask to float and create causal mask for attention
            attention_mask = attention_mask.to(dtype=torch.float32)
            
            # Use automatic mixed precision
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                # Base model forward pass
                base_output = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                hidden_states = base_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
                
                # Get language embeddings and combine with hidden states
                lang_embeddings = self.output.lang_embed(lang_ids)  # [batch_size, embed_dim]
                lang_embeddings = lang_embeddings.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
                
                # Project hidden states if needed
                if self.needs_projection:
                    hidden_states = self.dim_projection(hidden_states)
                
                # Combine features
                combined_features = torch.cat([hidden_states, lang_embeddings], dim=-1)
                
                # Create attention mask for scaled dot product attention
                # Shape: [batch_size, seq_len, seq_len]
                extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.size(1), -1)
                
                # Memory-efficient attention with proper mask type
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    combined_features,  # query [batch_size, seq_len, hidden_size]
                    combined_features,  # key
                    combined_features,  # value
                    attn_mask=extended_attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
                
                # Feature Processing
                attended_features = torch.nn.functional.gelu(attn_output)
                attended_features = self.feature_projection(attended_features)
                
                # Gating Mechanism
                gate_input = torch.cat([combined_features, attended_features], dim=-1)
                gate = torch.sigmoid(self.gate_layer(gate_input))
                
                # Combine gated features
                features = combined_features * gate + attended_features * (1 - gate)
                
                # Final classification using [CLS] token
                pooled = features[:, 0]  # Use [CLS] token
                logits = self.output(pooled, lang_ids)
                
                # Calculate loss if labels provided
                loss = None
                if labels is not None:
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels.float())
                
                # Get probabilities
                probs = torch.sigmoid(logits)
                
                # Calculate metrics if in eval mode
                if mode != 'train' and labels is not None:
                    self._calculate_language_metrics(probs, labels, lang_ids, loss, mode)
                
                return {
                    'loss': loss,
                    'logits': logits,
                    'probabilities': probs,
                    'hidden_states': hidden_states,
                    'attention_weights': attn_output,
                    'gate_values': gate.mean(dim=1)
                }
                
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