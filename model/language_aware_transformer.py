import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Tuple, Optional
import logging
import os
import json

logger = logging.getLogger(__name__)

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
        
        # Validate input parameters
        if num_labels <= 0:
            raise ValueError(f"Invalid num_labels: {num_labels}")
        if hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {hidden_size}")
        if num_attention_heads <= 0:
            raise ValueError(f"Invalid num_attention_heads: {num_attention_heads}")
        if not 0 <= dropout < 1:
            raise ValueError(f"Invalid dropout: {dropout}")
        
        # Base Model Configuration
        try:
            self.config = XLMRobertaConfig.from_pretrained(
                model_name,
                hidden_size=hidden_size,
                num_hidden_layers=24,
                num_attention_heads=num_attention_heads,
                output_hidden_states=True
            )
            self.base_model = XLMRobertaModel(self.config)
            logger.info(f"Initialized base model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing base model: {str(e)}")
            raise
        
        # Language-aware Attention
        self.lang_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating Mechanism
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Feature Processing
        self.feature_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Activation and Regularization
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Output Layer with Layer Normalization
        self.pre_output = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, num_labels)
        
        # Language-specific thresholds (learnable)
        self.thresholds = nn.Parameter(torch.ones(num_labels) * 0.5)
        
        # Initialize weights
        self._init_weights()
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
    
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

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass with gated feature combination and gradient checkpointing support
        """
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
            
            # Base Model with gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                base_output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.base_model),
                    input_ids,
                    attention_mask
                )
            else:
                base_output = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            embeddings = base_output.last_hidden_state
            
            # Language-aware Attention with gradient checkpointing
            if self.gradient_checkpointing and self.training:
                def attention_forward(*inputs):
                    return self.lang_attention(*inputs)
                
                attn_output, attn_weights = torch.utils.checkpoint.checkpoint(
                    attention_forward,
                    embeddings, embeddings, embeddings,
                    ~attention_mask.bool()
                )
            else:
                attn_output, attn_weights = self.lang_attention(
                    embeddings, embeddings, embeddings,
                    key_padding_mask=~attention_mask.bool()
                )
            
            # Feature Processing
            attended_features = self.gelu(attn_output)
            attended_features = self.feature_projection(attended_features)
            
            # Gating Mechanism
            combined = torch.cat([embeddings, attended_features], dim=-1)
            gate = torch.sigmoid(self.gate_layer(combined))
            
            # Gated feature combination
            features = gate * embeddings + (1 - gate) * attended_features
            features = self.layer_norm(features)
            features = self.dropout(features)
            
            # Pool sequence dimension (use [CLS] token)
            pooled = features[:, 0]
            
            # Output with layer normalization
            pooled = self.pre_output(pooled)
            logits = self.output(pooled)
            probabilities = torch.sigmoid(logits)
            
            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            
            # Ensure all outputs are on the same device
            device = pooled.device
            return {
                'loss': loss,
                'logits': logits,
                'probabilities': probabilities,
                'thresholds': self.thresholds.to(device),
                'hidden_states': base_output.hidden_states,
                'attention_weights': attn_weights,
                'gate_values': gate.mean(dim=1)  # For monitoring gate behavior
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
                'num_labels': self.output.out_features,
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