import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from typing import Tuple, Optional
import logging

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
        
        # Base Model Configuration
        try:
            self.config = XLMRobertaConfig.from_pretrained(
                model_name,
                hidden_size=hidden_size,
                num_hidden_layers=24,
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
            batch_first=True  # Important for modern PyTorch versions
        )
        
        # Activation and Regularization
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Feature Concatenation and Processing
        self.concat_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Output Layer
        self.output = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        
        # Language-specific thresholds (learnable)
        self.thresholds = nn.Parameter(torch.ones(num_labels) * 0.5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the custom layers"""
        def _init_layer(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        # Apply to custom layers
        _init_layer(self.concat_projection)
        _init_layer(self.output)
        _init_layer(self.layer_norm)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with optional loss calculation
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for loss calculation [batch_size, num_labels]
            
        Returns:
            Tuple containing:
            - logits: Raw logits before sigmoid [batch_size, num_labels]
            - probabilities: Sigmoid activated outputs [batch_size, num_labels]
            - thresholds: Learned thresholds [num_labels]
            - loss: Optional loss value if labels provided
        """
        try:
            # Base Model
            base_output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embeddings = base_output.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # Language-aware Attention
            attn_output, _ = self.lang_attention(
                embeddings, embeddings, embeddings,
                key_padding_mask=~attention_mask.bool()
            )
            
            # Feature Processing
            attended_features = self.gelu(attn_output)
            combined = torch.cat([embeddings, attended_features], dim=-1)
            combined = self.concat_projection(combined)
            combined = self.layer_norm(combined)
            combined = self.dropout(combined)
            
            # Pool sequence dimension (use [CLS] token or mean pooling)
            pooled = combined[:, 0]  # Use [CLS] token
            
            # Output
            logits = self.output(pooled)
            probabilities = self.sigmoid(logits)
            
            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                # Binary Cross Entropy Loss
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            
            return {
                'loss': loss,
                'logits': logits,
                'probabilities': probabilities,
                'thresholds': self.thresholds,
                'hidden_states': base_output.hidden_states
            }
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get the attention weights for interpretation"""
        return self.lang_attention.get_attention_weights()
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on"""
        return next(self.parameters()).device 