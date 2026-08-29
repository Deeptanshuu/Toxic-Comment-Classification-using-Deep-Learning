# language_aware_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMRobertaModel
from typing import Optional
import logging
import os
import json

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    'en': 0, 'ru': 1, 'tr': 2, 'es': 3,
    'fr': 4, 'it': 5, 'pt': 6
}

def validate_lang_ids(lang_ids):
    if not isinstance(lang_ids, torch.Tensor):
        lang_ids = torch.tensor(lang_ids, dtype=torch.long)
    # Use actual language count instead of hardcoded 9
    return torch.clamp(lang_ids, min=0, max=len(SUPPORTED_LANGUAGES)-1)

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, weights=None):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt)**self.gamma * bce_loss
        if weights is not None:
            focal_loss *= weights
        return focal_loss.mean()

class LanguageAwareTransformer(nn.Module):
    def __init__(
        self, 
        num_labels: int = 6,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        model_name: str = "xlm-roberta-large",
        dropout: float = 0.0,
        disable_lang_conditioning: bool = False
    ):
        super().__init__()

        # Ablation switch: when True the language pathway contributes nothing and the
        # model reduces to a language-agnostic control. Can also be flipped on an
        # existing instance (model.disable_lang_conditioning = True).
        self.disable_lang_conditioning = disable_lang_conditioning

        # Validate supported languages
        if not SUPPORTED_LANGUAGES:
            raise ValueError("No supported languages defined")
        logger.info(f"Initializing model with {len(SUPPORTED_LANGUAGES)} supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
        
        # Load pretrained model
        self.base_model = XLMRobertaModel.from_pretrained(model_name)
        self.config = self.base_model.config
        
        # Project to custom hidden size if different from original
        self.original_hidden_size = self.config.hidden_size
        self.needs_projection = hidden_size != self.original_hidden_size
        if self.needs_projection:
            self.dim_projection = nn.Sequential(
                nn.Linear(self.original_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            )
        
        # Working hidden size
        self.working_hidden_size = hidden_size if self.needs_projection else self.original_hidden_size
        
        # Language-aware attention components with dynamic language count
        num_languages = len(SUPPORTED_LANGUAGES)
        self.lang_embed = nn.Embedding(num_languages, 64)
        
        # Register supported languages for validation
        self.register_buffer('valid_lang_ids', torch.arange(num_languages))
        
        # Optimized language projection for attention bias
        self.lang_proj = nn.Sequential(
            nn.Linear(64, num_attention_heads * hidden_size // num_attention_heads),
            nn.LayerNorm(num_attention_heads * hidden_size // num_attention_heads),
            nn.Tanh()  # Bounded activation for stable attention scores
        )
        
        # Multi-head attention with optimized head dimension
        head_dim = hidden_size // num_attention_heads
        self.scale = head_dim ** -0.5  # Scaling factor for attention scores
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.post_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, num_labels)
        )
        
        self._init_weights()
        self.gradient_checkpointing = False
    
    def _init_weights(self):
        """Initialize weights with careful scaling"""
        for module in [self.lang_proj, self.q_proj, self.k_proj, self.v_proj, 
                      self.post_attention, self.classifier]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        # Use scaled initialization for attention projections
                        if layer in [self.q_proj, self.k_proj, self.v_proj]:
                            nn.init.normal_(layer.weight, std=0.02)
                        else:
                            nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.ones_(layer.weight)
                        nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                if module in [self.q_proj, self.k_proj, self.v_proj]:
                    nn.init.normal_(module.weight, std=0.02)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.base_model.gradient_checkpointing_disable()
    
    def validate_lang_ids(self, lang_ids: torch.Tensor) -> torch.Tensor:
        """
        Validate and normalize language IDs
        Args:
            lang_ids: Tensor of language IDs
        Returns:
            Validated and normalized language ID tensor
        Raises:
            ValueError if too many invalid IDs detected
        """
        if not isinstance(lang_ids, torch.Tensor):
            lang_ids = torch.tensor(lang_ids, dtype=torch.long, device=self.valid_lang_ids.device)
        elif lang_ids.dtype != torch.long:
            lang_ids = lang_ids.long()
        
        # Check for out-of-bounds IDs
        invalid_mask = ~torch.isin(lang_ids, self.valid_lang_ids)
        num_invalid = invalid_mask.sum().item()
        
        if num_invalid > 0:
            invalid_ratio = num_invalid / lang_ids.numel()
            if invalid_ratio > 0.1:  # More than 10% invalid
                raise ValueError(
                    f"Too many invalid language IDs detected ({num_invalid} out of {lang_ids.numel()}). "
                    f"Valid range is 0-{len(SUPPORTED_LANGUAGES)-1}"
                )
            # Log warning and clamp invalid IDs
            logger.warning(
                f"Found {num_invalid} invalid language IDs. "
                f"Valid range is 0-{len(SUPPORTED_LANGUAGES)-1}. "
                "Invalid IDs will be clamped to valid range."
            )
            lang_ids = torch.clamp(lang_ids, min=0, max=len(SUPPORTED_LANGUAGES)-1)
        
        return lang_ids
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        mode: str = 'train',
        compute_loss: bool = False
    ) -> dict:
        # `mode` is unused; kept because inference_optimized.py passes mode='inference'
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Handle language IDs with validation
        if lang_ids is None:
            lang_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Validate and normalize language IDs
        try:
            lang_ids = self.validate_lang_ids(lang_ids)
        except ValueError as e:
            logger.error(f"Language ID validation failed: {str(e)}")
            logger.error("Falling back to default language (0)")
            lang_ids = torch.zeros_like(lang_ids)
        
        # Base model forward pass
        hidden_states = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # Check for numerical instabilities
        if hidden_states.isnan().any():
            raise ValueError("NaN detected in hidden states")
        if hidden_states.isinf().any():
            raise ValueError("Inf detected in hidden states")

        # Project if needed
        if self.needs_projection:
            hidden_states = self.dim_projection(hidden_states)
        
        # Reshape for multi-head attention
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads

        # Project queries, keys, and values
        q = self.q_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Language-aware bias: a per-language offset added to every query, which
        # contributes lang_vec . k_j to the score for key j. The term varies along the
        # key axis (dim=-1), so softmax does not cancel it. lang_proj ends in Tanh, so
        # the offset is bounded; scaling by self.scale keeps it in the same units as
        # the content scores.
        if not self.disable_lang_conditioning:
            lang_emb = self.lang_embed(lang_ids)  # [batch_size, 64]
            lang_vec = self.lang_proj(lang_emb).view(batch_size, num_heads, 1, head_dim)
            attn_scores = attn_scores + torch.matmul(lang_vec, k.transpose(-2, -1)) * self.scale

        # Apply attention mask after the bias so padded keys stay masked out
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.bool().unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Compute attention weights and apply to values
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attention_output = torch.matmul(attn_weights, v)
        
        # Reshape and post-process
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        output = self.post_attention(attention_output)
        
        # Get logits using the [CLS] token output
        logits = self.classifier(output[:, 0])
        
        probabilities = torch.sigmoid(logits)

        # Prepare output dictionary
        result = {
            'logits': logits,
            'probabilities': probabilities
        }

        # Loss is opt-in: train.py computes its own and discards anything set here
        if compute_loss and labels is not None:
            loss_fct = WeightedBCEWithLogitsLoss()
            result['loss'] = loss_fct(logits, labels.float())

        return result
    
    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        config_dict = {
            'num_labels': self.classifier[-1].out_features,
            'hidden_size': self.config.hidden_size,
            'num_attention_heads': self.config.num_attention_heads,
            'model_name': self.config.name_or_path,
            'dropout': self.dropout.p
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
