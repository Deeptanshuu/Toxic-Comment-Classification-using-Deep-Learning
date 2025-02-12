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

class LanguageAwareClassifier(nn.Module):
    def __init__(self, hidden_size=1024, num_labels=6):
        super().__init__()
        self.lang_embed = nn.Embedding(7, 64)  # 7 languages
        
        # Base classifier layers
        self.classifier = nn.ModuleDict({
            'projection': nn.Linear(hidden_size + 64, 512),
            'activation': nn.Tanh(),
            'norm': nn.LayerNorm(512),
            'dropout': nn.Dropout(0.0),
            'output': nn.Linear(512, num_labels)
        })
        
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
        
        # Get language embeddings
        lang_emb = self.lang_embed(lang_ids)
        
        # Concatenate features with language embeddings
        combined = torch.cat([x, lang_emb], dim=1)
        
        # Apply classifier layers
        x = self.classifier['projection'](combined)
        x = self.classifier['activation'](x)
        x = self.classifier['norm'](x)
        x = self.classifier['dropout'](x)
        return self.classifier['output'](x)

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, logits, targets, weights=None):
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        if weights is not None:
            loss = loss * weights
        if self.reduction == 'mean':
            return loss.mean()
        return loss

class LanguageAwareTransformer(nn.Module):
    def __init__(
        self, 
        num_labels: int = 6,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        model_name: str = "xlm-roberta-large",
        dropout: float = 0.0
    ):
        super().__init__()
        
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
        
        # Language-aware layers
        self.pre_attention = nn.Sequential(
            nn.Linear(self.working_hidden_size + 64, self.working_hidden_size),
            nn.LayerNorm(self.working_hidden_size),
            nn.GELU()
        )
        
        self.lang_attention = nn.MultiheadAttention(
            embed_dim=self.working_hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.post_attention = nn.Sequential(
            nn.Linear(self.working_hidden_size, self.working_hidden_size),
            nn.LayerNorm(self.working_hidden_size),
            nn.GELU()
        )
        
        # Output layer
        self.output = LanguageAwareClassifier(
            hidden_size=self.working_hidden_size,
            num_labels=num_labels
        )
        
        self._init_weights()
        self.gradient_checkpointing = False
    
    def _init_weights(self):
        """Initialize weights of custom layers"""
        for module in [self.pre_attention, self.post_attention]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.constant_(layer.bias, 0)
                        nn.init.constant_(layer.weight, 1.0)
    
    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        self.base_model.gradient_checkpointing_disable()
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> dict:
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Handle language IDs
        if lang_ids is None:
            lang_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif not isinstance(lang_ids, torch.Tensor):
            lang_ids = torch.tensor(lang_ids, dtype=torch.long, device=device)
        
        # Base model forward pass
        hidden_states = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        # Project if needed
        if self.needs_projection:
            hidden_states = self.dim_projection(hidden_states)
        
        # Get language embeddings
        lang_embeddings = self.output.lang_embed(lang_ids)
        lang_embeddings = lang_embeddings.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        # Process features
        combined = torch.cat([hidden_states, lang_embeddings], dim=-1)
        features = self.pre_attention(combined)
        
        # Apply attention
        attention_output, _ = self.lang_attention(
            query=features,
            key=features,
            value=features,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False
        )
        
        # Post-process
        features = self.post_attention(attention_output)
        
        # Classification
        logits = self.output(features[:, 0], lang_ids)
        
        # Calculate loss if needed
        loss = None
        if labels is not None:
            loss_fct = WeightedBCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return {
            'loss': loss,
            'logits': logits,
            'probabilities': torch.sigmoid(logits)
        }
    
    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        config_dict = {
            'num_labels': self.output.classifier['output'].out_features,
            'hidden_size': self.config.hidden_size,
            'num_attention_heads': self.config.num_attention_heads,
            'model_name': self.config.name_or_path,
            'dropout': self.lang_attention.dropout
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
