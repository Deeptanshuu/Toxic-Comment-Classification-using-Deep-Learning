"""Model definition for Deeptanshuu/toxic-comment-multilingual-xlmr.

This is a custom architecture, not a stock Hugging Face model. It is
XLM-RoBERTa-large followed by one extra attention block whose scores carry a
per-language bias, then a small classification head with six independent
sigmoid outputs.

Two ways to load it, both supported:

1. Auto classes with remote code (the file you are reading is fetched from the
   repo and executed, which is why `trust_remote_code=True` is required):

       >>> from transformers import AutoModel, AutoTokenizer
       >>> model = AutoModel.from_pretrained(REPO, trust_remote_code=True)
       >>> tok = AutoTokenizer.from_pretrained(REPO)

2. The explicit helper at the bottom of this file, which needs no auto-class
   machinery and no remote-code execution:

       >>> from modeling_toxic_xlmr import load_model
       >>> model, tokenizer = load_model("/path/to/local/checkout")

Note the `>>> ` prefixes above. transformers scans this file line by line for
import statements when it loads it as remote code, and an unprefixed
`from modeling_toxic_xlmr import ...` in a docstring is read as a dependency on
a PyPI package of that name, which then fails to install.

The forward pass takes a third input beyond the usual pair:

    model(input_ids=..., attention_mask=..., lang_ids=...)

`lang_ids` is a LongTensor of shape [batch] holding one id per sequence, using
LANGUAGE_IDS below. If you omit it the model falls back to id 0 (English) for
every row and warns once. That is a silent quality loss on non-English text,
not an error.
"""

from dataclasses import dataclass
from typing import Optional
import json
import logging
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, XLMRobertaConfig, XLMRobertaModel
from transformers.utils import ModelOutput

logger = logging.getLogger(__name__)

# Language id mapping. These integers are baked into the trained lang_embed
# table; changing the mapping silently changes which language the model thinks
# it is looking at.
LANGUAGE_IDS = {
    "en": 0,  # English
    "ru": 1,  # Russian
    "tr": 2,  # Turkish
    "es": 3,  # Spanish
    "fr": 4,  # French
    "it": 5,  # Italian
    "pt": 6,  # Portuguese
}
ID_TO_LANGUAGE = {v: k for k, v in LANGUAGE_IDS.items()}

# Output order of the classifier. Do not reorder: index 1 is severe_toxic and
# index 2 is obscene, which is not the alphabetical or the intuitive order.
LABEL_NAMES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

_WARNED_MISSING_LANG_IDS = False


class ToxicCommentConfig(PretrainedConfig):
    """Configuration for :class:`ToxicCommentModel`.

    The XLM-RoBERTa encoder configuration is nested under ``encoder`` as a plain
    dict so the whole architecture is described by one config.json and the
    encoder can be rebuilt without a network fetch.
    """

    model_type = "toxic_comment_xlmr"

    def __init__(
        self,
        encoder=None,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        dropout: float = 0.0,
        disable_lang_conditioning: bool = False,
        languages=None,
        max_seq_length: int = 512,
        **kwargs,
    ):
        # Nested encoder config. Default is whatever XLMRobertaConfig defaults
        # to; the shipped config.json carries the real xlm-roberta-large values.
        self.encoder = encoder if encoder is not None else XLMRobertaConfig().to_dict()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.disable_lang_conditioning = disable_lang_conditioning
        self.languages = dict(languages) if languages is not None else dict(LANGUAGE_IDS)
        # Training-time tokenizer truncation length. Named max_seq_length and not
        # max_length because PretrainedConfig already owns max_length as a text
        # generation parameter and would overwrite it with 20.
        self.max_seq_length = max_seq_length

        kwargs.setdefault("problem_type", "multi_label_classification")
        kwargs.setdefault("id2label", {i: name for i, name in enumerate(LABEL_NAMES)})
        kwargs.setdefault("label2id", {name: i for i, name in enumerate(LABEL_NAMES)})
        super().__init__(**kwargs)


@dataclass
class ToxicCommentOutput(ModelOutput):
    """Output of :class:`ToxicCommentModel`.

    Supports both ``out.logits`` and ``out["probabilities"]`` access.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    probabilities: Optional[torch.FloatTensor] = None


class ToxicCommentModel(PreTrainedModel):
    """XLM-RoBERTa-large + one language-conditioned attention block + MLP head."""

    config_class = ToxicCommentConfig
    base_model_prefix = "base_model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["XLMRobertaLayer"]

    def __init__(self, config: ToxicCommentConfig):
        super().__init__(config)

        encoder_config = XLMRobertaConfig.from_dict(config.encoder)
        # Constructed from config rather than from_pretrained: every weight is
        # about to be overwritten by the checkpoint, so downloading the stock
        # xlm-roberta-large weights first would be 2.2 GB of wasted transfer.
        self.base_model = XLMRobertaModel(encoder_config)

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_labels = len(LABEL_NAMES)

        self.original_hidden_size = encoder_config.hidden_size
        self.needs_projection = hidden_size != self.original_hidden_size
        if self.needs_projection:
            self.dim_projection = nn.Sequential(
                nn.Linear(self.original_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )
        self.working_hidden_size = hidden_size if self.needs_projection else self.original_hidden_size

        num_languages = len(config.languages)
        self.lang_embed = nn.Embedding(num_languages, 64)
        self.register_buffer("valid_lang_ids", torch.arange(num_languages))

        self.lang_proj = nn.Sequential(
            nn.Linear(64, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),  # bounded, so the bias cannot saturate the softmax
        )

        self.head_dim = hidden_size // num_heads
        self.num_attention_heads = num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.post_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, num_labels),
        )

        self.disable_lang_conditioning = bool(config.disable_lang_conditioning)

        self.post_init()

    # PreTrainedModel exposes `base_model` as a property that resolves
    # `base_model_prefix`. Our encoder submodule is literally named
    # `base_model` (the checkpoint keys are `base_model.*`), so the inherited
    # property would call getattr(self, "base_model") on itself and recurse.
    # Read it out of the module registry instead.
    @property
    def base_model(self) -> nn.Module:
        module = self._modules.get("base_model")
        return self if module is None else module

    def _init_weights(self, module):
        """Initialize the custom head.

        Only reached when a model is built from scratch; `from_pretrained` runs
        under `no_init_weights` and skips this entirely.
        """
        if isinstance(module, nn.Linear):
            if module in (self.q_proj, self.k_proj, self.v_proj):
                nn.init.normal_(module.weight, std=0.02)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def validate_lang_ids(self, lang_ids: torch.Tensor) -> torch.Tensor:
        """Coerce to long, then clamp anything outside 0..6 into range."""
        if not isinstance(lang_ids, torch.Tensor):
            lang_ids = torch.tensor(lang_ids, dtype=torch.long, device=self.valid_lang_ids.device)
        elif lang_ids.dtype != torch.long:
            lang_ids = lang_ids.long()

        num_invalid = int((~torch.isin(lang_ids, self.valid_lang_ids)).sum().item())
        if num_invalid:
            logger.warning(
                "Found %d language ids outside 0-%d; clamping into range.",
                num_invalid,
                len(self.config.languages) - 1,
            )
            lang_ids = torch.clamp(lang_ids, min=0, max=len(self.config.languages) - 1)
        return lang_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        device = input_ids.device
        batch_size = input_ids.size(0)

        if lang_ids is None:
            global _WARNED_MISSING_LANG_IDS
            if not _WARNED_MISSING_LANG_IDS:
                _WARNED_MISSING_LANG_IDS = True
                warnings.warn(
                    "lang_ids was not supplied, so every sequence is being treated as "
                    "English (id 0). Pass lang_ids for non-English text; see "
                    "LANGUAGE_IDS in modeling_toxic_xlmr.py.",
                    stacklevel=2,
                )
            lang_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        lang_ids = self.validate_lang_ids(lang_ids.to(device))

        hidden_states = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        if self.needs_projection:
            hidden_states = self.dim_projection(hidden_states)

        batch_size, seq_len, hidden_size = hidden_states.shape
        num_heads = self.num_attention_heads
        head_dim = hidden_size // num_heads

        q = self.q_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Language bias. The language vector is added to the QUERIES, which
        # contributes lang_vec . k_j to the score for key j. That varies along
        # the key axis, so softmax does not cancel it. Adding a per-query
        # constant instead (the shape this model shipped with in April 2025)
        # cancels exactly and makes lang_ids a no-op.
        if not self.disable_lang_conditioning:
            lang_emb = self.lang_embed(lang_ids)
            lang_vec = self.lang_proj(lang_emb).view(batch_size, num_heads, 1, head_dim)
            attn_scores = attn_scores + torch.matmul(lang_vec, k.transpose(-2, -1)) * self.scale

        # Mask after the bias, so the language term can never leak attention
        # onto padding positions.
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.bool().unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )

        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        attention_output = torch.matmul(attn_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )

        output = self.post_attention(attention_output)
        logits = self.classifier(output[:, 0])  # [CLS] position
        probabilities = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        if return_dict is False:
            return tuple(x for x in (loss, logits, probabilities) if x is not None)
        return ToxicCommentOutput(loss=loss, logits=logits, probabilities=probabilities)


# --------------------------------------------------------------------------
# Explicit loader: no auto classes, no trust_remote_code, no magic.
# --------------------------------------------------------------------------

def load_model(model_dir: str, device: str = "cpu", tokenizer_name: str = None):
    """Load the model and its tokenizer from a local directory.

    Args:
        model_dir: directory holding config.json and pytorch_model.bin.
        device: torch device string.
        tokenizer_name: where to load the tokenizer from. Defaults to
            `model_dir` if it has tokenizer files, otherwise "xlm-roberta-large"
            (the tokenizer is stock and unmodified).

    Returns:
        (model, tokenizer), model in eval mode on `device`.
    """
    from transformers import AutoTokenizer

    with open(os.path.join(model_dir, "config.json")) as f:
        config = ToxicCommentConfig.from_dict(json.load(f))

    model = ToxicCommentModel(config)

    weights = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"No pytorch_model.bin in {model_dir}")
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:  # strict=True already raises, this is belt and braces
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")

    model.to(device)
    model.eval()

    if tokenizer_name is None:
        has_local = any(
            os.path.exists(os.path.join(model_dir, n))
            for n in ("tokenizer.json", "sentencepiece.bpe.model")
        )
        tokenizer_name = model_dir if has_local else "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return model, tokenizer
