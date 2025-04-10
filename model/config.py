from dataclasses import dataclass
from typing import Optional


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    num_image_tokens: int = 256
    # projection_dim: Optional[int] = None
    # projector_hidden_act: Optional[str] = None
    # vision_use_head: Optional[bool] = None


@dataclass
class GemmaConfig:
    vocab_size: int = 257216
    hidden_size: int = 2048
    intermediate_size: int = 16384
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pad_token_id: Optional[int] = 0
    num_image_tokens: int = 256


@dataclass
class PaliGemmaConfig:
    text_config: GemmaConfig
    vision_config: SiglipVisionConfig

    bos_token_id: int = 2
    eos_token_id: int = 1
    pad_token_id: int = 0
    ignore_index: int = -100
    image_token_index: int = 257152
    hidden_size: int = 2048
    projection_dim: int = 2048
    vocab_size: int = 257216
    model_type: str = "paligemma"
