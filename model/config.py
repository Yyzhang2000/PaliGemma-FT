from dataclasses import dataclass


@dataclass
class SiglipVisionConfig:
    hidden_size = 768
    intermediate_size = 3072
    num_hidden_layers = 12
    num_attention_heads = 12
    num_channels = 3
    image_size = 224
    patch_size = 16
    layer_norm_eps = 1e-6
    attention_dropout = 0.0
    num_image_tokens: int = 8


@dataclass
class GemmaConfig:
    vocab_size = 50265
    hidden_size = 768
    intermediate_size = 3072
    num_hidden_layers = 12
    num_attention_heads = 12
    num_key_value_heads = 12
    head_dim = 256
    max_position_embeddings = 8192
    rms_norm_eps = 1e-6
    rope_theta = 10000.0
    attention_bias = False
    attention_dropout = 0.0
    pad_token_id = None


@dataclass
class PaliGemmaConfig:
    text_config = GemmaConfig
    vision_config = SiglipVisionConfig

    ignore_index = ignore_index
    image_token_index = image_token_index
    vocab_size = vocab_size
    projection_dim = projection_dim
    hidden_size = hidden_size
    vision_config = vision_config
    is_encoder_decoder = False
    pad_token_id = pad_token_id

    text_config = text_config

    vocab_size = text_config.vocab_size
    text_config.num_image_tokens = (
        vision_config.image_size // vision_config.patch_size
    ) ** 2
    vision_config.projection_dim = projection_dim
