import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import SiglipVisionConfig


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # TODO
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_value: torch.FloatTensor) -> torch.Tensor:
        B, C, H, W = pixel_value.shape

        patch_embeds = self.patch_embedding(
            pixel_value
        )  # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # (B, NUM_PATCHES, EMBED_DIM)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Add position embedding
        embeddings = patch_embeds + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert (
            self.embed_dim % self.num_heads == 0
        ), f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"

        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5

        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = hidden_states.shape

        q, k, v = map(
            lambda x: x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2),
            (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ),
        )

        score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        score = score.softmax(dim=-1)

        score = F.dropout(score, p=self.dropout, training=self.training)
        attn = torch.matmul(score, v).transpose(1, 2).contiguous().view(B, S, D)
        attn = self.out_proj(attn)

        return attn, score


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn, _ = self.self_attn(hidden_states)
        hidden_states = attn + residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = mlp_out + residual

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)

        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)

        return hidden_states


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_transformer = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> torch.Tensor:
        hidden_states = self.vision_transformer(pixel_values)

        # [B, NUM_PATCHES, EMBED_DIM]
        return hidden_states
