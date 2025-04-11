import torch
import torch.nn as nn
import torch.nn.functional as F

import math


from typing import Optional, Dict
from .config import GemmaConfig
from .kv_cache import KVCache


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())

        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(
        self, dim, max_position_embeddings=2028, base: float = 10000.0, device=None
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):

        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )

        self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, -1)
        )

        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1).float()

            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size

        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        up = self.up_proj(x)
        down = self.down_proj(gate * up)

        return down


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, kv_heads, S, D = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(B, kv_heads, n_rep, S, D)
    hidden_states = hidden_states.reshape(B, kv_heads * n_rep, S, D)
    return hidden_states


class GemmaAttention(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads

        self.is_causal = True

        assert config.hidden_size % config.num_attention_heads == 0

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )

        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs
    ):
        B, S, D = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(
            B, S, self.config.num_attention_heads, self.config.head_dim
        ).transpose(1, 2)
        k = k.view(
            B, S, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)
        v = v.view(
            B, S, self.config.num_key_value_heads, self.config.head_dim
        ).transpose(1, 2)

        # Apply rotary embedding on q, k,
        cos, sin = self.rotary_emb(v, position_ids, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Get Key and Value from cache
        if kv_cache is not None and self.layer_idx is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        # Repeat Key and Value for each group
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.config.head_dim)

        assert attention_mask is not None
        scores = scores + attention_mask

        scores = F.softmax(scores, dim=-1).to(q.dtype)
        scores = F.dropout(
            scores, p=self.config.attention_dropout, training=self.training
        )

        attention = (
            torch.matmul(scores, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, S, self.config.num_attention_heads * self.config.head_dim)
        )

        out = self.o_proj(attention)

        return out, scores


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config, layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        residual = self.input_layernorm(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        input_embeds: torch.Tensor,
        kv_cache: Optional[KVCache],
        **kwargs
    ) -> torch.FloatTensor:

        hidden_states = input_embeds
        normalizer = torch.tensor(
            self.config.hidden_size**0.5,
            dtype=hidden_states.dtype,
        )
        hidden_states = hidden_states * normalizer

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.config = config

        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Dict:

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs

        logits = self.lm_head(hidden_states).float()

        return_data = {"logits": logits}

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data
