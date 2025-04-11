from .config import PaliGemmaConfig
from .siglip import SiglipVisionModel
from .gemma import GemmaForCausalLM
from .kv_cache import KVCache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()

        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: KVCache,
    ):
        _, _, embed_dim = image_features.shape
        B, S = input_ids.shape  # S is the size of the image plus text
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # Scale the image features
        # The original paper scales the image features by sqrt(d_k)
        scaled_image_features = image_features / (self.config.hidden_size**0.5)

        # ===== Get the final embedding of the image and text tokens =====
        final_embedding = torch.zeros(B, S, embed_dim, dtype=dtype, device=device)
        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # (B, S, embed_dim)
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding
        )
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        final_embedding = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
        )

        # ===== Get the attention mask =====
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is not None or kv_cache.num_items() == 0:
            causal_mask = torch.full(
                (B, q_len, q_len), fill_value=0, device=device, dtype=dtype
            )

        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (B, q_len, kv_len), fill_value=0, device=device, dtype=dtype
            )

        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: KVCache,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the image features and text embeddings
        inputs_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features=image_features,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                input_ids=input_ids,
                kv_cache=kv_cache,
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
