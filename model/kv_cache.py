import torch
from typing import Optional, List, Tuple


class KVCache:
    def __init__(self) -> None:
        # The Key and Value cache is a list of tensors.
        # With shape (batch_size, num_heads, seq_len, head_dim)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if len(self.key_cache) <= layer_idx:
            # The layer at layer index does not exist in the cache
            # This the pre-filling part
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def clear(self) -> None:
        # Clear the cache
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
