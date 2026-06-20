from typing import Optional, Tuple

import torch
from transformers import Gemma2Config
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

from petals.utils.cache import SingleLayerCache


class WrappedGemma2Block(Gemma2DecoderLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0
        past_key_value = layer_past

        if past_key_value is not None:
            past_key_values_length = past_key_value[0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            _past_key_value = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
            past_key_value = SingleLayerCache(self.layer_idx, _past_key_value[0], _past_key_value[1], past_key_values_length)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=hidden_states.device
        )
        cache_position = position_ids.clone()
        position_ids = position_ids.unsqueeze(0).reshape(-1, seq_length)

        outputs = super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )

        if use_cache:
            present_key_value = outputs[-1]
            if hasattr(present_key_value, "key_cache"):
                present_key_value = (present_key_value.key_cache[self.layer_idx], present_key_value.value_cache[self.layer_idx])
            else:
                present_key_value = present_key_value[self.layer_idx]
            present_key_value = self._reorder_cache_to_bloom(present_key_value, batch_size, seq_length_with_past)
            outputs = outputs[:-1] + (present_key_value,)

        return outputs

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        key_states = key_states.permute(0, 2, 1)
        key_states = key_states.reshape(
            batch_size, self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        value_states = value_states.reshape(*key_states.shape)
        return (key_states, value_states)

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int, seq_length: int
    ) -> Tuple[torch.Tensor]:
        key_states, value_states = key_value
        value_states = value_states.reshape(
            batch_size * self.self_attn.num_key_value_heads, seq_length, self.self_attn.head_dim
        )
        key_states = key_states.reshape(*value_states.shape)
        key_states = key_states.permute(0, 2, 1)
        return (key_states, value_states)
