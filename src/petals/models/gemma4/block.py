from typing import Optional, Tuple

import torch
from transformers import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

from petals.utils.cache import SingleLayerCache


class WrappedGemma4Block(Gemma4TextDecoderLayer):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
        text_config = getattr(config, "text_config", config)
        self.rotary_emb = Gemma4TextRotaryEmbedding(text_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        per_layer_input: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
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
        elif use_cache:
            past_key_value = SingleLayerCache(
                self.layer_idx,
                torch.empty(0),
                torch.empty(0),
                0
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )

        from transformers.models.gemma4.modeling_gemma4 import create_causal_mask, create_sliding_window_causal_mask
        mask_kwargs = {
            "config": getattr(self.config, "text_config", self.config),
            "inputs_embeds": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_value,
            "position_ids": position_ids,
        }
        layer_type = self.config.layer_types[self.layer_idx]
        if layer_type == "sliding_attention":
            attention_mask = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            attention_mask = create_causal_mask(**mask_kwargs)

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=hidden_states.device
            )
            position_ids = position_ids.unsqueeze(0).reshape(-1, seq_length)

        if position_embeddings is None:
            layer_type = self.config.layer_types[self.layer_idx]
            position_embeddings = self.rotary_emb(hidden_states, position_ids, layer_type)

        if per_layer_input is None and self.hidden_size_per_layer_input:
            per_layer_input = torch.ones(
                (batch_size, seq_length, self.hidden_size_per_layer_input),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

        outputs = super().forward(
            hidden_states,
            per_layer_input=per_layer_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            **kwargs
        )

        hidden_states = outputs

        if use_cache:
            present_key_value = (past_key_value.key_cache[self.layer_idx], past_key_value.value_cache[self.layer_idx])
            present_key_value = self._reorder_cache_to_bloom(present_key_value, batch_size, seq_length_with_past)
            outputs = (hidden_states, present_key_value)
        else:
            outputs = (hidden_states,)

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
