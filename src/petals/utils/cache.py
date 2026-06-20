import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


class SingleLayerCache(DynamicCache):
    def __init__(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, seen_tokens: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layers = [None] * layer_idx
        layer = DynamicLayer()
        layer.dtype = key_states.dtype
        layer.device = key_states.device
        layer.keys = key_states
        layer.values = value_states
        layer.is_initialized = True
        self.layers.append(layer)
        self._seen_tokens = seen_tokens

    @property
    def key_cache(self):
        return [None] * self.layer_idx + [self.layers[self.layer_idx].keys if len(self.layers) > self.layer_idx and self.layers[self.layer_idx] is not None else None]

    @property
    def value_cache(self):
        return [None] * self.layer_idx + [self.layers[self.layer_idx].values if len(self.layers) > self.layer_idx and self.layers[self.layer_idx] is not None else None]
