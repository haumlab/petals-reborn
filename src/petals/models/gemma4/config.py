import os
from typing import Optional, Union

from hivemind import get_logger
from transformers import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.gemma4.block import WrappedGemma4Block

logger = get_logger(__name__)


class DistributedGemma4Config(Gemma4Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedGemma4Block
    attn_class = Gemma4TextAttention
    block_prefix = "model.language_model.layers"

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            text_config = getattr(self, "text_config", None)
            if text_config is not None and hasattr(text_config, name):
                return getattr(text_config, name)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.pretraining_tp = 1
        config.use_cache = True
        return result
