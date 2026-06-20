import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.gemma2.block import WrappedGemma2Block

logger = get_logger(__name__)


class DistributedGemma2Config(Gemma2Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedGemma2Block
    attn_class = Gemma2Attention
    block_prefix = "model.layers"

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
