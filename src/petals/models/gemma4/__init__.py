from petals.models.gemma4.block import WrappedGemma4Block
from petals.models.gemma4.config import DistributedGemma4Config
from petals.models.gemma4.model import (
    DistributedGemma4ForCausalLM,
    DistributedGemma4Model,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGemma4Config,
    model=DistributedGemma4Model,
    model_for_causal_lm=DistributedGemma4ForCausalLM,
)
