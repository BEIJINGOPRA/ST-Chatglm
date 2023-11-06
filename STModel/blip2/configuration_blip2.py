import copy

from transformers import PretrainedConfig
from .myblip2_qformer import Blip2QFormer4STModel
from .myblip2 import Blip2Model4ST
from typing import Union, Dict, Any, Optional
import os

import logging

logger = logging.getLogger(__name__)

class Blip2QFormer4STConfig(PretrainedConfig):
    model_type = "blip2_qformer"
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        classifier_dropout=None,
        cross_attention_frequency=1,
        st_encoder_hidden_size=128,
        text_encoder_hidden_size=4096,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.classifier_dropout = classifier_dropout
        
        self.cross_attention_frequency = cross_attention_frequency
        
        self.st_encoder_hidden_size = st_encoder_hidden_size
        self.text_encoder_hidden_size = text_encoder_hidden_size

class Blip2STConfig(PretrainedConfig):
    model_type = "blip-2-ST"
    is_composition = True

    def __init__(self, qformer_config=None, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        self.qformer_config = Blip2QFormer4STConfig(**qformer_config)

        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.qformer_config.hidden_size
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["qformer_config"] = self.qformer_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
