import logging
import os
import torch
import copy

from torch import nn

from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from typing import Union, Optional, Tuple

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .myblip2_qformer import Blip2QFormer4STModel


@dataclass
class STClusterwithText(ModelOutput):
    loss:Optional[torch.FloatTensor] = None
    
    '''
    identity_loss:Optional[torch.FloatTensor] = None
    feature_loss:Optional[torch.FloatTensor] = None
    query_hidden_states: Optional[Tuple[torch.FloatTensor]] = None 
    '''
    

# 2. 承载QFormer的封装class
class Blip2Model4ST(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormer4STModel(config.qformer_config)
        self.gradient_checkpointing = True
        # 如果想让这句话生效，需要重写PreTrainedModel里面的_init_weight函数
        # 还需要 checkpointing 参数来使用checkpoint换显存
        # self.post_init()

    def forward(self,
                text_encoder_hidden_states,
                st_encoder_hidden_states,
                text_encoder_attention_mask=None, 
                st_encoder_attention_mask=None,  
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):

        batch_size = text_encoder_hidden_states.shape[0]

        # 1. 根据batch_size扩展query_tokens
        query_tokens = self.query_tokens.expand(batch_size,-1,-1)

        # 2. 将数据输入到Q-Former中，获得输出
        query_outputs = self.qformer(
            query_embeds = query_tokens,
            text_encoder_hidden_states=text_encoder_hidden_states,
            text_encoder_attention_mask=text_encoder_attention_mask, 
            st_encoder_hidden_states=st_encoder_hidden_states,
            st_encoder_attention_mask=st_encoder_attention_mask,  
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = query_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            return (sequence_output, pooled_output) + query_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=query_outputs.past_key_values,
            hidden_states=query_outputs.hidden_states,
            attentions=query_outputs.attentions,
            cross_attentions=query_outputs.cross_attentions,
        )

    