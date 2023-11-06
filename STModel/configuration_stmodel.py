import copy
import logging
from transformers import PretrainedConfig
from typing import Union, Dict, Any, Optional
import os

from blip2.configuration_blip2 import Blip2STConfig
from chatglm2_6b.configuration_chatglm import ChatGLMConfig

logger = logging.getLogger(__name__)


class ChatGLM4STConfig(PretrainedConfig):
    model_type = "stmodel"
    # def __init__(self, chatglmconfig=None, blip2config=None, **kwargs):
    def __init__(self, max_source_length=64, transformers_version="4.30.2", troch_dtype="float16", **kwargs): 
        self.ChatGLMConfig = None
        self.Blip2Config = None

        self.max_source_length = max_source_length
        self.transformers_version = transformers_version
        self.torch_dtype = troch_dtype
        super().__init__(**kwargs)

        '''
        if chatglmconfig is not None:
            self.ChatGLMConfig = ChatGLMConfig(**chatglmconfig)
        if blip2config is not None:
            blip2qformer4stconfig = blip2config['Blip2QFormer4STConfig'] 
            self.Blip2Config = Blip2STConfig(qformer_config=blip2qformer4stconfig, num_query_tokens=blip2config['query_token_nums'])
        '''

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        if self.ChatGLMConfig is not None:
            output['ChatGLMConfig'] = self.ChatGLMConfig.to_dict()
        else:
            output['ChatGLMConfig'] = None
        if self.Blip2Config is not None:
            output['Blip2Config'] = self.Blip2Config.to_dict()
        else:
            output['Blip2Config'] = None
        output['model_type'] = self.__class__.model_type
        return output
    
    def set_chatglm_config(self, chatglm_config):
        self.ChatGLMConfig = chatglm_config
    
    def set_blip2_config(self, blip2_config):
        self.Blip2Config = blip2_config
    
    '''
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path : Union[str, os.PathLike], **kwargs):
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if 'ChatGLMConfig' not in config_dict.keys():
            return cls(
                None,
                None,
                **kwargs
            )

        chatglm_config = config_dict['ChatGLMConfig']
        blip2_config = config_dict['Blip2Config']

        return cls(
            chatglm_config,
            blip2_config,
            **kwargs,
        )
    '''