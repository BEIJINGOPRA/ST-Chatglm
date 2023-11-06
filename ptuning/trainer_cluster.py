from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers.deepspeed import is_deepspeed_zero3_enabled
from trainer import PrefixTrainer
from blip_trainer import BlipTrainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class clusterTrainer(BlipTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset]=None,
        ignore_keys:Optional[List[str]]=None,
        metric_key_prefix: str = "eval",
        **gen_kwargs
    )->Dict[str, float]:
        '''
        run evaluation and return metric

        这里的评价metric标准，是分类准确率，送入一组 text 以及 ST-embedding，输出两个指标：
        1. 首先是聚类的准确率，这个通过sim_matrix+softmax选最大可能性
        2. 其次是feature相似性的准确率，这个通过sim_matrix计算相似性排序的label，再根据排序RC来衡量排序的准确率
        '''
        
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs  
        
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().predict(test_dataset=test_dataset,ignore_keys=ignore_keys,metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    )->Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        '''
        重写model的prediction的单步
        inputs表示的是模型接收的输入和模型的输出目标

        返回的是 loss，logits，labels
        '''
        if not selg.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only
            )
        
        # ##################### 在这里将输入数据放在GPU上 ###########################
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False

        # ######################## 生成任务参数设定 ###########################
        # ######################## 第一阶段训练用不到 gen_kwargs ##############
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # ##################### 这里的attention_mask用于ChatGLM ###########################
        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "position_ids" in inputs:
            gen_kwargs["position_ids"] = inputs.get("position_ids", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get("global_attention_mask", None)

        # ##################### 在这里添加Blip模型的attention mask #########################

        # ###############################################################################

        gen_kwargs["input_ids"] = inputs
        
        logger.info("You haven't complite the prediction step, Please finish this function!")
        
        assert 0 == 1


    def _pad_tensors_to_max_len(self, tensor, max_length):
        if selg.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else tokenizer.cos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token id haven't been set yet, please check your configuration")
        
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        outputs = model(**inputs)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        return (loss, outputs) if return_outputs else loss
     

