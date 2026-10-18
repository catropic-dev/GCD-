from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from enum import Enum
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from dataclasses import dataclass
from transformers.processing_utils import Unpack
from transformers.utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .cca_model import CCAProjection
from .sdl_loss import SoftCCALoss
from .utils import triplet_ranking_loss
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
logger = logging.get_logger(__name__)
_CHECKPOINT_FOR_DOC = 'meta-qwen2/Qwen2-2-7b-hf'
_CONFIG_FOR_DOC = 'Qwen2Config'
from transformers import Qwen2PreTrainedModel, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, KwargsForCausalLM, can_return_tuple, auto_docstring
from transformers.utils import ModelOutput

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mlp_class_logits: torch.FloatTensor = None
    com_class_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    class_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    mlp_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    com_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_type_ids: Optional[Tuple[torch.FloatTensor, ...]] = None
    class_golds: Optional[Tuple[torch.FloatTensor, ...]] = None
DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"

class ChatmlSpecialTokens(str, Enum):
    user = '<|im_start|>user'
    assistant = '<|im_start|>assistant'
    system = '<|im_start|>system'
    eos_token = '<|im_end|>'
    bos_token = '<s>'
    pad_token = '<pad>'

    @classmethod
    def list(cls):
        return [c.value for c in cls]
special_tokens = ChatmlSpecialTokens
chat_template = DEFAULT_CHATML_CHAT_TEMPLATE

class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ['lm_head.weight']
    _tp_plan = {'lm_head': 'colwise_rep'}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.hidden_size = config.hidden_size
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, num_logits_to_keep: int=0, **kwargs: Unpack[KwargsForCausalLM]) -> Union[Tuple, CausalLMOutputWithPast]:
        pass
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        if 'token_type_ids' not in kwargs:
            return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)
        final_outputs = self.compute_loss(input_ids, logits, hidden_states, kwargs)
        loss = final_outputs['loss']
        return CausalLMOutputWithPast(loss=loss, logits=final_outputs['logits'], mlp_class_logits=final_outputs['mlp_class_logits'], com_class_logits=final_outputs['com_class_logits'], past_key_values=outputs.past_key_values, class_hidden_states=final_outputs['class_hidden_states'], mlp_hidden_states=final_outputs['mlp_hidden_states'], com_hidden_states=final_outputs['com_hidden_states'], token_type_ids=kwargs['token_type_ids'], class_golds=kwargs['class_golds'])

    def set_class(self, num_labels, cca_k=10):
        self.class_mlp = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.hyper_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.score_dis = nn.Linear(self.hidden_size, num_labels, bias=True)
        self.score_com = nn.Linear(2 * cca_k, num_labels, bias=True)
        self.cca_k = cca_k
        self.cca_proj = CCAProjection(self.hidden_size, self.hidden_size, cca_k)
        self.soft_cca = SoftCCALoss(self.cca_k)
        self.soft_cca.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.hyper_proj.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.class_mlp.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.score_dis.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.score_com.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.cca_proj.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.add_module('class_mlp', self.class_mlp)
        self.add_module('hyper_proj', self.hyper_proj)
        self.add_module('score_dis', self.score_dis)
        self.add_module('score_com', self.score_com)
        self.add_module('cca_proj', self.cca_proj)
        self.add_module('soft_cca', self.soft_cca)

    def compute_loss(self, input_ids, logits, hidden_states, kwargs):
        loss = None
        mlp_class_logits = None
        com_class_logits = None
        loss_kwargs = {'dtype': hidden_states.dtype, 'device': hidden_states.device, 'requires_grad': True}
        loss_gen = torch.tensor(0.0, **loss_kwargs)
        loss_dis = torch.tensor(0.0, **loss_kwargs)
        loss_com = torch.tensor(0.0, **loss_kwargs)
        loss_class = torch.tensor(0.0, **loss_kwargs)
        loss_dis_pseudo = torch.tensor(0.0, **loss_kwargs)
        loss_com_pseudo = torch.tensor(0.0, **loss_kwargs)
        loss_class_pseudo = torch.tensor(0.0, **loss_kwargs)
        loss_cca = torch.tensor(0.0, **loss_kwargs)
        loss_triplet_ranking_loss = torch.tensor(0.0, **loss_kwargs)
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        if kwargs['gen_loss_weight'][0].item() != 0:
            gen_labels = input_ids.clone()
            gen_labels[kwargs['token_type_ids'] <= 0] = -100
            shift_gen_labels = gen_labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss_gen = loss_fct(shift_logits, shift_gen_labels)
        if kwargs['class_loss_weight'][0].item() != 0:
            if kwargs['class_golds'][kwargs['class_golds'] != -1].shape[0] > 0:
                shift_class_logits = logits[kwargs['class_golds'] != -1, :-1, :].contiguous()
                shift_class_logits = shift_class_logits.view(-1, self.config.vocab_size)
                gen_labels = input_ids.clone()
                gen_labels[kwargs['token_type_ids'] <= 2] = -100
                gen_labels = gen_labels[kwargs['class_golds'] != -1]
                shift_gen_labels = gen_labels[..., 1:].contiguous().view(-1).to(shift_class_logits.device)
                loss_class = loss_fct(shift_class_logits, shift_gen_labels)
        if kwargs['class_pseudo_loss_weight'][0].item() != 0:
            shift_class_pseudo_logits = logits[kwargs['class_golds'] == -1, :-1, :].contiguous()
            shift_class_pseudo_logits = shift_class_pseudo_logits.view(-1, self.config.vocab_size)
            gen_labels = input_ids.clone()
            gen_labels[kwargs['token_type_ids'] <= 2] = -100
            gen_labels = gen_labels[kwargs['class_golds'] == -1]
            shift_gen_labels = gen_labels[..., 1:].contiguous().view(-1).to(shift_class_pseudo_logits.device)
            if len(shift_gen_labels[shift_gen_labels != -100]) > 0:
                loss_class_pseudo = loss_fct(shift_class_pseudo_logits, shift_gen_labels)
        class_hidden_states, mlp_hidden_states, com_hidden_states = (None, None, None)
        if (kwargs['token_type_ids'] == 2).sum() != 0:
            class_hidden_states = hidden_states[kwargs['token_type_ids'] == 2]
        if kwargs['is_mlp'][0].item() == 0:
            class_logits = self.score_dis(class_hidden_states)
            if kwargs['dis_loss_weight'][0].item() != 0:
                if kwargs['class_golds'][kwargs['class_golds'] != -1].shape[0] > 0:
                    loss_dis = loss_fct(class_logits[kwargs['class_golds'] != -1], kwargs['class_golds'][kwargs['class_golds'] != -1])
            if kwargs['dis_pseudo_loss_weight'][0].item() != 0:
                if kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1].shape[0] > 0:
                    loss_dis_pseudo = loss_fct(class_logits[kwargs['pseudo_golds'] != -1], kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1])
            mlp_hidden_states = class_hidden_states
            com_hidden_states = class_hidden_states
            mlp_class_logits = self.score_dis(mlp_hidden_states)
            com_class_logits = mlp_class_logits
        elif kwargs['is_mlp'][0].item() >= 1:
            mlp_hidden_states = self.class_mlp(class_hidden_states)
            mlp_class_logits = self.score_dis(mlp_hidden_states)
            assert kwargs['dis_loss_weight'][0].item() != 0
            if kwargs['class_golds'][kwargs['class_golds'] != -1].shape[0] > 0:
                loss_dis = loss_fct(mlp_class_logits[kwargs['class_golds'] != -1], kwargs['class_golds'][kwargs['class_golds'] != -1])
            if kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1].shape[0] > 0:
                loss_dis_pseudo = loss_fct(mlp_class_logits[kwargs['pseudo_golds'] != -1], kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1])
            z1, z2, com_hidden_states = self.cca_proj(class_hidden_states, mlp_hidden_states)
            com_class_logits = self.score_com(com_hidden_states)
            if self.training:
                if kwargs['com_loss_weight'][0].item() != 0:
                    if kwargs['class_golds'][kwargs['class_golds'] != -1].shape[0] > 0:
                        loss_com = loss_fct(com_class_logits[kwargs['class_golds'] != -1], kwargs['class_golds'][kwargs['class_golds'] != -1])
                if kwargs['cca_loss_weight'][0].item() != 0:
                    mask = (kwargs['class_golds'] != -1) | (kwargs['pseudo_golds'] != -1)
                    z1 = z1[mask]
                    z2 = z2[mask]
                    if mask.sum() > 4:
                        if kwargs['cca_loss_func'][0].item() <= 2:
                            loss_cca = self.cca_proj.calculate_loss(z1, z2, mode=kwargs['cca_loss_func'][0].item())
                        else:
                            loss_cca = self.soft_cca(z1, z2)
                if kwargs['com_pseudo_loss_weight'][0].item() != 0:
                    if kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1].shape[0] > 0:
                        loss_com_pseudo = loss_fct(com_class_logits[kwargs['pseudo_golds'] != -1], kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1])
        loss = loss_gen * kwargs['gen_loss_weight'][0].item() + loss_class * kwargs['class_loss_weight'][0].item() + loss_dis * kwargs['dis_loss_weight'][0].item() + loss_com * kwargs['com_loss_weight'][0].item() + loss_cca * kwargs['cca_loss_weight'][0].item() + loss_class_pseudo * kwargs['class_pseudo_loss_weight'][0].item() + loss_dis_pseudo * kwargs['dis_pseudo_loss_weight'][0].item() + loss_com_pseudo * kwargs['com_pseudo_loss_weight'][0].item()
        outputs = {'loss': loss, 'logits': logits, 'mlp_class_logits': mlp_class_logits, 'com_class_logits': com_class_logits, 'class_hidden_states': class_hidden_states, 'mlp_hidden_states': mlp_hidden_states, 'com_hidden_states': com_hidden_states}
        return outputs
