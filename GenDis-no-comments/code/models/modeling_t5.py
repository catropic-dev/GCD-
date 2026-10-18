pass
import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput, Seq2SeqQuestionAnsweringModelOutput, Seq2SeqSequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings, add_start_docstrings_to_model_forward, is_torch_fx_proxy, is_torchdynamo_compiling, logging, replace_return_docstrings
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from .cca_model import CCAProjection
from .sdl_loss import SoftCCALoss
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'T5Config'
_CHECKPOINT_FOR_DOC = 'google-t5/t5-small'
from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5_START_DOCSTRING, T5Stack, PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING, T5_INPUTS_DOCSTRING, __HEAD_MASK_WARNING_MSG
from dataclasses import dataclass
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('../pretrained_models/t5-base')
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

@add_start_docstrings('T5 Model with a `language modeling` head on top.', T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_unexpected = ['decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight']
    _tied_weights_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'lm_head.weight']

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()
        self.model_parallel = False
        self.device_map = None
        self.hidden_size = config.d_model

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn("`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0': 0, 'encoder.block.1': 1, ...}", FutureWarning)
        self.device_map = get_device_map(len(self.encoder.block), range(torch.cuda.device_count())) if device_map is None else device_map
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn('Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.', FutureWarning)
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to('cpu')
        self.decoder = self.decoder.to('cpu')
        self.lm_head = self.lm_head.to('cpu')
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def set_class(self, num_labels, cca_k=10):
        self.class_mlp = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.score_dis = nn.Linear(self.hidden_size, num_labels, bias=True)
        self.score_com = nn.Linear(2 * cca_k, num_labels, bias=True)
        self.cca_k = cca_k
        self.cca_proj = CCAProjection(self.hidden_size, self.hidden_size, cca_k)
        self.soft_cca = SoftCCALoss(self.cca_k)
        self.class_mlp.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.score_dis.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.score_com.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.cca_proj.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.soft_cca.to(device=self.device, dtype=next(self.parameters()).dtype)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, cache_position: Optional[torch.LongTensor]=None, **kwargs) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        pass
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        if 'token_type_ids' in kwargs:
            input_ids[kwargs['token_type_ids'] > 1] = tok.pad_token_id
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, cache_position=cache_position)
        sequence_output = decoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.model_dim ** (-0.5)
        lm_logits = self.lm_head(sequence_output)
        if 'token_type_ids' not in kwargs:
            return Seq2SeqLMOutput(logits=lm_logits, past_key_values=decoder_outputs.past_key_values)
        outputs = self.compute_loss(input_ids, decoder_input_ids, lm_logits, sequence_output, kwargs)
        loss = outputs['loss']
        return CausalLMOutputWithPast(loss=outputs['loss'], logits=outputs['logits'], past_key_values=decoder_outputs.past_key_values, class_hidden_states=outputs['class_hidden_states'], mlp_hidden_states=outputs['mlp_hidden_states'], com_hidden_states=outputs['com_hidden_states'], token_type_ids=kwargs['token_type_ids'], class_golds=kwargs['class_golds'])

    def compute_loss(self, input_ids, decoder_input_ids, logits, hidden_states, kwargs):
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
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits.contiguous()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        if kwargs['gen_loss_weight'][0].item() != 0:
            gen_labels = input_ids.clone().view(-1)
            loss_gen = loss_fct(shift_logits, gen_labels)
        if kwargs['class_loss_weight'][0].item() != 0:
            if kwargs['class_golds'][kwargs['class_golds'] != -1].shape[0] > 0:
                shift_class_logits = logits[kwargs['class_golds'] != -1, :, :].contiguous()
                shift_class_logits = shift_class_logits.view(-1, self.config.vocab_size)
                gen_labels = input_ids.clone()
                gen_labels[kwargs['token_type_ids'] <= 0] = -100
                gen_labels = gen_labels[kwargs['class_golds'] != -1].view(-1)
                loss_class = loss_fct(shift_class_logits, gen_labels)
        if kwargs['class_pseudo_loss_weight'][0].item() != 0:
            if kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1].shape[0] > 0:
                shift_class_pseudo_logits = logits[kwargs['pseudo_golds'] != -1, :, :].contiguous()
                shift_class_pseudo_logits = shift_class_pseudo_logits.view(-1, self.config.vocab_size)
                gen_labels = input_ids.clone()
                gen_labels[kwargs['token_type_ids'] <= 0] = -100
                gen_labels = gen_labels[kwargs['pseudo_golds'] != -1].view(-1)
                loss_class_pseudo = loss_fct(shift_class_pseudo_logits, gen_labels)
        shift_hidden_states = hidden_states.contiguous()
        class_hidden_states, mlp_hidden_states, com_hidden_states = (None, None, None)
        class_hidden_states = shift_hidden_states[kwargs['token_type_ids'] == 1]
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
            if kwargs['cca_loss_weight'][0].item() != 0:
                if kwargs['cca_loss_func'][0].item() <= 2:
                    loss_cca = self.cca_proj.calculate_loss(z1, z2, mode=kwargs['cca_loss_func'][0].item())
                else:
                    loss_cca = self.soft_cca(z1, z2)
            if kwargs['com_loss_weight'][0].item() != 0:
                if kwargs['class_golds'][kwargs['class_golds'] != -1].shape[0] > 0:
                    loss_com = loss_fct(com_class_logits[kwargs['class_golds'] != -1], kwargs['class_golds'][kwargs['class_golds'] != -1])
            if kwargs['com_pseudo_loss_weight'][0].item() != 0:
                if kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1].shape[0] > 0:
                    loss_com_pseudo = loss_fct(com_class_logits[kwargs['pseudo_golds'] != -1], kwargs['pseudo_golds'][kwargs['pseudo_golds'] != -1])
        loss = loss_gen * kwargs['gen_loss_weight'][0].item() + loss_class * kwargs['class_loss_weight'][0].item() + loss_dis * kwargs['dis_loss_weight'][0].item() + loss_com * kwargs['com_loss_weight'][0].item() + loss_cca * kwargs['cca_loss_weight'][0].item() + loss_class_pseudo * kwargs['class_pseudo_loss_weight'][0].item() + loss_dis_pseudo * kwargs['dis_pseudo_loss_weight'][0].item() + loss_com_pseudo * kwargs['com_pseudo_loss_weight'][0].item()
        outputs = {'loss': loss, 'logits': logits, 'mlp_class_logits': mlp_class_logits, 'com_class_logits': com_class_logits, 'class_hidden_states': class_hidden_states, 'mlp_hidden_states': mlp_hidden_states, 'com_hidden_states': com_hidden_states}
        return outputs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            logger.warning('You might want to consider setting `use_cache=True` to speed up decoding')
            return past_key_values
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),)
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(f'reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched')
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(f'length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched')
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
