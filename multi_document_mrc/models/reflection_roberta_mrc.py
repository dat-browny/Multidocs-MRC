from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaEncoder,
    RobertaPooler,
    BaseModelOutputWithPoolingAndCrossAttentions,
    QuestionAnsweringModelOutput
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import logging

logger = logging.getLogger(__name__)

class RobertaForMRCReflection(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        #add S, E - learned parameter according to start/end
        self.S = nn.parameter.Parameter(torch.rand(config.hidden_size, requires_grad=True))
        self.E = nn.parameter.Parameter(torch.rand(config.hidden_size, requires_grad=True))

        #add softmax function
        self.softmax = nn.Softmax(dim=1)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_answer_predictor = nn.Linear(config.hidden_size, 2)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        has_answer_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        start_logits = torch.matmul(sequence_output, self.S).contiguous()
        end_logits = torch.matmul(sequence_output, self.E).contiguous()

        start_probs = self.softmax(start_logits)
        end_probs = self.softmax(end_logits)

        has_answer_logits = self.has_answer_predictor(sequence_output[:, 0, :])
        has_answer_logits = has_answer_logits.squeeze(-1).contiguous()
        has_answer_probs = self.softmax(has_answer_logits)

        ha_loss = None
        extractive_loss = None

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = sum([-torch.log(start_probs[i][ground_truth_start]) for i, ground_truth_start in enumerate(start_positions)])
            end_loss = sum([-torch.log(end_probs[i][ground_truth_end]) for i, ground_truth_end in enumerate(end_positions)])
            extractive_loss = (start_loss + end_loss) 

            score = torch.tensor([torch.matmul(self.S, sequence_output[i][start_positions[i]]) 
                    + torch.matmul(self.E, sequence_output[i][end_positions[i]]) 
                    - torch.matmul(self.S, sequence_output[i][0]) 
                    - torch.matmul(self.E, sequence_output[i][0]) 
                    for i in range(len(start_positions))], device=input_ids.device).unsqueeze(-1).contiguous()
        else: 
            start_index = [torch.argmax(i) for i in start_probs]
            end_index = [torch.argmax(i) for i in end_probs]

            score = torch.tensor([torch.matmul(self.S, sequence_output[i][start_index[i]]) 
                    + torch.matmul(self.E, sequence_output[i][end_index[i]]) 
                    - torch.matmul(self.S, sequence_output[i][0]) 
                    - torch.matmul(self.E, sequence_output[i][0]) 
                    for i in range(len(start_index))], device=input_ids.device).unsqueeze(-1).contiguous()
                

        if has_answer_labels is not None:
            if len(has_answer_labels.size()) > 1:
                has_answer_labels = has_answer_labels.squeeze(-1)
            ha_loss = sum([-torch.log(has_answer_probs[i][ground_truth_label]) for i, ground_truth_label in enumerate(has_answer_labels)])

        if ha_loss is not None and extractive_loss is not None:
            total_loss = ha_loss + extractive_loss
        else:
            total_loss = None
        
        def normalize(batch_tensor):
            batch_tensor -= batch_tensor.min(1, keepdim=True)[0]
            batch_tensor /= batch_tensor.max(1, keepdim=True)[0]
            batch_tensor = torch.sqrt(batch_tensor)
            means = batch_tensor.mean(dim=1, keepdim=True)
            stds = batch_tensor.std(dim=1, keepdim=True)
            normalized_data = (batch_tensor - means) / stds
            return normalized_data

        ans_type_predicted = [torch.argmax(prob) for prob in has_answer_probs]
        ans_type = torch.tensor([[1,0] if type==0 else [0,1] for type in ans_type_predicted], device=input_ids.device)
        ans_type_probs = has_answer_probs
        ans_type_prob = torch.tensor([[has_answer_probs[i,j]] for i, j in enumerate(ans_type_predicted)], device=input_ids.device)
        start_logits_top = normalize(start_logits.sort(descending=True)[0][:,:5])
        # end_logits_before = end_logits.sort(descending=True)[0][:,:5]
        end_logits_top = normalize(end_logits.sort(descending=True)[0][:,:5])
        start_probs_top = normalize(start_probs.sort(descending=True)[0][:,:5])
        end_probs_top = normalize(end_probs.sort(descending=True)[0][:,:5])

        head_features = torch.cat((score, ans_type, ans_type_probs, ans_type_prob, start_logits_top, end_logits_top, start_probs_top, end_probs_top), 1)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) #if total_loss is not None else output

        return MRCReflectionModelOutput(
            loss=total_loss,
            extractive_loss=extractive_loss,
            ha_loss=ha_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_answer_probs=has_answer_probs,
            score=score,
            head_features=head_features,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class MRCReflectionModelOutput(QuestionAnsweringModelOutput):
    loss: Optional[torch.FloatTensor] = None
    extractive_loss: Optional[torch.FloatTensor] = None
    ha_loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    has_answer_probs: torch.FloatTensor = None
    score: torch.FloatTensor = None
    head_features: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ReflectEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.ans_type_embeddings = nn.Embedding(5, config.hidden_size, padding_idx =0)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, ans_type_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        ans_type_embeddings = self.ans_type_embeddings(ans_type_ids)

        embeddings = inputs_embeds + token_type_embeddings + ans_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ReflectionModel(RobertaModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.num_head_features = 26
        self.concat_dim = self.num_head_features + self.config.hidden_size
        self.embeddings = ReflectEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        
        #Freeze layer parameter according to reference of paper
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.pooler.parameters():
            param.requires_grad = False        

        self.linear = nn.Linear(self.concat_dim, self.config.hidden_size)
        self.gelu = nn.GELU()
        self.A = nn.parameter.Parameter(torch.rand(self.config.hidden_size, requires_grad=True))
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss(reduction="sum")
        
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,

        ans_type_ids: torch.Tensor = None,

        head_features: Optional[torch.Tensor] = None,

        position_ids: Optional[torch.Tensor] = None,

        has_answer_labels: Optional[torch.FloatTensor] = None, 

        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            ans_type_ids=ans_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        features = torch.cat((sequence_output[:,0], head_features), 1)
        hidden_x = self.gelu(self.linear(features)) 

        ans_type_probs = self.sigmoid(torch.matmul(hidden_x, self.A))

        if has_answer_labels is not None:
            has_answer_labels = has_answer_labels.float()
            loss = self.bce(ans_type_probs, has_answer_labels)
        else: 
            loss = None

        if not return_dict:
            output = (ans_type_probs) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ReflectionModelOutput(
            loss=loss,
            ans_type_probs=ans_type_probs,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@dataclass
class ReflectionModelOutput(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    ans_type_probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class RobertaForMRCClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None
            else config.hidden_dropout_prob)

        self.num_hidden_states = 4
        self.dropout = nn.Dropout(classifier_dropout)
        if self.config.output_hidden_states:
            self.classifier = nn.Linear(config.hidden_size*self.num_hidden_states, config.num_labels)
            self.dense = nn.Linear(config.hidden_size*self.num_hidden_states, config.hidden_size*self.num_hidden_states)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            return_dict=return_dict,
            output_hidden_states=self.config.output_hidden_states,
        )

        if self.config.output_hidden_states:
            hidden_states = outputs.hidden_states
            print("================================================")
            print(len(outputs))
            for i in range(len(hidden_states)):
                print(hidden_states[i].shape)
            print("==============================")
            x = torch.cat(tuple([hidden_states[i] for i in [-self.num_hidden_states, 0, -1]]), dim=-1)
        else:
        #sequence output == last hidden states of RoBERTa model .
            sequence_output = outputs[0]
            x = sequence_output[:, 0, :] #take <s> token of a batch => dim [batch_size, 1, num_hidden_state]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
