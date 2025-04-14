import torch
import torch.nn as nn
import os
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings as HFBertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

class FFTBertEmbeddings(HFBertEmbeddings):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        print(f"[INFO FFTEmb Init Rank {int(os.environ.get('LOCAL_RANK', 0))}] Initialized FFTBertEmbeddings (for pretrained).")

    def fft_circular_convolution(self, v1, v2):
        """using fft circular convolution to fuse word & position embeddings in the last dimension"""
        fft1 = torch.fft.fft(v1, dim=-1)
        fft2 = torch.fft.fft(v2, dim=-1)
        convolved_fft = fft1 * fft2
        convolved = torch.fft.ifft(convolved_fft, dim=-1).real
        return convolved

    def normalize_vector(self, v, eps=1e-9):
        """L2 normalization"""
        norm = torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)
        normalized_v = v / (norm + eps)
        return normalized_v

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
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
            inputs_embeds = self.word_embeddings(input_ids) # Shape: [B, L, D]

        position_embeddings = self.position_embeddings(position_ids) # Shape: [1, L, D] or similar
        batch_size = input_shape[0]
        pos_emb_for_seq = position_embeddings[:, :seq_length, :]
        if pos_emb_for_seq.size(0) == 1: 
             expanded_pos_emb = pos_emb_for_seq.expand(batch_size, -1, -1) # Shape: [B, L, D]
        else: 
             expanded_pos_emb = pos_emb_for_seq
        fused_embeddings = self.fft_circular_convolution(inputs_embeds, expanded_pos_emb) # Shape: [B, L, D]

        # optipnal: L2 normalization
        #fused_embeddings = self.normalize_vector(fused_embeddings)
        #print("[DEBUG] L2 Normalization Applied to FFT Embeddings")
        final_embeddings = fused_embeddings + self.token_type_embeddings(token_type_ids) 
        final_embeddings = self.LayerNorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings)

        return final_embeddings

class FFTBertModel(BertModel): 
    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.embeddings = FFTBertEmbeddings(config)

class FFTBertForSequenceClassification(BertPreTrainedModel):
    """
    based on pretrained weights of BertForSequenceClassification.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config 
        self.bert = FFTBertModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init() 

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:] # outputs[0]=last_hidden_state, outputs[1]=pooler_output
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )