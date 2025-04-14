import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
import os
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings as HFBertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

from transformers import BertConfig, BertModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class FFTBertEmbeddings(HFBertEmbeddings):
    """
    A modified BERT embedding using fft convolution to fuse word & position embeddings.
    """
    def __init__(self, config: BertConfig):
        super().__init__(config)
        print(f"[INFO FFTEmb Init Rank {int(os.environ.get('LOCAL_RANK', 0))}] Initialized FFTBertEmbeddings (for from-scratch).")

    def fft_circular_convolution(self, v1, v2):
        fft1 = torch.fft.fft(v1, dim=-1)
        fft2 = torch.fft.fft(v2, dim=-1)
        convolved_fft = fft1 * fft2
        convolved = torch.fft.ifft(convolved_fft, dim=-1).real
        return convolved

    def normalize_vector(self, v, eps=1e-9):
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
        batch_size = input_shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        pos_emb_for_seq = position_embeddings[:, :seq_length, :]
        if pos_emb_for_seq.size(0) == 1:
             expanded_pos_emb = pos_emb_for_seq.expand(batch_size, -1, -1)
        else:
             expanded_pos_emb = pos_emb_for_seq

        fused_embeddings = self.fft_circular_convolution(inputs_embeds, expanded_pos_emb)

        #optional:L2 normalization
        #fused_embeddings = self.normalize_vector(fused_embeddings)
        #print("L2 Normalization Applied to FFT Embeddings")

        final_embeddings = fused_embeddings + self.token_type_embeddings(token_type_ids)
        final_embeddings = self.LayerNorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings)

        return final_embeddings

class FFTBertBaseForSequenceClassification(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert.embeddings = FFTBertEmbeddings(self.config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        inputs_embeds=None,
        output_hidden_states=True,
        output_attentions=False,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




