# 文件: models/FFTBert.py
# 版本: 用于从头随机初始化训练的模型

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint
import os
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings as HFBertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

# --- FFT 嵌入层定义 (与上面 pretrained 版本相同) ---
class FFTBertEmbeddings(HFBertEmbeddings):
    """
    使用 FFT 循环卷积融合词嵌入和位置嵌入的 BERT 嵌入层。
    (代码与 pretrained 版本中的 FFTBertEmbeddings 完全相同)
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
        # --- 1. 获取基础嵌入 ---
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
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # --- 2. 准备并应用 FFT 融合 ---
        pos_emb_for_seq = position_embeddings[:, :seq_length, :]
        if pos_emb_for_seq.size(0) == 1:
             expanded_pos_emb = pos_emb_for_seq.expand(batch_size, -1, -1)
        else:
             expanded_pos_emb = pos_emb_for_seq

        fused_embeddings = self.fft_circular_convolution(inputs_embeds, expanded_pos_emb)

        # --- (可选) L2 归一化 ---
        # fused_embeddings = self.normalize_vector(fused_embeddings)
        # print("[DEBUG] L2 Normalization Applied to FFT Embeddings")

        # --- 3. 合并与后续处理 ---
        final_embeddings = fused_embeddings + token_type_embeddings
        final_embeddings = self.LayerNorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings)

        return final_embeddings


# --- FFT Bert 模型主体 (从头初始化) ---
class FFTBertModel(BertPreTrainedModel): # 继承以复用 _init_weights
    """使用 FFTBertEmbeddings 的 BERT 模型主体 (从头初始化)"""
    # 指定这是一个基础模型，不加载预训练的 pooler 头
    base_model_prefix = "bert"

    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config) # 传递 config
        self.config = config

        # 实例化我们自定义的嵌入层和标准的 Encoder/Pooler
        self.embeddings = FFTBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # 权重由 post_init() 方法 (来自 BertPreTrainedModel) 初始化
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # forward 方法与 pretrained 版本中的 FFTBertModel 基本一致
    # (为简洁省略重复代码，可以直接复用上面 pretrained 版本里的 forward)
    def forward(self, *args, **kwargs):
        # 调用父类(BertModel)的forward方法可能会出错，因为它会调用自身的embeddings
        # 需要确保调用的是这个类实例化的 self.embeddings 和 self.encoder
        # 最安全的方式是复制粘贴 BertModel 的 forward 逻辑到这里，确保调用正确的子模块

        # --- 复用 BertModel 的 forward 逻辑 ---
        output_attentions = kwargs.get("output_attentions", self.config.output_attentions)
        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        return_dict = kwargs.get("return_dict", self.config.use_return_dict)

        input_ids=kwargs.get("input_ids")
        inputs_embeds=kwargs.get("inputs_embeds")
        attention_mask=kwargs.get("attention_mask")
        token_type_ids=kwargs.get("token_type_ids")
        position_ids=kwargs.get("position_ids")
        head_mask=kwargs.get("head_mask")
        encoder_hidden_states=kwargs.get("encoder_hidden_states")
        encoder_attention_mask=kwargs.get("encoder_attention_mask")
        past_key_values=kwargs.get("past_key_values")
        use_cache=kwargs.get("use_cache", self.config.use_cache if self.config.is_decoder else False)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        # ... (省略 input_shape, device, past_key_values_length, mask 处理逻辑 - 与 pretrained 版本相同)
        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        else:
            raise ValueError("Specify either input_ids or inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        # ... (省略 encoder_extended_attention_mask 和 head_mask 处理)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_extended_attention_mask = None # 简化处理，假设非 decoder

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds, past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output, pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values, hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions, cross_attentions=encoder_outputs.cross_attentions,
        )


# --- 用于序列分类的 FFT Bert 模型 (从头初始化) ---
class FFTBertForSequenceClassification(BertPreTrainedModel):
    """
    用于序列分类的 FFT-BERT 模型 (从头随机初始化)。
    使用方法:
    config = BertConfig(...) # 定义模型结构
    model = FFTBertForSequenceClassification(config)
    """
    def __init__(self, config: BertConfig):
        super().__init__(config) # 初始化父类，主要是设置 config
        self.num_labels = config.num_labels
        self.config = config

        # 实例化 FFTBertModel
        self.bert = FFTBertModel(config, add_pooling_layer=True)

        # 分类头
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重
        # post_init() 会调用继承自 BertPreTrainedModel 的 _init_weights 方法
        # 对 self.bert (包括 embeddings, encoder, pooler) 和 self.classifier 进行初始化
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
        """
        前向传播函数，与标准的 BertForSequenceClassification 兼容。
        (代码与 pretrained 版本中的 forward 完全相同)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )