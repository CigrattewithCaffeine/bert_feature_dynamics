# 文件: models/FFTBert_pretrained.py
# 版本: 用于加载预训练权重的模型

import torch
import torch.nn as nn
import os
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings as HFBertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput

# --- FFT 嵌入层定义 ---
class FFTBertEmbeddings(HFBertEmbeddings):
    """
    使用 FFT 循环卷积融合词嵌入和位置嵌入的 BERT 嵌入层。
    """
    def __init__(self, config: BertConfig):
        super().__init__(config)
        print(f"[INFO FFTEmb Init Rank {int(os.environ.get('LOCAL_RANK', 0))}] Initialized FFTBertEmbeddings (for pretrained).")

    def fft_circular_convolution(self, v1, v2):
        """使用 FFT 实现 v1 和 v2 沿最后一个维度的循环卷积"""
        fft1 = torch.fft.fft(v1, dim=-1)
        fft2 = torch.fft.fft(v2, dim=-1)
        convolved_fft = fft1 * fft2
        convolved = torch.fft.ifft(convolved_fft, dim=-1).real
        return convolved

    def normalize_vector(self, v, eps=1e-9):
        """(可选) 沿最后一个维度对向量进行 L2 归一化"""
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

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids) # Shape: [B, L, D]

        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids) # Shape: [1, L, D] or similar
        # --- 2. 准备并应用 FFT 融合 ---
        batch_size = input_shape[0]
        # 确保 position_embeddings 适配 batch 和 seq_length
        pos_emb_for_seq = position_embeddings[:, :seq_length, :]
        if pos_emb_for_seq.size(0) == 1: # 处理 [1, L, D] 的情况
             expanded_pos_emb = pos_emb_for_seq.expand(batch_size, -1, -1) # Shape: [B, L, D]
        else: # 如果已经是 [B, L, D] (某些实现可能是这样)
             expanded_pos_emb = pos_emb_for_seq

        # 应用 FFT 循环卷积
        fused_embeddings = self.fft_circular_convolution(inputs_embeds, expanded_pos_emb) # Shape: [B, L, D]

        # --- (可选) L2 归一化 ---
        # 尝试注释掉/取消注释这行来进行实验对比
        #fused_embeddings = self.normalize_vector(fused_embeddings)
        #print("[DEBUG] L2 Normalization Applied to FFT Embeddings")

        # --- 3. 合并与后续处理 ---
        final_embeddings = fused_embeddings
        final_embeddings = self.LayerNorm(final_embeddings)
        final_embeddings = self.dropout(final_embeddings)

        return final_embeddings

# --- FFT Bert 模型主体 (基于预训练模型结构) ---
class FFTBertModel(BertModel): # 继承自 BertModel 以保持结构一致性
    """使用 FFTBertEmbeddings 的 BERT 模型主体"""
    def __init__(self, config: BertConfig, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        # 用我们的 FFT 版本替换掉原始的嵌入层
        self.embeddings = FFTBertEmbeddings(config)

    # forward 方法直接继承自 BertModel，它会自动调用 self.embeddings.forward()

# --- 用于序列分类的 FFT Bert 模型 (加载预训练权重) ---
class FFTBertForSequenceClassification(BertPreTrainedModel):
    """
    用于序列分类的 FFT-BERT 模型 (基于预训练权重)。
    使用方法: model = FFTBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=...)
    它会自动加载 bert-base-uncased 的权重，然后用 FFTBertEmbeddings 替换原始嵌入层。
    """
    # 忽略 pooler 权重不匹配 (因为有时会重新初始化或配置不同)
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config # 保存 config

        # 实例化 FFTBertModel
        # 注意: BertPreTrainedModel 的 from_pretrained 会加载权重到这个 self.bert 中
        # 之后 FFTBertModel 的 __init__ (如果被调用) 或这里的实例化会确保使用 FFTBertEmbeddings
        self.bert = FFTBertModel(config, add_pooling_layer=True)

        # 分类头
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化分类头权重 (bert部分的权重由 from_pretrained 加载)
        self.post_init() # 应用 _init_weights 到整个模型，但预训练部分会被加载的权重覆盖

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
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 FFTBertModel 获取输出
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

        # 使用 Pooler 的输出 ([CLS] token representation)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 通过分类头
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        # 如果提供了 labels，计算损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 构建输出
        if not return_dict:
            output = (logits,) + outputs[2:] # outputs[0]=last_hidden_state, outputs[1]=pooler_output
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )