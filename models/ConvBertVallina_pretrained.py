import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
class Conv2DEmbeddings_Vallina(nn.Module):
    def __init__(self, config):
        super(Conv2DEmbeddings_Vallina, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 简化的2D卷积层，只用于融合word和position embeddings
        self.fusion_conv = nn.Conv2d(
            in_channels=2,  # word和position作为两个输入通道
            out_channels=1,  # 输出融合后的单一通道
            kernel_size=(1, 1),  # 使用1x1卷积，只做通道维度上的融合
            bias=False  # 不需要偏置
        )
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = self.word_embeddings(input_ids)
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        word_embeds = inputs_embeds
        pos_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # 将word和position嵌入堆叠成[batch, 2, seq_len, hidden_size]的形状
        stacked_embeds = torch.stack([word_embeds, pos_embeds], dim=1)
        
        # 应用卷积进行融合
        fused_embeds = self.fusion_conv(stacked_embeds).squeeze(1)
        
        # 添加token type embeddings
        embeddings = fused_embeds + token_type_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings