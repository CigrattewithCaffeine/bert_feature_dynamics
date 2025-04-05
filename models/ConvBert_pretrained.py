import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

class Conv2DEmbeddings(nn.Module):
    """
    使用Conv2D的交互式嵌入：
    将token embedding和position embedding作为不同通道输入到2D卷积中，
    然后与token type embedding相加，后续进行LayerNorm和dropout。
    支持HuggingFace标准接口，包括inputs_embeds。
    """
    def __init__(self, config):
        super(Conv2DEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 2D卷积层用于word和position embeddings的交互
        self.interactive_conv = nn.Conv2d(
            in_channels=2,  # word和position作为两个输入通道
            out_channels=1,  # 输出融合后的单一通道
            kernel_size=(1, 3),  # 在嵌入维度上不做卷积，在序列方向上使用kernel_size=3
            padding=(0, 1)
        )
        
        # 可选：添加残差连接的权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = self.word_embeddings(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        word_embeds = inputs_embeds
        pos_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        word_embeds_reshaped = word_embeds.unsqueeze(1)
        pos_embeds_reshaped = pos_embeds.unsqueeze(1)
        combined_embeds = torch.cat([word_embeds_reshaped, pos_embeds_reshaped], dim=1)
        conv_output = self.interactive_conv(combined_embeds)
        conv_output = conv_output.squeeze(1)

        simple_sum = word_embeds + pos_embeds
        embeddings = (1 - self.residual_weight) * simple_sum + self.residual_weight * conv_output
        embeddings = embeddings + token_type_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Conv2DBertForSequenceClassification(nn.Module):
    """
    Conv2D BERT模型：
    1. 加载预训练的BertForSequenceClassification。
    2. 替换其bert.embeddings为自定义的Conv2DEmbeddings。
    3. 其余部分保持不变，直接用于微调下游任务（如二分类）。
    """
    def __init__(self, pretrained_model_name_or_path, num_labels=2):
        super(Conv2DBertForSequenceClassification, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
        config.output_hidden_states = True 
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config)
        
        # 替换embedding层
        self.model.bert.embeddings = Conv2DEmbeddings(config)
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, output_hidden_states=False):
        self.model.config.output_hidden_states = output_hidden_states
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


# 简单测试示例（可选）
if __name__ == "__main__":
    from transformers import BertTokenizer

    pretrained_model_name = "pretrained_models/bert-base-uncased"
    model = Conv2DBertForSequenceClassification(pretrained_model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    sample_text = "This movie is amazing."
    inputs = tokenizer(sample_text, return_tensors="pt")

    # 测试前向传播
    outputs = model(**inputs)
    print("Logits:", outputs.logits.detach().numpy())
#"""