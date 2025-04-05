import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

class BaseBertForSequenceClassification(nn.Module):
    """
    基于预训练模型的 BaseBERT，用于下游任务微调（方案二）。
    直接加载 Hugging Face 提供的预训练权重，不修改 embedding 层。
    """
    def __init__(self, pretrained_model_name_or_path="pretrained_models/bert-base-uncased", num_labels=2):
        super(BaseBertForSequenceClassification, self).__init__()
        # 加载预训练模型和配置，指定任务的类别数
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels
        )

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        labels=None, 
        inputs_embeds=None,
        output_hidden_states=False
    ):
        # 调用预训练模型的 forward 方法，支持输出 hidden states
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states
        )
"""
# 简单测试示例（可选）
if __name__ == "__main__":
     from transformers import BertTokenizer
    # 指定预训练模型名称
     pretrained_model_name = "pretrained_models/bert-base-uncased"
     model = BaseBertForSequenceClassification(pretrained_model_name, num_labels=2)
    
    # 加载对应的 Tokenizer
     tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
     sample_text = "This movie is amazing."
     inputs = tokenizer(sample_text, return_tensors="pt")
    
    # 前向传播
     outputs = model(**inputs)
     print("Logits:", outputs.logits.detach().numpy())
     pred_label = outputs.logits.argmax(dim=-1).item()
     print("Predicted label:", pred_label)
"""