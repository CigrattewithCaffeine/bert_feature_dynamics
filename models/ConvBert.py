import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ConvBert_pretrained import Conv2DEmbeddings 

class Conv2DBertBaseForSequenceClassification(nn.Module):
    """
    完整從零初始化的 Conv2D BERT 模型，用於序列分類
    沒有使用任何預訓練權重，只改 embedding 結構
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel(self.config)
        self.bert.embeddings = Conv2DEmbeddings(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, output_hidden_states=True,output_attentions=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        pooled_output = outputs.pooler_output  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
""""
# 简单测试示例（可选）
if __name__ == "__main__":
    from transformers import BertTokenizer, BertConfig
    # 创建配置
    config = BertConfig(
        vocab_size=30522,  # 词表大小
        hidden_size=768,   # 隐藏层大小
        num_hidden_layers=12,
        num_attention_heads=12,
        num_labels=2       # 分类数量
    )
    
    # 创建随机初始化的模型
    model = Conv2DBertBaseForSequenceClassification(config)
    
    # 使用基础的 BERT tokenizer
    pretrained_model_name = "pretrained_models/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    
    # 测试文本
    sample_text = "This movie is amazing."
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 打印结果
    print("Input text:", sample_text)
    print("Logits:", outputs.logits.numpy())
    pred_label = outputs.logits.argmax(dim=-1).item()
    print("Predicted label:", pred_label)
    
    # 打印每层的 hidden states 形状
    if outputs.hidden_states is not None:
        for idx, hidden_state in enumerate(outputs.hidden_states):
            print(f"Layer {idx} hidden state shape:", hidden_state.shape)
#"""