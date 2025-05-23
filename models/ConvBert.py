import torch
import torch.nn as nn
from transformers import BertConfig, BertModel,PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ConvBert_pretrained import Conv2DEmbeddings 

class Conv2DBertBaseForSequenceClassification(PreTrainedModel):
    config_class = BertConfig
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert.embeddings = Conv2DEmbeddings(self.config)
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
""""
# test sample
if __name__ == "__main__":
    from transformers import BertTokenizer, BertConfig
    config = BertConfig(
        vocab_size=30522,  
        hidden_size=768,   
        num_hidden_layers=12,
        num_attention_heads=12,
        num_labels=2       
    )

    model = Conv2DBertBaseForSequenceClassification(config)
    
    pretrained_model_name = "pretrained_models/bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    
    sample_text = "This movie is amazing."
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    model.eval()  
    with torch.no_grad():
        outputs = model(**inputs)
        
    print("Input text:", sample_text)
    print("Logits:", outputs.logits.numpy())
    pred_label = outputs.logits.argmax(dim=-1).item()
    print("Predicted label:", pred_label)
    
    if outputs.hidden_states is not None:
        for idx, hidden_state in enumerate(outputs.hidden_states):
            print(f"Layer {idx} hidden state shape:", hidden_state.shape)
#"""