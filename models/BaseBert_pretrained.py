import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

class BaseBertForSequenceClassification(nn.Module):
    """
    BaseBert based on pretrained bert model for sequence classification.
    using pretrained weights.
    """
    def __init__(self, pretrained_model_name_or_path="pretrained_models/bert-base-uncased", num_labels=2):
        super(BaseBertForSequenceClassification, self).__init__()
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
        output_hidden_states=True,
        output_attentions=False
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
"""
# test sample
if __name__ == "__main__":
     from transformers import BertTokenizer
     pretrained_model_name = "pretrained_models/bert-base-uncased"
     model = BaseBertForSequenceClassification(pretrained_model_name, num_labels=2)
    
     tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
     sample_text = "This movie is amazing."
     inputs = tokenizer(sample_text, return_tensors="pt")
    
     outputs = model(**inputs)
     print("Logits:", outputs.logits.detach().numpy())
     pred_label = outputs.logits.argmax(dim=-1).item()
     print("Predicted label:", pred_label)
"""