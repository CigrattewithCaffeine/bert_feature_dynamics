import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
    
class Conv2DEmbeddings(nn.Module):
    """
    Interactive embeddings using Conv2D:
    Takes token embedding and position embedding as different channels into 2D convolution,
    then adds token type embedding, followed by LayerNorm and dropout.
    Supports HuggingFace standard interface, including inputs_embeds.
    """
    def __init__(self, config):
        super(Conv2DEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.interactive_conv = nn.Conv2d(
            in_channels=2,     # two input channels: token and position embeddings
            out_channels=1,     # one output channel
            kernel_size=(3, 3), # Kernel size
            padding=(1, 1),  
            bias=False        
        )
        
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
            inputs_embeds = self.word_embeddings(input_ids) # [B, L, D]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        seq_length = input_shape[1]
        device = inputs_embeds.device 
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape) 
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        word_embeds = inputs_embeds 
        pos_embeds = self.position_embeddings(position_ids) 

        word_embeds_reshaped = word_embeds.unsqueeze(1)
        pos_embeds_reshaped = pos_embeds.unsqueeze(1)
        combined_embeds = torch.cat([word_embeds_reshaped, pos_embeds_reshaped], dim=1)
        conv_output = self.interactive_conv(combined_embeds)
        fused_embeddings = self.activation(conv_output.squeeze(1))

        embeddings = fused_embeddings # [B, L, D]
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Conv2DBertForSequenceClassification(nn.Module):
    """
    Conv2D BERT model:
    1. Loads a pretrained BertForSequenceClassification.
    2. Replaces its bert.embeddings with custom Conv2DEmbeddings.
    3. Keeps the rest unchanged, ready for fine-tuning on downstream tasks (e.g., binary classification).
    """
    def __init__(self, pretrained_model_name_or_path, num_labels=2):
        super(Conv2DBertForSequenceClassification, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
        config.output_hidden_states = True 
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config)
        self.model.bert.embeddings = Conv2DEmbeddings(config)
        
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        output_hidden_states=True,
        output_attentions=True,  
    ):
        self.model.config.output_hidden_states = output_hidden_states
        self.model.config.output_attentions = output_attentions
        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return outputs


"""
# test sample
if __name__ == "__main__":
    from transformers import BertTokenizer
    pretrained_model_name = "pretrained_models/bert-base-uncased"
    model = Conv2DBertForSequenceClassification(pretrained_model_name, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    sample_text = "This movie is amazing."
    inputs = tokenizer(sample_text, return_tensors="pt")
    outputs = model(**inputs)
    print("Logits:", outputs.logits.detach().numpy())
"""