import os
import sys
import argparse
import torch
import numpy as np
import json
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.data_utils import load_sst2
from utils.train_utils import create_dataloader
from models.BaseBert_pretrained import BaseBertForSequenceClassification
from models.ConvBert_pretrained import Conv2DBertForSequenceClassification
from models.ConvBert import Conv2DBertBaseForSequenceClassification
from models.BaseBert import BaseBertBaseForSequenceClassification
from datetime import datetime
from transformers import BertConfig

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def train_epoch(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    if hasattr(model, "model") and hasattr(model.model, "bert") and hasattr(model.model.bert, "embeddings"):
        if hasattr(model.model.bert.embeddings, "residual_weight"):
            print(f"Residual weight: {model.model.bert.embeddings.residual_weight.item():.4f}")

    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def save_cls_features(model, dataloader, save_dir, epoch, device):
    model.eval()
    features_dict = {}
    labels_list = []
    attention_dict = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),
                output_hidden_states=True,
                output_attentions=True
            )
            
            # 调试输出，检查是否正确返回注意力权重
            if epoch == 0 and len(features_dict) == 0:
                print("Hidden states available:", outputs.hidden_states is not None)
                print("Attention weights available:", outputs.attentions is not None)
                if hasattr(outputs, "attentions") and outputs.attentions:
                    print("Attention shape:", [a.shape for a in outputs.attentions])
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions
            labels = batch["labels"].cpu().numpy()

            for layer_idx, layer_output in enumerate(hidden_states):
                cls_features = layer_output[:, 0, :].cpu().numpy()
                features_dict.setdefault(layer_idx, []).append(cls_features)

            for layer_idx, attn in enumerate(attentions):
                mean_attn = attn[:, :, 0, :].mean(dim=1).cpu().numpy()
                attention_dict.setdefault(layer_idx, []).append(mean_attn)

            labels_list.append(labels)

    os.makedirs(save_dir, exist_ok=True)
    for layer_idx in features_dict:
        features = np.concatenate(features_dict[layer_idx], axis=0)
        np.save(os.path.join(save_dir, f"layer{layer_idx}_epoch{epoch}.npy"), features)

    for layer_idx in attention_dict:
        attn_scores = np.concatenate(attention_dict[layer_idx], axis=0)
        np.save(os.path.join(save_dir, f"attention_layer{layer_idx}_epoch{epoch}.npy"), attn_scores)

    labels = np.concatenate(labels_list, axis=0)
    np.save(os.path.join(save_dir, f"labels_epoch{epoch}.npy"), labels)

def get_model(model_type):
    if model_type == "conv2D_pretrained":
        return Conv2DBertForSequenceClassification("bert-base-uncased", num_labels=2)
    elif model_type == "base_pretrained":
        return BaseBertForSequenceClassification("bert-base-uncased", num_labels=2)
    elif model_type == "conv2D":
        config = BertConfig(num_labels=2)
        return Conv2DBertBaseForSequenceClassification(config)
    elif model_type == "base":
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
        return BaseBertBaseForSequenceClassification(config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def freeze_bert_layers(model, num_layers_to_freeze, freeze_embeddings=True):
    """冻结BERT模型的指定层数"""
    if hasattr(model, "model"):  # 处理包装模型的情况
        bert_model = model.model.bert
    else:
        bert_model = model.bert
    
    # 根据参数决定是否冻结嵌入层
    if freeze_embeddings:
        for param in bert_model.embeddings.parameters():
            param.requires_grad = False
        print("Embeddings layer frozen")
    else:
        print("Embeddings layer kept trainable")
    
    # 冻结编码器层
    for i in range(min(12, num_layers_to_freeze)):
        if hasattr(bert_model.encoder.layer, str(i)):
            for param in bert_model.encoder.layer[i].parameters():
                param.requires_grad = False
            print(f"Encoder layer {i} frozen")
    
    # 打印参数的梯度状态
    print("Parameter gradient status:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

def unfreeze_all(model):
    """解冻所有层"""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen")

def main():
    parser = argparse.ArgumentParser(description="Train and extract BERT CLS features")
    parser.add_argument("--model_type", type=str, required=True, help="Model type to use")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of warmup epochs")
    parser.add_argument("--freeze_layers", type=int, default=12, help="Number of layers to freeze during warmup (0-12)")
    parser.add_argument("--freeze_embeddings", type=int, default=1, help="freeze embeddings layer or not")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SST2 dataset...")
    sst2_splits = load_sst2()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = create_dataloader(sst2_splits['train'], tokenizer, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(sst2_splits['dev'], tokenizer, batch_size=args.batch_size)
    feature_loader = create_dataloader(sst2_splits['dev'], tokenizer, batch_size=args.batch_size, shuffle=False)

    print(f"Initializing model: {args.model_type}...")
    model = get_model(args.model_type)
    model = model.to(device)

    # 首先冻结指定的层
    if args.warmup_epochs > 0 and args.freeze_layers > 0:
        print(f"Starting with {args.freeze_layers} layers frozen for {args.warmup_epochs} epochs")
    # 将args.freeze_embeddings转换为布尔值
        freeze_emb = args.freeze_embeddings == 1
        freeze_bert_layers(model, args.freeze_layers, freeze_embeddings=freeze_emb)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_dir = f"/content/drive/MyDrive/bert_feature_outputs/{args.model_type}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    metrics = []

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # 如果达到了预热结束的条件，解冻所有层
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            print("Warmup complete, unfreezing all layers")
            unfreeze_all(model)
            # 更新优化器，因为有些参数现在变为可训练了
            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            # 重新计算总步数并更新scheduler
            remaining_steps = len(train_loader) * (args.num_epochs - epoch)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=remaining_steps)

        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        save_cls_features(model, feature_loader, save_dir, epoch, device)
        print(f"Saved features for epoch {epoch + 1}")

        metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": accuracy
        })

        with open(os.path.join(save_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print("Training completed!")

if __name__ == "__main__":
    main()
