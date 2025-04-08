import os
import sys
import argparse
import torch
import numpy as np
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.data_utils import load_sst2
from utils.train_utils import create_dataloader
from models.BaseBert_pretrained import BaseBertForSequenceClassification
from models.ConvBert_pretrained import Conv2DBertForSequenceClassification
# from models.ConvFFT_pretrained import ConvFFTBertForSequenceClassification  # todo
# from models.ConvFFTFusion_pretrained import ConvFFTFusionBertForSequenceClassification  # todo
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

    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def save_cls_features(model, dataloader, save_dir, epoch, device):
    model.eval()
    features_dict = {}
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            )

            hidden_states = outputs.hidden_states
            labels = batch["labels"].cpu().numpy()

            for layer_idx, layer_output in enumerate(hidden_states):
                cls_features = layer_output[:, 0, :].cpu().numpy()
                if layer_idx not in features_dict:
                    features_dict[layer_idx] = []
                features_dict[layer_idx].append(cls_features)

            labels_list.append(labels)

    os.makedirs(save_dir, exist_ok=True)
    for layer_idx, features_list in features_dict.items():
        features = np.concatenate(features_list, axis=0)
        np.save(os.path.join(save_dir, f"layer{layer_idx}_epoch{epoch}.npy"), features)

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
    # elif model_type == "convfft_pretrained":
    #     return ConvFFTBertForSequenceClassification(...)
    # elif model_type == "convfft_fusion_pretrained":
    #     return ConvFFTFusionBertForSequenceClassification(...)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def main():
    parser = argparse.ArgumentParser(description="Train and extract BERT CLS features")
    parser.add_argument("--model_type", type=str, required=True, help="Model type to use")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
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

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    save_dir = f"/content/drive/MyDrive/bert_feature_outputs/{args.model_type}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        save_cls_features(model, feature_loader, save_dir, epoch, device)
        print(f"Saved features for epoch {epoch + 1}")

    print("Training completed!")

if __name__ == "__main__":
    main()
