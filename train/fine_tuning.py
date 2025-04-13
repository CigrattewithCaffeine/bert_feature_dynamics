import os
import sys
import argparse
import torch
import numpy as np
import json
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup,BertForSequenceClassification
from tqdm import tqdm
from utils.data_utils import load_sst2
from utils.train_utils import create_dataloader
from models.BaseBert_pretrained import BaseBertForSequenceClassification
from models.ConvBert_pretrained import Conv2DBertForSequenceClassification 
from models.ConvBert import Conv2DBertBaseForSequenceClassification      
from models.BaseBert import BaseBertBaseForSequenceClassification        
from models.FFTBert import FFTBertBaseForSequenceClassification      
from models.FFTBert_pretrained import FFTBertForSequenceClassification    
from datetime import datetime
from transformers import BertConfig, AutoConfig
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def calculate_metrics(param):
    if param is None or param.numel() == 0:
        return 0.0, 1.0
    l2_norm = torch.norm(param.data.detach(), p=2).item()
    sparsity = (param.data.detach() == 0).float().mean().item()
    return l2_norm, sparsity

def collect_grad_metrics(model):
    grad_metrics = {}
    
    # 检查主模型是否是Conv2D类型的直接实例
    is_direct_conv2d = isinstance(model, (Conv2DBertForSequenceClassification, Conv2DBertBaseForSequenceClassification))
    
    # 检查是否是通过model.model访问的Conv2D结构
    is_nested_conv2d = False
    if hasattr(model, 'model') and isinstance(model.model, BertForSequenceClassification):
        if hasattr(model.model, 'bert') and hasattr(model.model.bert, 'embeddings'):
            is_nested_conv2d = hasattr(model.model.bert.embeddings, 'interactive_conv')
    
    # 获取embeddings梯度
    if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
        # 直接访问模式
        embeddings = model.bert.embeddings
    elif hasattr(model, 'model') and hasattr(model.model, 'bert') and hasattr(model.model.bert, 'embeddings'):
        # 嵌套访问模式
        embeddings = model.model.bert.embeddings
    else:
        embeddings = None
    
    # 收集词嵌入和位置嵌入的梯度
    if embeddings is not None:
        if hasattr(embeddings, 'word_embeddings') and embeddings.word_embeddings.weight.grad is not None:
            grad_metrics["word_emb_grad_norm"] = torch.norm(embeddings.word_embeddings.weight.grad).item()
        if hasattr(embeddings, 'position_embeddings') and embeddings.position_embeddings.weight.grad is not None:
            grad_metrics["pos_emb_grad_norm"] = torch.norm(embeddings.position_embeddings.weight.grad).item()
        
        # 特别关注interactive_conv的梯度
        if hasattr(embeddings, 'interactive_conv') and embeddings.interactive_conv.weight.grad is not None:
            grad_metrics["interactive_conv_grad_norm"] = torch.norm(embeddings.interactive_conv.weight.grad).item()
            # 添加更详细的卷积核梯度信息
            if embeddings.interactive_conv.weight.grad.numel() > 0:
                grad_metrics["interactive_conv_grad_mean"] = embeddings.interactive_conv.weight.grad.abs().mean().item()
                grad_metrics["interactive_conv_grad_max"] = embeddings.interactive_conv.weight.grad.abs().max().item()
    
    # 获取attention layers梯度
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
        encoder_layers = model.bert.encoder.layer
    elif hasattr(model, 'model') and hasattr(model.model, 'bert') and hasattr(model.model.bert, 'encoder'):
        encoder_layers = model.model.bert.encoder.layer
    else:
        encoder_layers = None
    
    if encoder_layers is not None:
        for layer_idx in [0, 5, 10]:
            if layer_idx < len(encoder_layers):
                layer = encoder_layers[layer_idx]
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'self') and hasattr(layer.attention.self, 'query'):
                    if layer.attention.self.query.weight.grad is not None:
                        grad_metrics[f"attn_{layer_idx+1}_query_grad_norm"] = torch.norm(layer.attention.self.query.weight.grad).item()
    
    # 获取classifier梯度
    if hasattr(model, 'classifier') and model.classifier.weight.grad is not None:
        grad_metrics["classifier_grad_norm"] = torch.norm(model.classifier.weight.grad).item()
    elif hasattr(model, 'model') and hasattr(model.model, 'classifier') and model.model.classifier.weight.grad is not None:
        grad_metrics["classifier_grad_norm"] = torch.norm(model.model.classifier.weight.grad).item()
    return grad_metrics

def train_epoch(model, train_loader, optimizer, scheduler, device, model_type, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    is_conv2d_model = isinstance(model, (Conv2DBertForSequenceClassification, Conv2DBertBaseForSequenceClassification))
    
    all_batch_grad_metrics = []

    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if batch_idx == 0:
            print(f"\n--- Gradient Debug Info (Epoch {epoch}, Batch 0) ---")
            
            # Embeddings
            if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                if hasattr(model.bert.embeddings, 'word_embeddings') and model.bert.embeddings.word_embeddings.weight.grad is not None:
                    print(f"  Word Embeddings Grad Norm: {torch.norm(model.bert.embeddings.word_embeddings.weight.grad).item():.4f}")
                else:
                    print("  Word Embeddings Grad: N/A (or frozen)")
                if hasattr(model.bert.embeddings, 'position_embeddings') and model.bert.embeddings.position_embeddings.weight.grad is not None:
                    print(f"  Position Embeddings Grad Norm: {torch.norm(model.bert.embeddings.position_embeddings.weight.grad).item():.4f}")
                else:
                    print("  Position Embeddings Grad: N/A (or frozen)")
            else:
                print(" Embeddings grads not accessible via model.bert.embeddings")

            # Attention Layers (1, 6, 11 - corresponding to index 0, 5, 10)
            if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
                for layer_idx in [0, 5, 10]:
                    if layer_idx < len(model.bert.encoder.layer):
                        layer = model.bert.encoder.layer[layer_idx]
                        if hasattr(layer, 'attention') and hasattr(layer.attention, 'self') and hasattr(layer.attention.self, 'query') and layer.attention.self.query.weight.grad is not None:
                            print(f"  Attention Layer {layer_idx+1} (Query) Grad Norm: {torch.norm(layer.attention.self.query.weight.grad).item():.4f}")
                        else:
                            print(f"  Attention Layer {layer_idx+1} (Query) Grad: N/A (or frozen)")
                    else:
                        print(f"  Attention Layer {layer_idx+1}: Does not exist")
            else:
                print(" Attention layer grads not accessible via model.bert.encoder.layer")

            # Classifier Head
            if hasattr(model, 'classifier') and model.classifier.weight.grad is not None:
                print(f"  Classifier Head Grad Norm: {torch.norm(model.classifier.weight.grad).item():.4f}")
            else:
                print("  Classifier Head Grad: N/A (or frozen)")

            # Conv2D Specific Debug Prints
            if is_conv2d_model:
                # 定位并检查interactive_conv
                interactive_conv = None
                if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings') and hasattr(model.bert.embeddings, 'interactive_conv'):
                    interactive_conv = model.bert.embeddings.interactive_conv
                elif hasattr(model, 'model') and hasattr(model.model, 'bert') and hasattr(model.model.bert, 'embeddings'):
                    if hasattr(model.model.bert.embeddings, 'interactive_conv'):
                        interactive_conv = model.model.bert.embeddings.interactive_conv
    
                if interactive_conv is not None:
                    if interactive_conv.weight.grad is not None:
                        grad_norm = torch.norm(interactive_conv.weight.grad).item()
                        print(f"  Interactive Conv Grad Norm: {grad_norm:.4f}")
                        print(f"  Interactive Conv Grad Mean: {interactive_conv.weight.grad.abs().mean().item():.4f}")
                        print(f"  Interactive Conv Grad Max: {interactive_conv.weight.grad.abs().max().item():.4f}")
                        print(f"  Interactive Conv Weight Shape: {interactive_conv.weight.shape}")
            
                        # 打印具体的权重和梯度样本
                        flat_weight = interactive_conv.weight.view(-1)
                        flat_grad = interactive_conv.weight.grad.view(-1)
                        for i in range(min(4, flat_weight.numel())):
                            print(f"  Conv Weight[{i}]: {flat_weight[i].item():.4f}, Grad[{i}]: {flat_grad[i].item():.6f}")
                    else:
                            print("  Interactive Conv Grad: N/A (or frozen)")
                else:
                   print("  Interactive Conv Layer Not Found")
            print("--- End Gradient Debug Info ---")
            
            # Collect gradient metrics for this batch
            batch_grad_metrics = collect_grad_metrics(model)
            all_batch_grad_metrics.append(batch_grad_metrics)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Calculate average gradient metrics across batches
    avg_grad_metrics = {}
    if all_batch_grad_metrics:
        for key in all_batch_grad_metrics[0].keys():
            avg_grad_metrics[key] = sum(batch[key] for batch in all_batch_grad_metrics if key in batch) / len(all_batch_grad_metrics)
    
    return total_loss / len(train_loader), avg_grad_metrics

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

            if epoch == 0 and len(features_dict) == 0:
                print("Hidden states available:", outputs.hidden_states is not None)
                print("Attention weights available:", outputs.attentions is not None)
                if hasattr(outputs, "attentions") and outputs.attentions:
                    print("Attention shape:", [a.shape for a in outputs.attentions])
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions
            labels = batch["labels"].cpu().numpy()

            if hidden_states:
                for layer_idx, layer_output in enumerate(hidden_states):
                    cls_features = layer_output[:, 0, :].cpu().numpy()
                    features_dict.setdefault(layer_idx, []).append(cls_features)
            else:
                print("Warning: Hidden states not available in model output.")

            if attentions:
                for layer_idx, attn in enumerate(attentions):
                    mean_attn = attn[:, :, 0, :].mean(dim=1).cpu().numpy()
                    attention_dict.setdefault(layer_idx, []).append(mean_attn)
            else:
                print("Warning: Attention weights not available in model output.")

            labels_list.append(labels)

    os.makedirs(save_dir, exist_ok=True)
    if features_dict:
        for layer_idx in features_dict:
            features = np.concatenate(features_dict[layer_idx], axis=0)
            np.save(os.path.join(save_dir, f"layer{layer_idx}_epoch{epoch}.npy"), features)

    if attention_dict:
        for layer_idx in attention_dict:
            attn_scores = np.concatenate(attention_dict[layer_idx], axis=0)
            np.save(os.path.join(save_dir, f"attention_layer{layer_idx}_epoch{epoch}.npy"), attn_scores)

    if labels_list:
        labels = np.concatenate(labels_list, axis=0)
        np.save(os.path.join(save_dir, f"labels_epoch{epoch}.npy"), labels)

def get_model(model_type):
    if model_type == "conv2D_pretrained":
        return Conv2DBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    elif model_type == "base_pretrained":
        return BaseBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    elif model_type == "conv2D":
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
        return Conv2DBertBaseForSequenceClassification(config)
    elif model_type == "base":
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
        return BaseBertBaseForSequenceClassification(config)
    elif model_type == "fft":
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=2)
        return FFTBertBaseForSequenceClassification(config)
    elif model_type == "fft_pretrained":
        return FFTBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def freeze_bert_layers(model, num_layers_to_freeze, freeze_embeddings=True):
    print(f"\n--- Entering freeze_bert_layers ---")
    print(f"Attempting to freeze {num_layers_to_freeze} encoder layers.")
    print(f"Parameter freeze_embeddings: {freeze_embeddings}")

    bert_model = None
    if hasattr(model, "bert"):
        bert_model = model.bert
        print("Accessing model parameters via model.bert")
    elif hasattr(model, "model") and hasattr(model.model, "bert"):
        bert_model = model.model.bert
        print("Accessing model parameters via model.model.bert")
    else:
        print("[WARNING] Cannot find standard bert model structure (model.bert or model.model.bert). Freeze might not work as expected.")
        if hasattr(model, 'embeddings') and hasattr(model, 'encoder'):
            print("Assuming model itself has 'embeddings' and 'encoder' attributes.")
            bert_model = model
        else:
            print("[ERROR] Could not identify BERT components. Aborting freeze.")
            return

    if bert_model is None:
        print("[ERROR] bert_model identification failed. Aborting freeze.")
        return

    print("\nProcessing Embeddings Layer...")
    if hasattr(bert_model, 'embeddings'):
        if freeze_embeddings:
            print("  Action: Freezing embeddings...")
            count_emb_frozen = 0
            for name, param in bert_model.embeddings.named_parameters():
                param.requires_grad = False
                count_emb_frozen += 1
            print(f"  Result: Embeddings layer frozen ({count_emb_frozen} parameters).")
        else:
            print("  Action: Keeping embeddings trainable.")
            count_emb_trainable = 0
            for name, param in bert_model.embeddings.named_parameters():
                param.requires_grad = True
                count_emb_trainable += 1
            print(f"  Result: Embeddings layer kept trainable ({count_emb_trainable} parameters).")
    else:
        print(" [WARNING] No 'embeddings' attribute found in bert_model. Skipping embedding freeze/unfreeze.")

    print("\nProcessing Encoder Layers...")
    if hasattr(bert_model, 'encoder') and hasattr(bert_model.encoder, 'layer'):
        num_actual_layers = len(bert_model.encoder.layer)
        config_num_layers = getattr(getattr(bert_model, 'config', None), 'num_hidden_layers', num_actual_layers)

        actual_layers_to_freeze = min(config_num_layers, num_layers_to_freeze)
        print(f"  Model reports {config_num_layers} layers. Attempting to freeze first {actual_layers_to_freeze} (0 to {actual_layers_to_freeze - 1})...")

        for i in range(actual_layers_to_freeze):
            if i < num_actual_layers:
                count_layer_frozen = 0
                print(f"  Freezing Encoder layer {i}...")
                for name, param in bert_model.encoder.layer[i].named_parameters():
                    param.requires_grad = False
                    count_layer_frozen += 1
                print(f"    Layer {i} frozen ({count_layer_frozen} parameters).")
            else:
                print(f"  Warning: Attempted to freeze layer {i}, but model only has {num_actual_layers} layers.")
                break
    else:
        print(" [WARNING] No 'encoder' or 'encoder.layer' attribute found in bert_model. Skipping encoder layer freeze.")

    print("\n--- Final Parameter Gradient Status Check ---")
    trainable_params_count = 0
    frozen_params_count = 0
    for name, param in model.named_parameters():
        status = "Trainable" if param.requires_grad else "FROZEN"
        if param.requires_grad:
            trainable_params_count += param.numel()
        else:
            frozen_params_count += param.numel()
    print(f"Total Trainable Parameters: {trainable_params_count}")
    print(f"Total Frozen Parameters: {frozen_params_count}")
    print(f"--- Exiting freeze_bert_layers ---\n")

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen")

def main():
    parser = argparse.ArgumentParser(description="Train and extract BERT CLS features")
    parser.add_argument("--model_type", type=str, required=True, help="Model type to use")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of layers to freeze during warmup (0-12)")
    parser.add_argument("--freeze_embeddings", type=int, default=0, help="freeze embeddings layer or not (1 for True, 0 for False)")
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

    if args.warmup_epochs > 0 and (args.freeze_layers > 0 or args.freeze_embeddings == 1):
        print(f"Starting with {args.freeze_layers} layers frozen and embeddings frozen={args.freeze_embeddings==1} for {args.warmup_epochs} epochs")
        freeze_emb = args.freeze_embeddings == 1
        freeze_bert_layers(model, args.freeze_layers, freeze_embeddings=freeze_emb)
    elif args.freeze_layers > 0 or args.freeze_embeddings == 1:
        print(f"[Warning] freeze_layers or freeze_embeddings is set, but warmup_epochs is 0. Layers will be frozen for the entire training unless unfrozen manually.")
        freeze_emb = args.freeze_embeddings == 1
        freeze_bert_layers(model, args.freeze_layers, freeze_embeddings=freeze_emb)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    try:
        base_output_dir = "/content/drive/MyDrive/bert_feature_outputs"
        save_dir = os.path.join(base_output_dir, f"{args.model_type}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output will be saved to: {save_dir}")
    except Exception as e:
        print(f"Error creating save directory '{save_dir}'. Please check permissions or path. Error: {e}")
        print("Using local directory './bert_feature_outputs' as fallback.")
        base_output_dir = "./bert_feature_outputs"
        save_dir = os.path.join(base_output_dir, f"{args.model_type}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

    metrics = []
    is_conv2d_model_main = isinstance(model, (Conv2DBertForSequenceClassification, Conv2DBertBaseForSequenceClassification))

    print("Starting training...")
    for epoch in range(args.num_epochs):
        if epoch == args.warmup_epochs and args.warmup_epochs > 0 and (args.freeze_layers > 0 or args.freeze_embeddings == 1):
            print(f"\n--- Epoch {epoch}: Warmup complete, unfreezing layers ---")
            unfreeze_all(model)
            print("Re-initializing optimizer for all parameters...")
            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            remaining_steps = len(train_loader) * (args.num_epochs - epoch)
            print(f"Re-initializing scheduler for remaining {remaining_steps} steps...")
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=remaining_steps)
            print("--- Unfreezing complete ---")

        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        train_loss, grad_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, args.model_type, epoch)
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
        }
        
        # Add gradient metrics to epoch metrics
        for key, value in grad_metrics.items():
            epoch_metrics[key] = value
        
        with torch.no_grad():
            # Embeddings
            if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                if hasattr(model.bert.embeddings, 'word_embeddings'):
                    l2, spars = calculate_metrics(model.bert.embeddings.word_embeddings.weight)
                    epoch_metrics["word_emb_l2"] = l2
                    epoch_metrics["word_emb_sparsity"] = spars
                if hasattr(model.bert.embeddings, 'position_embeddings'):
                    l2, spars = calculate_metrics(model.bert.embeddings.position_embeddings.weight)
                    epoch_metrics["pos_emb_l2"] = l2
                    epoch_metrics["pos_emb_sparsity"] = spars
                if hasattr(model.bert.embeddings, "residual_weight"):
                    res_weight = model.bert.embeddings.residual_weight
                    epoch_metrics["residual_weight"] = res_weight.item() if torch.is_tensor(res_weight) else res_weight
                else:
                    epoch_metrics["residual_weight"] = None

            # Attention Layers (1, 6, 11) - Calculate for Query weight
            if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
                for layer_idx in [0, 5, 10]:
                    if layer_idx < len(model.bert.encoder.layer):
                        layer = model.bert.encoder.layer[layer_idx]
                        if hasattr(layer, 'attention') and hasattr(layer.attention, 'self') and hasattr(layer.attention.self, 'query'):
                            l2, spars = calculate_metrics(layer.attention.self.query.weight)
                            epoch_metrics[f"attn_{layer_idx+1}_query_l2"] = l2
                            epoch_metrics[f"attn_{layer_idx+1}_query_sparsity"] = spars

            # Classifier Head
            if hasattr(model, 'classifier'):
                l2, spars = calculate_metrics(model.classifier.weight)
                epoch_metrics["classifier_l2"] = l2
                epoch_metrics["classifier_sparsity"] = spars

            # Conv2D 1x1 Kernel (if applicable)
            if is_conv2d_model_main:
                # Main 1x1 conv kernel
                if hasattr(model, 'conv1x1') and hasattr(model.conv1x1, 'weight'):
                    l2, spars = calculate_metrics(model.conv1x1.weight)
                    epoch_metrics["conv1x1_l2"] = l2
                    epoch_metrics["conv1x1_sparsity"] = spars
                
                # Interactive conv in embeddings
                interactive_conv = None
                if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings') and hasattr(model.bert.embeddings, 'interactive_conv'):
                    interactive_conv = model.bert.embeddings.interactive_conv
                elif hasattr(model, 'model') and hasattr(model.model, 'bert') and hasattr(model.model.bert, 'embeddings'):
                    if hasattr(model.model.bert.embeddings, 'interactive_conv'):
                        interactive_conv = model.model.bert.embeddings.interactive_conv

                if interactive_conv is not None:
                    l2, spars = calculate_metrics(interactive_conv.weight)
                    epoch_metrics["interactive_conv_l2"] = l2
                    epoch_metrics["interactive_conv_sparsity"] = spars
                    # 添加额外的统计指标
                    with torch.no_grad():
                        weight = interactive_conv.weight
                        epoch_metrics["interactive_conv_weight_mean"] = weight.abs().mean().item()
                        epoch_metrics["interactive_conv_weight_max"] = weight.abs().max().item()

        metrics.append(epoch_metrics)

        save_cls_features(model, feature_loader, save_dir, epoch, device)
        print(f"Saved features for epoch {epoch + 1}")

        try:
            metrics_path = os.path.join(save_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics to {metrics_path}: {e}")

    print("Training completed!")
    final_metrics_path = os.path.join(save_dir, "training_metrics.json")
    print(f"Final metrics saved to: {final_metrics_path}")

if __name__ == "__main__":
    main()