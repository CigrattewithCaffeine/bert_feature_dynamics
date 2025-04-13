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

# Helper function to calculate metrics - placed here for clarity
def calculate_metrics(param):
    """Calculates L2 norm and sparsity for a given parameter tensor."""
    if param is None or param.numel() == 0:
        return 0.0, 1.0 # Or handle as appropriate
    l2_norm = torch.norm(param.data.detach(), p=2).item()
    sparsity = (param.data.detach() == 0).float().mean().item()
    return l2_norm, sparsity

def train_epoch(model, train_loader, optimizer, scheduler, device, model_type, epoch): # Added model_type and epoch for conditional printing
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")

    # Check if model is Conv2D type ONCE before the loop for efficiency
    is_conv2d_model = isinstance(model, (Conv2DBertForSequenceClassification, Conv2DBertBaseForSequenceClassification))
    # Determine if we should print gradients (e.g., only first batch of first epoch)
    print_grads_this_epoch = epoch == 0 # Example: Print only during the first epoch

    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # --- Gradient Debugging Prints ---
        # Only print for the first batch of the first epoch to avoid excessive output
        if print_grads_this_epoch and batch_idx == 0:
            print("\n--- Gradient Debug Info (Epoch 0, Batch 0) ---")
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
            if is_conv2d_model and hasattr(model, 'conv1x1') and hasattr(model.conv1x1, 'weight'):
                conv_weight = model.conv1x1.weight
                if conv_weight.grad is not None:
                    # Ensure tensor has enough elements before accessing
                    if conv_weight.numel() > 1:
                         # Safely get elements, handle potential dimension mismatch if kernel isn't 4D
                        try:
                             w_val_0 = conv_weight.data.view(-1)[0].item()
                             g_val_0 = conv_weight.grad.view(-1)[0].item()
                             w_val_1 = conv_weight.data.view(-1)[1].item()
                             g_val_1 = conv_weight.grad.view(-1)[1].item()
                             print(f"  Conv2D 1x1 Weight[0]: {w_val_0:.4f}, Grad[0]: {g_val_0:.4f}")
                             print(f"  Conv2D 1x1 Weight[1]: {w_val_1:.4f}, Grad[1]: {g_val_1:.4f}")
                        except IndexError:
                            print("  Conv2D 1x1: Could not access specific weight/grad elements (tensor too small?).")
                        except Exception as e:
                             print(f"  Conv2D 1x1: Error accessing specific weight/grad elements: {e}")
                    else:
                         print("  Conv2D 1x1: Weight tensor too small for detailed print.")
                else:
                    print("  Conv2D 1x1 Grad: N/A (or frozen)")
            elif is_conv2d_model:
                 print("  Conv2D 1x1 Layer or Weight: Not found (attribute 'conv1x1.weight' missing?)")
            print("--- End Gradient Debug Info ---")
        # --- End Gradient Debugging Prints ---

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

    # Note: Residual weight print kept as it was in the original evaluate function
    if hasattr(model, "model") and hasattr(model.model, "bert") and hasattr(model.model.bert, "embeddings"):
        if hasattr(model.model.bert.embeddings, "residual_weight"):
            # Check if it's a tensor before calling .item()
            res_weight = model.model.bert.embeddings.residual_weight
            if torch.is_tensor(res_weight):
                 print(f"Residual weight: {res_weight.item():.4f}")
            else:
                 print(f"Residual weight: {res_weight:.4f}") # Assuming it might already be a float

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

            # Check if hidden_states is None or empty before proceeding
            if hidden_states:
                for layer_idx, layer_output in enumerate(hidden_states):
                    cls_features = layer_output[:, 0, :].cpu().numpy()
                    features_dict.setdefault(layer_idx, []).append(cls_features)
            else:
                # Handle case where hidden states are not output
                print("Warning: Hidden states not available in model output.")


            # Check if attentions is None or empty before proceeding
            if attentions:
                for layer_idx, attn in enumerate(attentions):
                    mean_attn = attn[:, :, 0, :].mean(dim=1).cpu().numpy()
                    attention_dict.setdefault(layer_idx, []).append(mean_attn)
            else:
                 # Handle case where attention weights are not output
                 print("Warning: Attention weights not available in model output.")


            labels_list.append(labels)

    os.makedirs(save_dir, exist_ok=True)
    if features_dict: # Only save if features were extracted
        for layer_idx in features_dict:
            features = np.concatenate(features_dict[layer_idx], axis=0)
            np.save(os.path.join(save_dir, f"layer{layer_idx}_epoch{epoch}.npy"), features)

    if attention_dict: # Only save if attentions were extracted
        for layer_idx in attention_dict:
            attn_scores = np.concatenate(attention_dict[layer_idx], axis=0)
            np.save(os.path.join(save_dir, f"attention_layer{layer_idx}_epoch{epoch}.npy"), attn_scores)

    if labels_list: # Only save if labels were collected
        labels = np.concatenate(labels_list, axis=0)
        np.save(os.path.join(save_dir, f"labels_epoch{epoch}.npy"), labels)

def get_model(model_type):
    # Assuming these model paths/names are correct and classes are defined
    if model_type == "conv2D_pretrained":
        # Ensure 'num_labels' is passed correctly if needed by from_pretrained
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
    """冻结BERT模型的指定层数，并加入更详细的调试打印"""
    print(f"\n--- Entering freeze_bert_layers ---")
    print(f"Attempting to freeze {num_layers_to_freeze} encoder layers.")
    print(f"Parameter freeze_embeddings: {freeze_embeddings}")

    # 检查模型结构，获取核心 bert 模型
    bert_model = None
    # Try finding bert directly or nested within 'model'
    if hasattr(model, "bert"):
        bert_model = model.bert
        print("Accessing model parameters via model.bert")
    elif hasattr(model, "model") and hasattr(model.model, "bert"): # Common in custom wrappers
        bert_model = model.model.bert
        print("Accessing model parameters via model.model.bert")
    else:
        print("[WARNING] Cannot find standard bert model structure (model.bert or model.model.bert). Freeze might not work as expected.")
        # Attempt to proceed assuming model *is* the bert model or has compatible structure
        # This might need adjustment based on your actual model definitions
        if hasattr(model, 'embeddings') and hasattr(model, 'encoder'):
             print("Assuming model itself has 'embeddings' and 'encoder' attributes.")
             bert_model = model
        else:
             print("[ERROR] Could not identify BERT components. Aborting freeze.")
             return # Exit if we can't find the structure

    # Check if bert_model was successfully identified before proceeding
    if bert_model is None:
         print("[ERROR] bert_model identification failed. Aborting freeze.")
         return

    # 根据参数决定是否冻结嵌入层
    print("\nProcessing Embeddings Layer...")
    if hasattr(bert_model, 'embeddings'): # Check if embeddings exist
        if freeze_embeddings:
            print("  Action: Freezing embeddings...")
            count_emb_frozen = 0
            for name, param in bert_model.embeddings.named_parameters():
                param.requires_grad = False
                # print(f"    Frozen: embeddings.{name}") # Uncomment for verbose detail
                count_emb_frozen += 1
            print(f"  Result: Embeddings layer frozen ({count_emb_frozen} parameters).")
        else:
            print("  Action: Keeping embeddings trainable.")
            count_emb_trainable = 0
            for name, param in bert_model.embeddings.named_parameters():
                param.requires_grad = True
                # print(f"    Trainable: embeddings.{name}") # Uncomment for verbose detail
                count_emb_trainable += 1
            print(f"  Result: Embeddings layer kept trainable ({count_emb_trainable} parameters).")
    else:
         print(" [WARNING] No 'embeddings' attribute found in bert_model. Skipping embedding freeze/unfreeze.")


    # 冻结编码器层
    print("\nProcessing Encoder Layers...")
    if hasattr(bert_model, 'encoder') and hasattr(bert_model.encoder, 'layer'): # Check if encoder and layers exist
        num_actual_layers = len(bert_model.encoder.layer)
        # Use bert_model.config if available, otherwise try to get layer count directly
        config_num_layers = getattr(getattr(bert_model, 'config', None), 'num_hidden_layers', num_actual_layers)

        actual_layers_to_freeze = min(config_num_layers, num_layers_to_freeze) # Ensure not exceeding actual layers
        print(f"  Model reports {config_num_layers} layers. Attempting to freeze first {actual_layers_to_freeze} (0 to {actual_layers_to_freeze - 1})...")

        for i in range(actual_layers_to_freeze):
            if i < num_actual_layers: # Double check index is valid for the list
                count_layer_frozen = 0
                print(f"  Freezing Encoder layer {i}...")
                for name, param in bert_model.encoder.layer[i].named_parameters():
                    param.requires_grad = False
                    # print(f"    Frozen: encoder.layer.{i}.{name}") # Uncomment for verbose detail
                    count_layer_frozen += 1
                print(f"    Layer {i} frozen ({count_layer_frozen} parameters).")
            else:
                print(f"  Warning: Attempted to freeze layer {i}, but model only has {num_actual_layers} layers.")
                break # Stop if index goes out of bounds
    else:
         print(" [WARNING] No 'encoder' or 'encoder.layer' attribute found in bert_model. Skipping encoder layer freeze.")


    # 再次打印所有参数的最终梯度状态以供核对
    print("\n--- Final Parameter Gradient Status Check ---")
    trainable_params_count = 0
    frozen_params_count = 0
    for name, param in model.named_parameters(): # Iterate over the original top-level model
        status = "Trainable" if param.requires_grad else "FROZEN"
        # print(f"  {name}: {status}") # Uncomment for exhaustive list
        if param.requires_grad:
            trainable_params_count += param.numel()
        else:
            frozen_params_count += param.numel()
    print(f"Total Trainable Parameters: {trainable_params_count}")
    print(f"Total Frozen Parameters: {frozen_params_count}")
    print(f"--- Exiting freeze_bert_layers ---\n")


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
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument("--freeze_layers", type=int, default=0, help="Number of layers to freeze during warmup (0-12)")
    parser.add_argument("--freeze_embeddings", type=int, default=0, help="freeze embeddings layer or not (1 for True, 0 for False)") # Clarified help text
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading SST2 dataset...")
    sst2_splits = load_sst2()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = create_dataloader(sst2_splits['train'], tokenizer, batch_size=args.batch_size, shuffle=True)
    val_loader = create_dataloader(sst2_splits['dev'], tokenizer, batch_size=args.batch_size)
    # Ensure feature_loader uses dev split as per original code, no shuffle
    feature_loader = create_dataloader(sst2_splits['dev'], tokenizer, batch_size=args.batch_size, shuffle=False)


    print(f"Initializing model: {args.model_type}...")
    model = get_model(args.model_type)
    model = model.to(device)

    # 首先冻结指定的层
    if args.warmup_epochs > 0 and (args.freeze_layers > 0 or args.freeze_embeddings == 1): # Freeze if layers > 0 OR embeddings flag is set
        print(f"Starting with {args.freeze_layers} layers frozen and embeddings frozen={args.freeze_embeddings==1} for {args.warmup_epochs} epochs")
        freeze_emb = args.freeze_embeddings == 1 # Convert int to bool
        freeze_bert_layers(model, args.freeze_layers, freeze_embeddings=freeze_emb)
    elif args.freeze_layers > 0 or args.freeze_embeddings == 1:
         print(f"[Warning] freeze_layers or freeze_embeddings is set, but warmup_epochs is 0. Layers will be frozen for the entire training unless unfrozen manually.")
         freeze_emb = args.freeze_embeddings == 1
         freeze_bert_layers(model, args.freeze_layers, freeze_embeddings=freeze_emb)


    # Filter parameters for optimizer *after* potential freezing
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_epochs
    # Adjust scheduler steps if some epochs are warm-up with frozen layers?
    # Current setup keeps total_steps based on all epochs. This is usually fine.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    # Ensure the base directory exists or handle potential errors
    try:
        # Construct the full path including the potential Drive mount point
        # Using os.path.join for better cross-platform compatibility
        base_output_dir = "/content/drive/MyDrive/bert_feature_outputs" # Base directory
        save_dir = os.path.join(base_output_dir, f"{args.model_type}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output will be saved to: {save_dir}")
    except Exception as e:
        print(f"Error creating save directory '{save_dir}'. Please check permissions or path. Error: {e}")
        # Decide how to handle: exit, use local path, etc.
        print("Using local directory './bert_feature_outputs' as fallback.")
        base_output_dir = "./bert_feature_outputs"
        save_dir = os.path.join(base_output_dir, f"{args.model_type}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)


    metrics = []
    is_conv2d_model_main = isinstance(model, (Conv2DBertForSequenceClassification, Conv2DBertBaseForSequenceClassification)) # Check once in main

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # If warming up and layers were frozen, unfreeze after warmup epochs
        if epoch == args.warmup_epochs and args.warmup_epochs > 0 and (args.freeze_layers > 0 or args.freeze_embeddings == 1):
            print(f"\n--- Epoch {epoch}: Warmup complete, unfreezing layers ---")
            unfreeze_all(model)
            # Re-create optimizer with *all* parameters (now trainable)
            print("Re-initializing optimizer for all parameters...")
            optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            # Re-create scheduler for the remaining steps
            remaining_steps = len(train_loader) * (args.num_epochs - epoch)
            print(f"Re-initializing scheduler for remaining {remaining_steps} steps...")
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=remaining_steps)
            print("--- Unfreezing complete ---")

        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        # Pass model_type and epoch to train_epoch for conditional grad printing
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.model_type, epoch)
        val_loss, accuracy = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        # --- L2 Norm and Sparsity Calculation ---
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
        }
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
                 # Add residual_weight if it exists (as in original code)
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
            if is_conv2d_model_main and hasattr(model, 'conv1x1') and hasattr(model.conv1x1, 'weight'):
                 l2, spars = calculate_metrics(model.conv1x1.weight)
                 epoch_metrics["conv1x1_l2"] = l2
                 epoch_metrics["conv1x1_sparsity"] = spars
        # --- End L2/Sparsity Calculation ---

        metrics.append(epoch_metrics) # Append the detailed metrics

        # Save features (call kept as is)
        save_cls_features(model, feature_loader, save_dir, epoch, device)
        print(f"Saved features for epoch {epoch + 1}")


        # Save metrics to JSON file
        try:
             metrics_path = os.path.join(save_dir, "training_metrics.json")
             with open(metrics_path, "w") as f:
                 json.dump(metrics, f, indent=2)
        except Exception as e:
             print(f"Error saving metrics to {metrics_path}: {e}")


    print("Training completed!")
    # Optionally print final metrics path
    final_metrics_path = os.path.join(save_dir, "training_metrics.json")
    print(f"Final metrics saved to: {final_metrics_path}")


if __name__ == "__main__":
    main()