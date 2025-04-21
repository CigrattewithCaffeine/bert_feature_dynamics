import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import random 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import glob
import re
import torch
import textstat  

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from utils.data_utils import load_sst2
    from models.BaseBert import BaseBertBaseForSequenceClassification
    from models.ConvBert import Conv2DBertBaseForSequenceClassification
    from models.FFTBert import FFTBertBaseForSequenceClassification
    from transformers import BertTokenizer, BertConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_probe(X, y, probe_type='classification', random_state=42, n_splits=5):
    """
    Trains and evaluates a probe using cross-validation.
    """
    if X is None or y is None or X.shape[0] != y.shape[0] or X.shape[0] < n_splits:
        return {"error": "Invalid input data"}

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = {}

    if probe_type == 'classification':
        if len(np.unique(y)) < 2: 
            return {"accuracy": 0.0, "warning": "Not enough classes"}
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver='lbfgs', max_iter=1000, random_state=random_state, 
                               C=0.1, class_weight='balanced')
        )
        cv_scores = cross_val_score(probe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        scores['accuracy'] = np.mean(cv_scores)

    elif probe_type == 'regression':
        probe = make_pipeline(StandardScaler(), LinearRegression())
        r2_scores = cross_val_score(probe, X, y, cv=cv, scoring='r2', n_jobs=-1)
        neg_mse_scores = cross_val_score(probe, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        scores['r2'] = np.mean(r2_scores)
        scores['neg_mse'] = np.mean(neg_mse_scores)
    else:
        raise ValueError("probe_type must be 'classification' or 'regression'")

    return scores

def get_model_instance_probe(model_type, config, checkpoint_path=None, device='cpu'):
    """Instantiates a model, loads checkpoint ONLY IF provided."""
    model_map = {
        'base': BaseBertBaseForSequenceClassification,
        'conv2d': Conv2DBertBaseForSequenceClassification,
        'fft': FFTBertBaseForSequenceClassification
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported model_type for probing: {model_type}")
    
    print(f"Instantiating {model_type} model (for probe feature extraction)...")
    model = model_map[model_type](config)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"    Warning: Error loading checkpoint for {model_type}: {e}. Using initialized weights.")
    elif checkpoint_path:
        print(f"    Warning: Checkpoint path not found: {checkpoint_path}. Using initialized weights.")
    else:
        print(f"    No checkpoint provided. Using initialized weights (Random Initial State).")

    model.to(device)
    model.eval()
    return model

def extract_cls_hidden_states(model, dataloader, layer_index, device):
    """Extracts [CLS] hidden states for a specific layer from a dataloader."""
    cls_states = []
    print(f"Extracting [CLS] states for layer {layer_index}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Layer {layer_index} Extraction", leave=False):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            try:
                outputs = model(**inputs, output_hidden_states=True, output_attentions=False)
                hidden_states = outputs.hidden_states
                
                if hidden_states and len(hidden_states) > layer_index:
                    cls_output = hidden_states[layer_index][:, 0, :].detach().cpu().numpy()
                    cls_states.append(cls_output)
                else:
                    print(f"Warning: Layer {layer_index} hidden state not found in outputs.")
                    return None
            except Exception as e:
                print(f"Error during hidden state extraction: {e}")
                return None

    return np.concatenate(cls_states, axis=0) if cls_states else None

def find_feature_files(feature_dir, layers_to_probe, epochs_to_probe):
    """Finds feature and label files matching the specified layers and epochs."""
    found_files = {}  # Structure: {epoch: {'labels': path, 'layers': {layer: path}}}
    print(f"Searching for features in: {feature_dir}")
    
    # Find label files
    label_files = glob.glob(os.path.join(feature_dir, "labels_epoch*.npy"))
    epoch_label_map = {}
    
    for f_path in label_files:
        match = re.search(r'labels_epoch(\d+)\.npy', os.path.basename(f_path))
        if match:
            epoch = int(match.group(1))
            if epochs_to_probe is None or epoch in epochs_to_probe:
                epoch_label_map[epoch] = f_path
                found_files[epoch] = {'labels': f_path, 'layers': {}}
    
    if not epoch_label_map:
        return {}
    
    # Find layer files for each epoch
    layer_files = glob.glob(os.path.join(feature_dir, "layer*_epoch*.npy"))
    for f_path in layer_files:
        match = re.search(r'layer(\d+)_epoch(\d+)\.npy', os.path.basename(f_path))
        if match:
            layer = int(match.group(1))
            epoch = int(match.group(2))
            if epoch in epoch_label_map and (layers_to_probe is None or layer in layers_to_probe):
                found_files[epoch]['layers'][layer] = f_path
    
    # Validate files
    valid_epochs = list(found_files.keys())
    for epoch in valid_epochs:
        if not found_files[epoch]['labels'] or not found_files[epoch]['layers']:
            print(f"Warning: Missing label or layer files for epoch {epoch} in {feature_dir}. Skipping.")
            del found_files[epoch]
    
    print(f"Found data for epochs: {sorted(list(found_files.keys()))} in {feature_dir}")
    return found_files

def parse_range_or_list(arg_str):
    """Parses a string like '0,1,5' or '0-5,10,12' into a set of integers."""
    if not arg_str:
        return None
    
    indices = set()
    parts = arg_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    raise ValueError(f"Invalid range: {start}-{end}")
                indices.update(range(start, end + 1))
            except ValueError as e:
                print(f"Warning: Could not parse range '{part}'. Skipping. Error: {e}")
        else:
            try:
                indices.add(int(part))
            except ValueError:
                print(f"Warning: Could not parse integer '{part}'. Skipping.")
    
    return indices if indices else None

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            return_tensors='pt', 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

def main():
    parser = argparse.ArgumentParser(description="Run probing analysis on saved [CLS] hidden states.")
    parser.add_argument("--feature_base_dir", type=str, required=True, 
                        help="Base directory containing subfolders for each model run.")
    parser.add_argument("--model_runs", type=str, required=True, nargs='+', 
                        help="List of *trained* model run subfolder names to analyze.")
    parser.add_argument("--model_types", type=str, required=True, nargs='+', 
                        help="Corresponding model types ('base', 'conv2d', 'fft') for each run.")
    parser.add_argument("--data_dir", type=str, default="../data/sst2", 
                        help="Directory containing the SST-2 dataset.")
    parser.add_argument("--layers", type=str, default="0,6,12", 
                        help="Comma-separated list/ranges of layer indices to probe.")
    parser.add_argument("--epochs", type=str, default=None, 
                        help="Comma-separated list/ranges of epoch indices to probe.")
    parser.add_argument("--output_file", type=str, default="./probing_results.json", 
                        help="Path to save the probing results (JSON format).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use ('cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for extracting initial hidden states.")
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Max sequence length for tokenizer.")

    args = parser.parse_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Parse layers/epochs
    layers_to_probe = parse_range_or_list(args.layers) or set(range(13))  # Default 0-12 if not specified
    epochs_to_probe = parse_range_or_list(args.epochs)  # None means probe all found epochs

    # Load data
    print("Loading SST-2 data...")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased')

        # Load train and dev splits
        sst2_splits = load_sst2()
        train_texts = sst2_splits['train']['text'].tolist()
        dev_texts = sst2_splits['dev']['text'].tolist()
        dev_labels_sentiment = sst2_splits['dev']['label'].to_numpy()

        # Calculate FKGL scores
        print("Calculating Flesch-Kincaid Grade Level scores for dev set...")
        dev_fkgl_scores = np.array([textstat.flesch_kincaid_grade(text) for text in tqdm(dev_texts, desc="FKGL Calc")])

        # Create TF-IDF features
        print("Calculating TF-IDF features for dev set...")
        vectorizer = TfidfVectorizer(max_features=5000)
        vectorizer.fit(train_texts)
        X_dev_tfidf = vectorizer.transform(dev_texts).toarray()
        print(f"  TF-IDF feature shape: {X_dev_tfidf.shape}")

        # Create dataloader for dev set
        dev_dataset = TextDataset(dev_texts, tokenizer, args.max_seq_length)
        dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size)

    except Exception as e:
        print(f"Error during data loading or preprocessing: {e}")
        sys.exit(1)

    # Extract initial hidden states
    initial_hidden_states = {}
    print("\n--- Extracting Initial Hidden States (Untrained Models) ---")
    for model_type in set(args.model_types):  # Use set to avoid duplicate extraction
        initial_hidden_states[model_type] = {}
        untrained_model = get_model_instance_probe(model_type, config, checkpoint_path=None, device=device)
        
        for layer in layers_to_probe:
            X_dev_initial = extract_cls_hidden_states(untrained_model, dev_dataloader, layer, device)
            if X_dev_initial is not None and X_dev_initial.shape[0] == len(dev_labels_sentiment):
                initial_hidden_states[model_type][layer] = X_dev_initial
            else:
                print(f"Warning: Initial hidden state extraction failed for {model_type} layer {layer}")
                initial_hidden_states[model_type][layer] = None
        
        del untrained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Main probing loop
    all_results = {}
    print("\n--- Starting Probing Analysis ---")

    if len(args.model_runs) != len(args.model_types):
        print("Error: Number of --model_runs must match number of --model_types.")
        sys.exit(1)

    # Run baseline probes once (TF-IDF features)
    print("Running Baseline Probes (TF-IDF Features)...")
    tfidf_sentiment_results = run_probe(X_dev_tfidf, dev_labels_sentiment, probe_type='classification', random_state=args.seed)
    tfidf_fkgl_results = run_probe(X_dev_tfidf, dev_fkgl_scores, probe_type='regression', random_state=args.seed)
    print(f" TF-IDF Sentiment Accuracy: {tfidf_sentiment_results.get('accuracy', 'Error'):.4f}")
    print(f"TF-IDF FKGL R^2: {tfidf_fkgl_results.get('r2', 'Error'):.4f}, Neg MSE: {tfidf_fkgl_results.get('neg_mse', 'Error'):.4f}")
    
    baseline_tfidf_results = {
        'sentiment_probe': tfidf_sentiment_results,
        'fkgl_probe': tfidf_fkgl_results
    }

    # Process each model run
    for model_run_name, model_type in zip(args.model_runs, args.model_types):
        print(f"\n--- Probing Model Run: {model_run_name} (Type: {model_type}) ---")
        model_feature_dir = os.path.join(args.feature_base_dir, model_run_name)
        all_results[model_run_name] = {'baseline_tfidf': baseline_tfidf_results}

        if not os.path.isdir(model_feature_dir):
            print(f"Warning: Directory not found '{model_feature_dir}'. Skipping.")
            continue

        run_files = find_feature_files(model_feature_dir, layers_to_probe, epochs_to_probe)
        if not run_files:
            print(f"No valid data files found for this run matching specified layers/epochs.")
            continue

        # Process each epoch
        for epoch in sorted(run_files.keys()):
            print(f"  Processing Epoch {epoch}...")
            epoch_str = f"epoch_{epoch}"
            all_results[model_run_name][epoch_str] = {}

            labels_path = run_files[epoch]['labels']
            layer_files = run_files[epoch]['layers']

            # Load labels and validate
            try:
                y_trained_sentiment = np.load(labels_path)
                if not np.array_equal(y_trained_sentiment, dev_labels_sentiment):
                    print(f"Warning: Labels mismatch for epoch {epoch}. Using expected dev labels.")
                if len(y_trained_sentiment) != len(dev_fkgl_scores):
                    print(f"Error: Label count ({len(y_trained_sentiment)}) doesn't match FKGL score count ({len(dev_fkgl_scores)}). Skipping epoch.")
                    continue
            except Exception as e:
                print(f"Error loading labels from {labels_path}: {e}. Skipping epoch {epoch}.")
                continue

            # Process each layer
            for layer in sorted(layer_files.keys()):
                layer_str = f"layer_{layer}"
                print(f"    Probing Layer {layer}...")
                all_results[model_run_name][epoch_str][layer_str] = {}
                feature_path = layer_files[layer]

                # Load trained hidden states
                try:
                    X_trained = np.load(feature_path)
                    if X_trained.shape[0] != len(dev_labels_sentiment):
                        print(f"Error: Trained hidden state count ({X_trained.shape[0]}) mismatch. Skipping layer {layer}.")
                        continue
                except Exception as e:
                    print(f"Error loading trained features from {feature_path}: {e}. Skipping layer {layer}.")
                    continue

                # Run baseline probe (initial hidden states)
                X_initial = initial_hidden_states.get(model_type, {}).get(layer)
                if X_initial is not None:
                    print(f"      Running Baseline Probe (Initial Hidden States)...")
                    initial_sentiment_results = run_probe(X_initial, dev_labels_sentiment, probe_type='classification', random_state=args.seed)
                    initial_fkgl_results = run_probe(X_initial, dev_fkgl_scores, probe_type='regression', random_state=args.seed)
                    
                    print(f"        Initial State Sentiment Accuracy: {initial_sentiment_results.get('accuracy', 'Error'):.4f}")
                    print(f"        Initial State FKGL R^2: {initial_fkgl_results.get('r2', 'Error'):.4f}, "
                          f"Neg MSE: {initial_fkgl_results.get('neg_mse', 'Error'):.4f}")
                    
                    all_results[model_run_name][epoch_str][layer_str]['baseline_initial_state'] = {
                        'sentiment_probe': initial_sentiment_results,
                        'fkgl_probe': initial_fkgl_results
                    }
                else:
                    print(f"      Skipping Baseline Probe (Initial Hidden States not available).")

                # Run main probe (trained hidden states)
                print(f"      Running Main Probe (Trained Hidden States)...")
                trained_sentiment_results = run_probe(X_trained, dev_labels_sentiment, probe_type='classification', random_state=args.seed)
                trained_fkgl_results = run_probe(X_trained, dev_fkgl_scores, probe_type='regression', random_state=args.seed)
                
                print(f"        Trained State Sentiment Accuracy: {trained_sentiment_results.get('accuracy', 'Error'):.4f}")
                print(f"        Trained State FKGL R^2: {trained_fkgl_results.get('r2', 'Error'):.4f}, "
                      f"Neg MSE: {trained_fkgl_results.get('neg_mse', 'Error'):.4f}")
                
                all_results[model_run_name][epoch_str][layer_str]['main_trained_state'] = {
                    'sentiment_probe': trained_sentiment_results,
                    'fkgl_probe': trained_fkgl_results
                }

    # Save results
    print(f"\nSaving probing results to {args.output_file}...")
    try:
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(o):
                if isinstance(o, np.generic):
                    return o.item()
                raise TypeError
            json.dump(all_results, f, indent=4, default=convert)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    print("\nProbing analysis finished.")

if __name__ == "__main__":
    main()