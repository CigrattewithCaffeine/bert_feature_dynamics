import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import glob
import torch
import json

def visualize_features(features_dict, labels, save_dir, epoch=None, model_name=None):
    """single layer visualization
    Args:
        features_dict: dict, {layer: features}
        labels: label list
        save_dir: save directory
        epoch: epoch number
        model_name: model name
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # differnt colors for labels, color blind friendly
    color_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
    unique_labels = np.unique(labels)
    
    num_layers = len(features_dict)
    rows = int(np.ceil(np.sqrt(num_layers)))
    cols = int(np.ceil(num_layers / rows))
    
    plt.figure(figsize=(5*cols, 4*rows),num=None)
    
    for layer, features in features_dict.items():
        # implement PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        
        # draw scatter plot
        plt.subplot(rows, cols, layer + 1)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color_idx = i % len(color_palette)
            plt.scatter(
                reduced_features[mask, 0], 
                reduced_features[mask, 1],
                c=color_palette[color_idx],
                label=f'Label {label}',
                alpha=0.6,
                edgecolors='w',
                linewidth=0.8,
                s=80
            )
        
        plt.title(f"Layer {layer} CLS Features")
        if layer % cols == 0:  
            plt.ylabel("PCA Component 2")
        if layer >= (rows-1) * cols:  
            plt.xlabel("PCA Component 1")
        
        #explained_var = pca.explained_variance_ratio_
        #plt.legend(title=f"Explained: {explained_var[0]:.2f}, {explained_var[1]:.2f}")
        
        plt.grid(alpha=0.3)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    title = f"{model_name}_epoch_{epoch}" if epoch is not None and model_name is not None else "features"
    plt.savefig(os.path.join(save_dir, f"{title}.png"), dpi=300, bbox_inches='tight')
    print(f"plots have been saved in: {os.path.join(save_dir, f'{title}.png')}")
    plt.close()



def run_feature_visualization(
    feature_dir: str = None,
    output_dir: str = None,
    pca_dim: int = 2,
    figsize=(10, 8), 
    random_state: int = 42,
    model_name: str = None,
    max_samples=None
):
    """
    pca and visualize CLS hidden states

    """
    if feature_dir is None:
        if model_name is None:
            raise ValueError("When feature_dir is None, model_name must be provided.")
        saved_root = "saved_features"
        candidates = [d for d in os.listdir(saved_root) if d.startswith(model_name)]
        if not candidates:
            raise FileNotFoundError(f"fail to find model_name : {model_name}")
        feature_dir = os.path.join(saved_root, sorted(candidates)[-1])

    if output_dir is None:
        output_dir = os.path.join("visualizations", feature_dir.replace("saved_features/", ""))

    os.makedirs(output_dir, exist_ok=True)

    feature_files = [f for f in os.listdir(feature_dir) if f.endswith(".npy") and not f.endswith("_labels.npy")]

    color_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]

    epoch_layer_files = {}
    for fname in feature_files:
        base_name = os.path.splitext(fname)[0]
        parts = base_name.split("_")
        epoch = None
        layer = None
        for part in parts:
            if part.startswith("layer"):
                layer = int(part[5:])
            if part.startswith("epoch"):
                epoch = int(part[5:])
        if epoch is not None and layer is not None:
            if epoch not in epoch_layer_files:
                epoch_layer_files[epoch] = []
            epoch_layer_files[epoch].append((layer, fname))

    all_visualization_metadata = []
    pca_explained = {}

    for epoch, files in tqdm(epoch_layer_files.items(), desc="Processing epochs"):
        epoch_visualizations = []
        pca_explained[epoch] = {}

        for layer, fname in tqdm(files, desc=f"Processing files for epoch {epoch}"):
            path = os.path.join(feature_dir, fname)
            features = np.load(path)
            if max_samples is not None:
                features = features[:max_samples]
            if features.ndim != 2:
                print(f"skip {fname}: shape not 2D.")
                continue

            label_candidates = [f for f in os.listdir(feature_dir) if f.endswith(f"labels_epoch{epoch}.npy")]
            if label_candidates:
                labels_path = os.path.join(feature_dir, label_candidates[0])
                labels = np.load(labels_path)
                if max_samples is not None:
                    labels = labels[:max_samples]
            else:
                n_samples = features.shape[0]
                labels = np.zeros(n_samples, dtype=int)
                labels[n_samples//2:] = 1

            pca = PCA(n_components=pca_dim, random_state=random_state)
            features_pca = pca.fit_transform(features)

            pca_explained[epoch][layer] = pca.explained_variance_ratio_[:2].tolist()

            unique_labels = np.unique(labels)

            plt.figure(figsize=figsize, num=None)
            for i, label in enumerate(unique_labels):
                mask = labels == label
                color_idx = i % len(color_palette)
                plt.scatter(
                    features_pca[mask, 0], 
                    features_pca[mask, 1],
                    c=color_palette[color_idx],
                    label=f'Label {label}',
                    alpha=0.6,
                    edgecolors='w',
                    linewidth=0.8,
                    s=80
                )

            model_info = f"{model_name}, " if model_name else ""
            plt.title(f'Layer {layer}, Epoch {epoch}', fontsize=14)
            plt.legend(loc='best')
            plt.grid(alpha=0.3)
            plt.xlabel(f"PC1", fontsize=12)
            plt.ylabel(f"PC2", fontsize=12)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            model_prefix = f"{model_name}_" if model_name else ""
            save_path = os.path.join(output_dir, f'{model_prefix}layer{layer}_epoch{epoch}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            epoch_visualizations.append({
                'layer': layer,
                'epoch': epoch,
                'pca_data_file': path,
                'labels_file': labels_path if 'labels_path' in locals() else None,
                'unique_labels': unique_labels.tolist(),
                'model_name': model_name
            })

        all_visualization_metadata.extend(epoch_visualizations)

    with open(os.path.join(output_dir, "pca_explained_variance.json"), "w") as f:
        json.dump(pca_explained, f, indent=2)

    print(f"PCA explained variance save to: {os.path.join(output_dir, 'pca_explained_variance.json')}")
    create_summary_visualization(
        visualizations_metadata=all_visualization_metadata,
        color_palette=color_palette,
        output_dir=output_dir,
        max_samples=max_samples
    )
    print(f"All visualization plots saved to: {output_dir}")

def create_summary_visualization(visualizations_metadata, color_palette, output_dir, max_samples=None):
    """
    visualize all layers and epochs in a summary plot
    """
    if not visualizations_metadata:
        print("no data to visualize")
        return
    
    # 获取所有独特的层和epoch
    layers = sorted(list(set([v['layer'] for v in visualizations_metadata])))
    epochs = sorted(list(set([v['epoch'] for v in visualizations_metadata])))
    
    n_layers = len(layers)
    n_epochs = len(epochs)
    
    model_name = visualizations_metadata[0].get('model_name', '')
    model_prefix = f"{model_name}_" if model_name else ""

    fig = plt.figure(figsize=(n_epochs * 3, n_layers * 2.5), num=None)
    gs = gridspec.GridSpec(n_layers, n_epochs, figure=fig)

    for i, layer in enumerate(layers):
        for j, epoch in enumerate(epochs):
            vis_metadata = None
            for v in visualizations_metadata:
                if v['layer'] == layer and v['epoch'] == epoch:
                    vis_metadata = v
                    break
            if vis_metadata is None:
                continue
            ax = fig.add_subplot(gs[i, j])
            features = np.load(vis_metadata['pca_data_file'])
            if max_samples is not None:
                features = features[:max_samples]
            if vis_metadata['labels_file'] and os.path.exists(vis_metadata['labels_file']):
                labels = np.load(vis_metadata['labels_file'])
                if max_samples is not None:
                    labels = labels[:max_samples]
            else:
                n_samples = features.shape[0]
                labels = np.zeros(n_samples, dtype=int)
                labels[n_samples//2:] = 1
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features)
            unique_labels = np.array(vis_metadata['unique_labels'])
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                color_idx = idx % len(color_palette)
                ax.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    c=color_palette[color_idx],
                    alpha=0.6,  
                    edgecolors='w',
                    linewidth=0.5,  
                    s=20  
                )
            if i == n_layers - 1:  
                ax.set_xlabel("PC1", fontsize=8)
            if j == 0:  
                ax.set_ylabel("PC2", fontsize=8)
            ax.set_title(f'Layer{layer},Epoch{epoch}', fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=6)
            del features, labels, features_pca
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_prefix}summary_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    save_path_hires = os.path.join(output_dir, f'{model_prefix}summary_visualization_hires.png')
    plt.savefig(save_path_hires, dpi=1200, bbox_inches='tight')
    
    plt.close()
    print(f"visualization summary saved to: {save_path}")


def create_comparison_frame(model_dirs, output_dir, model_names=None):
    """
    Create a comparison frame for multiple models.
    """
    resolved_dirs = []
    for name in model_dirs:
        if os.path.exists(name):
            resolved_dirs.append(name)
        else:
            saved_root = "saved_features"
            candidates = [d for d in os.listdir(saved_root) if d.startswith(name)]
            if not candidates:
                raise FileNotFoundError(f"file not found: {name}")
            resolved_dirs.append(os.path.join(saved_root, sorted(candidates)[-1]))

    model_dirs = resolved_dirs
    os.makedirs(output_dir, exist_ok=True)

    if model_names is None:
        model_names = [os.path.basename(d) for d in model_dirs]
    
    if model_names is None:
        model_names = [os.path.basename(d) for d in model_dirs]
    all_layer_epoch_pairs = set()
    
    for i, model_dir in enumerate(model_dirs):
        feature_pattern = os.path.join(model_dir, "layer*_epoch*.npy")
        if model_names[i]:
            feature_pattern = os.path.join(model_dir, f"{model_names[i]}_layer*_epoch*_features.npy")
        
        feature_files = glob.glob(feature_pattern)
        
        for f in feature_files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            if model_names[i]:
                #  "ModelName_layer0_epoch0_features.npy"
                layer = int(parts[1][5:])
                epoch = int(parts[2][5:])
            else:
                #  "layer0_epoch0_features.npy"
                layer = int(parts[0][5:])
                epoch = int(parts[1][5:])
            
            all_layer_epoch_pairs.add((layer, epoch))
    color_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
    for layer, epoch in tqdm(sorted(all_layer_epoch_pairs), desc="creating comparison plots"):
        plt.figure(figsize=(5 * len(model_dirs), 5), num=None)
        
        for i, model_dir in enumerate(model_dirs):
            model_name = model_names[i] if model_names else os.path.basename(model_dir)
            if model_name:
                file_path = os.path.join(model_dir, f"{model_name}_layer{layer}_epoch{epoch}_features.npy")
                labels_path = os.path.join(model_dir, f"{model_name}_epoch{epoch}_labels.npy")
            else:
                file_path = os.path.join(model_dir, f"layer{layer}_epoch{epoch}_features.npy")
                labels_path = os.path.join(model_dir, f"epoch{epoch}_labels.npy")
                
            if not os.path.exists(file_path):
                print(f"file not exist: {file_path}")
                continue
                
            features = np.load(file_path)
            if os.path.exists(labels_path):
                labels = np.load(labels_path)
                print(f"using true labels: {labels_path}")
            else:
                n_samples = features.shape[0]
                labels = np.zeros(n_samples, dtype=int)
                labels[n_samples//2:] = 1
                print(f"label file not found, using default labels: {labels_path}")
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features)
            unique_labels = np.unique(labels)
            ax = plt.subplot(1, len(model_dirs), i+1)
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                color_idx = idx % len(color_palette)
                plt.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    c=color_palette[color_idx],
                    label=f'Label {label}',
                    alpha=0.6,
                    edgecolors='w',
                    linewidth=0.8,
                    s=60
                )
            plt.title(f'{model_name}, Layer {layer}, Epoch {epoch}', fontsize=12)
            plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.2f})", fontsize=10)
            plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.2f})", fontsize=10)
            plt.legend(loc='best', fontsize=8)
            plt.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            del features, labels, features_pca
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'comparison_layer{layer}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"all comparing plots saved to: {output_dir}")

# NOTE: not in use now，saved for future probing.py 
def extract_features(model, dataloader, device):
    """extract features from model"""
    model.eval()
    all_features = {}
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states    
            if not all_features:
                for i in range(len(hidden_states)):
                    all_features[i] = []
            for layer, layer_output in enumerate(hidden_states):
                cls_vec = layer_output[:, 0, :]  # [batch_size, hidden_size]
                all_features[layer].append(cls_vec.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
    for layer in all_features:
        all_features[layer] = np.vstack(all_features[layer])
    all_labels = np.concatenate(all_labels)
    return all_features, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='feature visualization')
    parser.add_argument('--feature_dir', type=str, default=None, required=False, help='feature directory')
    parser.add_argument('--output_dir', type=str, default=None, required=False, help='output directory')
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to visualize')
    parser.add_argument('--compare', action='store_true', help='if compare models')
    parser.add_argument('--dirs', nargs='+', default=[], help='compare models directories')
    parser.add_argument('--names', nargs='+', default=[], help='compare models names')
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.dirs:
            print("error: --dirs must be provided for comparison.")
            exit(1)
        create_comparison_frame(args.dirs, args.output_dir, args.names if args.names else None)
    else:
        run_feature_visualization(
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_samples=args.max_samples
        )
