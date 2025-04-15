import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
from utils.data_utils import load_sst2
from models.BaseBert import BaseBertBaseForSequenceClassification
from models.ConvBert import Conv2DBertBaseForSequenceClassification
from models.FFTBert import FFTBertBaseForSequenceClassification
from scipy.stats import entropy as calculate_scipy_entropy
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disabling deterministic algorithms for performance if needed, but keep for consistency now
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_model_instance(model_type, config, checkpoint_path=None, device='cpu'):
    """Instantiates a model and loads checkpoint if provided."""
    model = None
    print(f"Initializing {model_type} model...")
    if model_type == 'base':
        model = BaseBertBaseForSequenceClassification(config)
    elif model_type == 'conv2d':
        model = Conv2DBertBaseForSequenceClassification(config)
    elif model_type == 'fft':
        model = FFTBertBaseForSequenceClassification(config)
    else:
        raise ValueError(f"Unsupported model_type for visualization: {model_type}")
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint for {model_type} from: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location=device)
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False) # Use strict=False initially if unsure
                print(f"Checkpoint loaded successfully for {model_type}.")
            except Exception as e:
                print(f"Error loading checkpoint for {model_type}: {e}. Using initialized model.")
        else:
            print(f"Warning: Checkpoint path not found for {model_type}: {checkpoint_path}. Using initialized model.")
    else:
         print(f"Warning: No checkpoint path provided for {model_type}. Using initialized model.")

    model.to(device)
    model.eval()
    return model

def get_embeddings(model, tokenizer, sentences, max_len, device):
    """Extracts embeddings after the embedding layer for given sentences."""
    all_embeddings = []
    all_tokens = []
    all_attention_masks = []

    print(f"Extracting embeddings using {model.__class__.__name__}...")
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Processing sentences"):
            inputs = tokenizer(sentence, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids)).to(device) 
            # --- Access the embedding module ---
            embedding_module = None
            if hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
                embedding_module = model.bert.embeddings
            elif hasattr(model, 'embeddings'): 
                 embedding_module = model.embeddings
            else:
                 print(f"Warning: Could not find embedding module in {model.__class__.__name__}. Skipping sentence.")
                 continue
            # --- End Access ---
            try:
                # Pass necessary inputs to the *embedding module's* forward method
                # Ensure all expected arguments by the specific embedding class are passed
                # Assuming standard BertEmbeddings or compatible signature
                embedding_output = embedding_module(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                )
                embeddings_np = embedding_output.detach().cpu().numpy()
                mask_np = attention_mask.cpu().numpy()
                seq_len = mask_np.sum()
                if seq_len > 0:
                    all_embeddings.append(embeddings_np[0, :seq_len, :]) # Get embeddings for this sentence
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].tolist())
                    all_tokens.extend(tokens) # Collect tokens corresponding to embeddings
                    all_attention_masks.append(mask_np[0, :seq_len]) # Collect masks
            except Exception as e:
                print(f"Error during embedding extraction for a sentence: {e}")
                continue

    if not all_embeddings:
        return None, None
    # Concatenate embeddings from all sentences
    # Need careful concatenation if sequence lengths vary
    try:
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        # final_mask = np.concatenate(all_attention_masks, axis=0) # Mask may not be needed further if we only selected non-pad
        print(f"Extracted {final_embeddings.shape[0]} token embeddings.")
        return final_embeddings, all_tokens
    except ValueError as e:
        print(f"Error concatenating embeddings, likely due to inconsistent shapes: {e}")
        return None, None
    
def plot_tsne(embeddings_dict, output_dir, perplexity=30, n_iter=1000, random_state=42):
    print("Generating combined t-SNE plot...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_embeddings_list = []
    model_labels = []
    model_types_present = [] 
    # model_types = sorted(list(embeddings_dict.keys()))
    model_types = ['base', 'fft', 'conv2d']
    print("Combining embeddings from different models...")
    for model_type in model_types:
        if model_type in embeddings_dict and embeddings_dict[model_type] is not None and embeddings_dict[model_type].shape[0] > 0:
            embeddings = embeddings_dict[model_type]
            num_tokens = embeddings.shape[0]
            print(f"  Adding {num_tokens} embeddings from {model_type}")
            all_embeddings_list.append(embeddings)
            model_labels.extend([model_type] * num_tokens)
            model_types_present.append(model_type)
        else:
            print(f"  Skipping {model_type} due to missing or empty embeddings.")

    if not all_embeddings_list:
        print("Error: No valid embeddings found for any model type. Cannot generate t-SNE plot.")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title('Combined t-SNE Plot - No Data')
        ax.text(0.5, 0.5, 'No Embedding Data Available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plot_path = os.path.join(output_dir, "tsne_embedding_comparison.png")
        plt.savefig(plot_path)
        plt.close(fig)
        return
    try:
        combined_embeddings = np.concatenate(all_embeddings_list, axis=0)
        print(f"Combined embedding shape: {combined_embeddings.shape}")
        if combined_embeddings.shape[0] <= perplexity:
             print(f"Warning: Number of samples ({combined_embeddings.shape[0]}) is less than or equal to perplexity ({perplexity}). Adjusting perplexity.")
             perplexity = max(5, combined_embeddings.shape[0] - 1) 
    except ValueError as e:
        print(f"Error concatenating embeddings: {e}. Check embedding shapes.")
        return
    # --- Perform t-SNE on Combined Data ---
    print(f"Running t-SNE on combined embeddings (perplexity={perplexity}, n_iter={n_iter}, random_state={random_state})...")
    try:
        tsne = TSNE(n_components=2,
                    perplexity=perplexity,
                    n_iter=n_iter,
                    random_state=random_state, # 固定随机状态
                    init='pca',               # 使用 PCA 初始化通常更快更稳定
                    learning_rate='auto')     # 自动学习率
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        print("t-SNE finished.")
        df = pd.DataFrame({
            'Dim1': embeddings_2d[:, 0],
            'Dim2': embeddings_2d[:, 1],
            'ModelType': model_labels
        })
        print("Generating plot...")
        plt.figure(figsize=(10, 8)) 
        ax = plt.gca()
        # Define the color palette
        # 注意：这里的key ('base', 'fft', 'conv') 需要和你 embeddings_dict 中的 key 一致！
        # 如果你的key是 'conv2d'，请修改下面的 palette
        palette = {
            "base": "#4682B4",  # Steel Blue
            "fft": "#FF6F61",   # Coral
            "conv2d": "#2E8B57"  # Sea Green (assuming key is conv2d)
        }
        
        plot_palette = {k: v for k, v in palette.items() if k in model_types_present}
        sns.scatterplot(
            data=df,
            x='Dim1',
            y='Dim2',
            hue='ModelType',      #
            hue_order=model_types_present,
            palette=plot_palette, 
            alpha=0.7,           
            s=10,                 
            ax=ax,
            linewidth=0          
        )

        ax.set_title('t-SNE Visualization of Embeddings')
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(title='Model Type') 
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "tsne_embedding_comparison.png") 
        plt.savefig(plot_path)
        plt.close() # 
        print(f"Combined t-SNE plot saved to {plot_path}")
    except Exception as e:
        print(f"An error occurred during t-SNE or plotting: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging


def plot_similarity_heatmap(embeddings_dict, output_dir, sample_size=500):
    """Calculates and plots cosine similarity heatmaps."""
    print("Generating similarity heatmaps...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_models = len(embeddings_dict)
    fig_width = 6 * num_models
    fig_height = 5
    fig, axes = plt.subplots(1, num_models, figsize=(fig_width, fig_height))
    if num_models == 1:
        axes = [axes]
    model_types = list(embeddings_dict.keys())

    for i, model_type in enumerate(model_types):
        embeddings = embeddings_dict[model_type]
        ax = axes[i]
        # Subsample if too many tokens for efficient calculation/visualization
        num_tokens = embeddings.shape[0]
        indices = np.arange(num_tokens)
        if num_tokens > sample_size:
            print(f"Subsampling {sample_size} embeddings (out of {num_tokens}) for {model_type} heatmap...")
            indices = np.random.choice(num_tokens, sample_size, replace=False)
            embeddings_sample = embeddings[indices, :]
        else:
            embeddings_sample = embeddings

        print(f"Calculating cosine similarity for {model_type} (Sampled shape: {embeddings_sample.shape})...")
        try:
            similarity_matrix = cosine_similarity(embeddings_sample)

            sns.heatmap(similarity_matrix, ax=ax, cmap='RdBu_r', cbar=True, square=True, xticklabels=False, yticklabels=False)
            ax.set_title(f'Cosine Similarity Heatmap ({model_type})\n(Sampled {embeddings_sample.shape[0]} tokens)')
            print(f"Heatmap generated for {model_type}.")
        except Exception as e:
            print(f"Error during heatmap generation for {model_type}: {e}")
            ax.set_title(f'Similarity Heatmap ({model_type}) - Error')
            ax.text(0.5, 0.5, f'Heatmap Error:\n{e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "similarity_heatmap_comparison.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Similarity heatmaps saved to {plot_path}")

def plot_dimension_stats(embeddings_dict, output_dir):
    print("Generating per-dimension statistics plots...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_colors = {
        "base": {"line": "#2B4F81", "fill": "#4682B4"},
        "fft": {"line": "#B22222", "fill": "#FF6F61"},
        "conv": {"line": "#145A32", "fill": "#2E8B57"},
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    dimensions = np.arange(next(iter(embeddings_dict.values())).shape[1])

    for model_type, embeddings in embeddings_dict.items():
        print(f"Calculating dimension stats for {model_type}...")

        try:
            mean_vals = np.mean(embeddings, axis=0)
            std_vals = np.std(embeddings, axis=0)
            min_vals = np.min(embeddings, axis=0)
            max_vals = np.max(embeddings, axis=0)

            colors = model_colors.get(model_type, {"line": "black", "fill": "gray"})
            ax.plot(dimensions, mean_vals, label=f'{model_type} Mean', color=colors["line"], linewidth=2)
            ax.fill_between(dimensions, mean_vals - std_vals, mean_vals + std_vals,
                            color=colors["fill"], alpha=0.25, label=f'{model_type} ±1 Std Dev', linewidth=0)

        except Exception as e:
            print(f"Error during dimension stats plotting for {model_type}: {e}")
            ax.text(0.5, 0.5, f'Error plotting {model_type}:\n{e}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

    ax.set_title('Per-Dimension Embedding Statistics')
    ax.set_xlabel("Embedding Dimension Index")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "dimension_stats_comparison.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Dimension statistics plot saved to {plot_path}")


def calculate_entropy_hist(dimension_data, n_bins=50):
    """Calculates entropy based on histogram binning."""
    counts, bin_edges = np.histogram(dimension_data, bins=n_bins, density=False)
    probabilities = counts / len(dimension_data)
    # Use scipy's entropy which handles p=0 cases correctly
    return calculate_scipy_entropy(probabilities, base=2)

# Replace the previous plot_dimension_stats function with this one
def plot_dimension_stats(embeddings_dict, output_dir, n_dims_to_select=12, n_top_criteria=50, n_bins_entropy=50):
    """
    Selects dimensions based on variance and entropy intersection from the 'base' model,
    then plots statistics for these dimensions across all models.
    """
    print("Selecting dimensions based on 'base' model variance and entropy...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_embeddings = embeddings_dict.get('base')
    if base_embeddings is None or base_embeddings.shape[0] == 0:
        print("Error: 'base' model embeddings not found or empty. Cannot select dimensions.")
        return

    hidden_size = base_embeddings.shape[1]
    if hidden_size == 0:
        print("Error: Embedding hidden size is 0.")
        return

    try:
        # --- Dimension Selection based on 'base' model ---
        print("Calculating variance and entropy for 'base' model dimensions...")
        base_variances = np.var(base_embeddings, axis=0)
        # Calculate entropy for each dimension using the histogram method
        base_entropies = np.array([calculate_entropy_hist(base_embeddings[:, d], n_bins=n_bins_entropy) for d in tqdm(range(hidden_size), desc="Calculating entropy")])

        # Get top N indices for each criterion
        top_var_indices = set(np.argsort(base_variances)[-n_top_criteria:])
        top_ent_indices = set(np.argsort(base_entropies)[-n_top_criteria:])

        # Find intersection
        intersection_indices = sorted(list(top_var_indices.intersection(top_ent_indices)))

        # Select up to n_dims_to_select from intersection, fallback to top variance if intersection is too small
        if len(intersection_indices) >= n_dims_to_select:
            selected_indices = intersection_indices[:n_dims_to_select]
            print(f"Selected {len(selected_indices)} dimensions from Var∩Ent intersection: {selected_indices}")
        else:
            print(f"Var∩Ent intersection too small ({len(intersection_indices)}). Falling back to top {n_dims_to_select} variance dimensions.")
            # Fallback to top variance dimensions if intersection is empty or too small
            selected_indices = sorted(list(top_var_indices))[-n_dims_to_select:]
            if not selected_indices: # Handle case where even variance is zero
                 selected_indices = list(range(min(n_dims_to_select, hidden_size)))
                 print(f"Warning: Could not select meaningful dimensions, using first {len(selected_indices)} dims.")
            print(f"Selected {len(selected_indices)} dimensions based on top variance: {selected_indices}")

        if not selected_indices:
             print("Error: No dimensions were selected.")
             return
        # --- End Dimension Selection ---

        num_models = len(embeddings_dict)
        fig, axes = plt.subplots(num_models, 1, figsize=(12, 4 * num_models), sharex=True)
        if num_models == 1:
            axes = [axes]

        model_types = list(embeddings_dict.keys())
        num_selected = len(selected_indices)
        x_ticks = np.arange(num_selected) # Use 0, 1, 2... for plotting position

        print("Calculating and plotting stats for selected dimensions...")
        for i, model_type in enumerate(model_types):
            embeddings = embeddings_dict.get(model_type)
            ax = axes[i]

            if embeddings is None or embeddings.shape[0] == 0:
                print(f"Skipping dimension stats for {model_type} due to missing embeddings.")
                ax.set_title(f'Selected Dimension Stats ({model_type}) - No Data')
                ax.text(0.5, 0.5, 'No Embedding Data', ha='center', va='center', transform=ax.transAxes)
                continue

            try:
                # Filter data for selected dimensions BEFORE calculating stats
                selected_embeddings = embeddings[:, selected_indices]

                mean_vals = np.mean(selected_embeddings, axis=0)
                std_vals = np.std(selected_embeddings, axis=0)
                min_vals = np.min(selected_embeddings, axis=0)
                max_vals = np.max(selected_embeddings, axis=0)

                ax.plot(x_ticks, mean_vals, label='Mean', color='blue', zorder=3)
                ax.fill_between(x_ticks, mean_vals - std_vals, mean_vals + std_vals, color='blue', alpha=0.2, label='Mean ± 1 Std Dev', zorder=2)
                ax.plot(x_ticks, min_vals, label='Min', color='gray', linestyle='--', alpha=0.7, zorder=1)
                ax.plot(x_ticks, max_vals, label='Max', color='gray', linestyle='--', alpha=0.7, zorder=1)

                ax.set_title(f'Selected Dimension Embedding Statistics ({model_type})')
                ax.set_ylabel("Value")
                ax.legend(loc='best') # Adjust legend location dynamically
                ax.grid(True, linestyle='--', alpha=0.5)
                # Set x-axis ticks and labels correctly
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(selected_indices, rotation=45, ha='right', fontsize=8)


                print(f"Dimension stats plotted for {model_type}.")

            except Exception as e:
                 print(f"Error during dimension stats plotting for {model_type}: {e}")
                 ax.set_title(f'Selected Dimension Stats ({model_type}) - Error')
                 ax.text(0.5, 0.5, f'Stats Plot Error:\n{e}', ha='center', va='center', transform=ax.transAxes)

        axes[-1].set_xlabel("Selected Embedding Dimension Index")
        fig.suptitle("Statistics for Selected Embedding Dimensions (Based on 'base' Var∩Ent)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
        plot_path = os.path.join(output_dir, "selected_dimension_stats_comparison.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Selected dimension statistics plots saved to {plot_path}")

    except Exception as e:
        print(f"An error occurred during dimension selection or plotting setup: {e}")
        import traceback
        traceback.print_exc()


# --- End plot_dimension_stats ---

def plot_norm_distribution(embeddings_dict, output_dir):
    """Plots the distribution of L2 norms using combined violin and KDE plots."""
    print("Generating combined L2 norm distribution plots...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_norms_data = []
    model_types = ['base', 'fft', 'conv2d']
    models_present = []

    print("Calculating L2 norms...")
    for model_type in model_types:
        embeddings = embeddings_dict.get(model_type) 
        if embeddings is None or embeddings.shape[0] == 0:
             print(f"Skipping norm calculation for {model_type} due to missing/empty embeddings.")
             continue
        models_present.append(model_type)
        try:
            norms = np.linalg.norm(embeddings, axis=1)
            for norm in norms:
                all_norms_data.append({'Norm': norm, 'ModelType': model_type})
            print(f"Calculated {len(norms)} norms for {model_type}.")
        except Exception as e:
            print(f"Error calculating norms for {model_type}: {e}")

    if not all_norms_data:
        print("No norm data to plot.")
        return

    df = pd.DataFrame(all_norms_data)
    palette = {
        "base": "#36648B",   # A slightly different blue
        "fft": "#FF6F61",    # Coral
        "conv2d": "#2E8B57"   # Sea Green
    }
    
    plot_palette = {k: v for k, v in palette.items() if k in models_present}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) 

    # --- Combined Violin Plot ---
    try:
        print("Generating combined violin plot...")
        sns.violinplot(x='ModelType', y='Norm', data=df, ax=axes[0],
                       hue='ModelType', # Use hue to map colors
                       palette=plot_palette,
                       order=models_present, 
                       hue_order=models_present,
                       bw_adjust=0.5, 
                       inner='quartile', 
                       cut=0,
                       legend=False 
                       )
        # Attempt to make inner lines match palette - more complex
        # for k, violin in enumerate(axes[0].collections):
        #     if k < len(models_present): # Only color the main violin bodies
        #         violin.set_alpha(0.7) # General alpha for fill (might not work directly)

        axes[0].set_title('Distribution of Embedding L2 Norms (Violin)')
        axes[0].set_xlabel("Model Type")
        axes[0].set_ylabel("L2 Norm")
        axes[0].grid(True, linestyle='--', alpha=0.5)

    except Exception as e:
        print(f"Error plotting combined violin plot: {e}")
        axes[0].set_title('Violin Plot - Error')
        axes[0].text(0.5, 0.5, f'Violin Plot Error:\n{e}', ha='center', va='center', transform=axes[0].transAxes)


    # --- Combined KDE Plot ---
    try:
        print("Generating combined KDE plot...")
        sns.kdeplot(data=df, x='Norm', hue='ModelType', ax=axes[1],
                    hue_order=models_present,
                    palette=plot_palette,
                    fill=True,
                    alpha=0.35,        
                    common_norm=False,  
                    warn_singular=False
                    )
        axes[1].set_title('Distribution of Embedding L2 Norms (KDE)')
        axes[1].set_xlabel("L2 Norm")
        axes[1].set_ylabel("Density")
        axes[1].grid(True, linestyle='--', alpha=0.5)

    except Exception as e:
        print(f"Error plotting combined KDE plot: {e}")
        axes[1].set_title('KDE Plot - Error')
        axes[1].text(0.5, 0.5, f'KDE Plot Error:\n{e}', ha='center', va='center', transform=axes[1].transAxes)


    fig.suptitle("Comparison of Embedding L2 Norm Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plot_path = os.path.join(output_dir, "norm_distribution_comparison_combined.png") 
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Combined L2 norm distribution plots saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare BERT embedding layer outputs.")
    parser.add_argument("--base_model_checkpoint", type=str, required=False, help="Path to the saved checkpoint for the 'base' model.")
    parser.add_argument("--conv_model_checkpoint", type=str, required=False, help="Path to the saved checkpoint for the 'conv2d' model.")
    parser.add_argument("--fft_model_checkpoint", type=str, required=False, help="Path to the saved checkpoint for the 'fft' model.")
    parser.add_argument("--data_dir", type=str, default="../data/sst2", help="Directory containing the SST-2 dataset (specifically train.tsv).")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of sentences to sample from the training set.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--output_dir", type=str, default="./embedding_vis_output", help="Directory to save the output plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or None for auto-detect).")
    # t-SNE parameters
    parser.add_argument("--tsne_perplexity", type=int, default=30, help="Perplexity for t-SNE.")
    parser.add_argument("--tsne_iterations", type=int, default=1000, help="Number of iterations for t-SNE.")
    # Heatmap parameters
    parser.add_argument("--heatmap_sample_size", type=int, default=500, help="Max number of tokens to sample for heatmap visualization.")


    args = parser.parse_args()

    set_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Error loading tokenizer or config: {e}")
        sys.exit(1)
    try:
        sst2_splits = load_sst2(args.data_dir)
        train_df = sst2_splits['train']
        if len(train_df) < args.num_samples:
            print(f"Warning: Requested {args.num_samples} samples, but training set only has {len(train_df)}. Using all training samples.")
            args.num_samples = len(train_df)
        sampled_df = train_df.sample(n=args.num_samples, random_state=args.seed)
        sentences = sampled_df['sentence'].tolist() 
        print(f"Sampled {len(sentences)} sentences.")
    except Exception as e:
        print(f"Error loading or sampling data: {e}")
        sys.exit(1)
    models = {}
    checkpoints = {
        'base': args.base_model_checkpoint,
        'conv2d': args.conv_model_checkpoint,
        'fft': args.fft_model_checkpoint
    }
    for model_type, ckpt_path in checkpoints.items():
         models[model_type] = get_model_instance(model_type, config, ckpt_path, device)
    embeddings_dict = {}
    tokens_dict = {}
    for model_type, model in models.items():
        embeddings, tokens = get_embeddings(model, tokenizer, sentences, args.max_seq_length, device)
        embeddings_dict[model_type] = embeddings
        tokens_dict[model_type] = tokens 

    plot_tsne(embeddings_dict, tokens_dict, args.output_dir,
              perplexity=args.tsne_perplexity, n_iter=args.tsne_iterations, random_state=args.seed)
    plot_similarity_heatmap(embeddings_dict, args.output_dir, sample_size=args.heatmap_sample_size)
    plot_dimension_stats(embeddings_dict, args.output_dir)
    plot_norm_distribution(embeddings_dict, args.output_dir)
    print("\nEmbedding visualization script finished.")

if __name__ == "__main__":
    main()