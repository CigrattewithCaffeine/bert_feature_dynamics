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
    
def plot_tsne(embeddings_dict, output_dir, perplexity, n_iter, random_state):
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


def plot_similarity_heatmap(embeddings_dict, output_dir, sample_size):
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
            ax.set_title(f'Cosine Similarity Heatmap ({model_type})')
            print(f"Heatmap generated for {model_type}.")
        except Exception as e:
            print(f"Error during heatmap generation for {model_type}: {e}")
            ax.set_title(f'Similarity Heatmap ({model_type}) - Error')
            ax.text(0.5, 0.5, f'Heatmap Error:\n{e}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "similarity_heatmap_comparison.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Similarity heatmaps saved to {plot_path}")

def plot_dimension_stats(embeddings_dict, output_dir):
    """
    Plots mean, standard deviation, min, and max values across all dimensions
    for all models in a single plot.
    """
    print("Generating per-dimension statistics plots...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define colors for each model
    model_colors = {
        "base": {"line": "#2B4F81", "fill": "#4682B4"},
        "fft": {"line": "#B22222", "fill": "#FF6F61"},
        "conv2d": {"line": "#145A32", "fill": "#2E8B57"},  # Changed from "conv" to "conv2d" to match keys
    }

    # Create a single figure with one axis for all models
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Find the first non-empty embeddings to determine dimensions
    first_valid_embeddings = None
    for emb in embeddings_dict.values():
        if emb is not None and emb.shape[0] > 0:
            first_valid_embeddings = emb
            break
    
    if first_valid_embeddings is None:
        print("Error: No valid embeddings found.")
        return
    
    dimensions = np.arange(first_valid_embeddings.shape[1])  # Get all dimensions

    # Plot each model's statistics on the same axis
    for model_type, embeddings in embeddings_dict.items():
        if embeddings is None or embeddings.shape[0] == 0:
            print(f"Skipping {model_type} due to missing or empty embeddings.")
            continue
        
        print(f"Calculating dimension stats for {model_type}...")
        
        try:
            # Calculate statistics across all tokens for each dimension
            mean_vals = np.mean(embeddings, axis=0)
            std_vals = np.std(embeddings, axis=0)
            
            # Get colors for this model (fallback to black/gray if not specified)
            colors = model_colors.get(model_type, {"line": "black", "fill": "gray"})
            
            # Plot mean line
            ax.plot(dimensions, mean_vals, 
                    label=f'{model_type} Mean', 
                    color=colors["line"], 
                    linewidth=1.5,
                    alpha=0.9)
            
            # Plot standard deviation band
            ax.fill_between(dimensions, 
                           mean_vals - std_vals, 
                           mean_vals + std_vals,
                           color=colors["fill"], 
                           alpha=0.2, 
                           label=f'{model_type} ±1 Std Dev')
            
            print(f"Plotted statistics for {model_type} with {len(dimensions)} dimensions.")
            
        except Exception as e:
            print(f"Error during dimension stats plotting for {model_type}: {e}")
            import traceback
            traceback.print_exc()

    # Set plot title and labels
    ax.set_title('Per-Dimension Embedding Statistics (All Models)', fontsize=14)
    ax.set_xlabel("Embedding Dimension Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend with better positioning and formatting
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Improve overall layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, "dimension_stats_all_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Full dimension statistics plot saved to {plot_path}")


def calculate_entropy_hist(dimension_data, n_bins=50):
    """Calculates entropy based on histogram binning."""
    counts, bin_edges = np.histogram(dimension_data, bins=n_bins, density=False)
    probabilities = counts / len(dimension_data)
    # Use scipy's entropy which handles p=0 cases correctly
    return calculate_scipy_entropy(probabilities, base=2)


def plot_dimension_violins(embeddings_dict, output_dir, n_dims_to_select=12, n_top_criteria=50, n_bins_entropy=50):
    """为选定的维度绘制提琴图，每个维度显示三个模型的分布对比
    选择标准：熵最高的前50个维度与方差最大的前50个维度的交集，然后按综合得分排序"""
    print("为选定维度生成提琴图分布...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 确定要使用的模型类型
    model_types = ['base', 'fft', 'conv2d']
    models_present = []
    
    # 颜色设置
    palette = {
        "base": "#36648B",   # 蓝色
        "fft": "#FF6F61",    # 珊瑚色
        "conv2d": "#2E8B57"  # 海绿色
    }
    
    # 从base模型选择维度
    base_embeddings = embeddings_dict.get('base')
    if base_embeddings is None or base_embeddings.shape[0] == 0:
        print("错误：'base'模型嵌入为空，无法选择维度。")
        return

    hidden_size = base_embeddings.shape[1]
    print(f"选择维度中，嵌入总维度大小：{hidden_size}")
    
    # 计算方差
    print("计算各维度方差...")
    base_variances = np.var(base_embeddings, axis=0)
    
    # 计算熵
    print("计算各维度熵...")
    base_entropies = np.array([calculate_entropy_hist(base_embeddings[:, d], n_bins=n_bins_entropy) 
                              for d in tqdm(range(hidden_size), desc="计算熵")])
    
    # 获取方差和熵排名前n_top_criteria的维度索引
    top_var_indices = set(np.argsort(base_variances)[-n_top_criteria:])
    top_ent_indices = set(np.argsort(base_entropies)[-n_top_criteria:])

    # 找到交集
    intersection_indices = list(top_var_indices.intersection(top_ent_indices))
    print(f"方差和熵高维度的交集大小: {len(intersection_indices)}")
    
    # 根据方差和熵的组合得分对交集维度排序
    # 使用方差和熵的归一化值之和作为得分
    if intersection_indices:
        # 归一化方差和熵以便合理组合
        norm_var = (base_variances - np.min(base_variances)) / (np.max(base_variances) - np.min(base_variances))
        norm_ent = (base_entropies - np.min(base_entropies)) / (np.max(base_entropies) - np.min(base_entropies))
        
        # 计算交集维度的综合得分
        combined_scores = [(i, norm_var[i] + norm_ent[i]) for i in intersection_indices]
        # 按得分降序排序
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择得分最高的n_dims_to_select个维度
        selected_indices = [idx for idx, _ in combined_scores[:n_dims_to_select]]
        print(selected_indices)
    else:
        # 如果交集为空，则回退到按方差选择
        print("警告：方差和熵的交集为空，回退到方差选择")
        selected_indices = sorted(list(top_var_indices))[:n_dims_to_select]
    
    # 确保我们有所需的维度数量
    final_n_dims = min(len(selected_indices), n_dims_to_select)
    selected_indices = sorted(selected_indices[:final_n_dims])  # 按维度索引排序
    
    print(f"选择了以下{len(selected_indices)}个维度用于可视化: {selected_indices}")
    
    # 收集所有模型在所选维度上的数据
    all_dimension_data = []
    
    for model_type in model_types:
        embeddings = embeddings_dict.get(model_type)
        if embeddings is None or embeddings.shape[0] == 0:
            print(f"跳过{model_type}，因为嵌入为空。")
            continue
            
        models_present.append(model_type)
        
        try:
            # 为每个选定的维度添加数据
            for dim_idx in selected_indices:
                dim_values = embeddings[:, dim_idx]
                for val in dim_values:
                    all_dimension_data.append({
                        'Value': val, 
                        'Dimension': f"Dim {dim_idx}", 
                        'ModelType': model_type
                    })
            print(f"已收集{model_type}模型的维度数据。")
        except Exception as e:
            print(f"收集{model_type}的维度数据时出错: {e}")
    
    if not all_dimension_data:
        print("没有数据可绘制。")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(all_dimension_data)
    
    # 只保留出现的模型的调色板
    plot_palette = {k: v for k, v in palette.items() if k in models_present}
    
    # 创建提琴图
    plt.figure(figsize=(20, 10))
    
    try:
        print("生成提琴图...")
        # 使用catplot创建分面网格图
        g = sns.catplot(
            data=df,
            x='Dimension', 
            y='Value',
            hue='ModelType',
            kind='violin',
            palette=plot_palette,
            hue_order=models_present,
            split=False,        # 不拆分提琴图
            inner='quartile',   # 显示四分位数
            scale='width',      # 所有提琴图宽度相同
            linewidth=1,        # 轮廓线宽度
            height=8,           # 图形高度
            aspect=2,           # 宽高比
            legend_out=False,   # 将图例放在图中
            dodge=True,         # 并排显示不同模型的提琴图
            bw=0.2              # 带宽控制（影响平滑度）
        )
        
        # 配置图形
        g.set_xticklabels(rotation=45)
        g.fig.suptitle('Embedding Value of Chosen Dimensions', fontsize=16, y=1.02)
        g.set_axis_labels("Embeddings", "Value")
        
        # 移除顶部科学计数法显示并添加网格
        for ax in g.axes.flat:
            ax.ticklabel_format(style='plain', axis='y')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        
        # 保存图形
        plot_path = os.path.join(output_dir, "dimension_violin_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"维度提琴图保存至 {plot_path}")
        
    except Exception as e:
        print(f"绘制提琴图时出错: {e}")
        import traceback
        traceback.print_exc()


def plot_kde_distributions(embeddings_dict, output_dir, n_dims_to_select=12, n_top_criteria=50, n_bins_entropy=50):
    """绘制维度值的KDE分布图"""
    print("生成组合KDE分布图...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 确定要使用的模型类型
    model_types = ['base', 'fft', 'conv2d']
    models_present = []
    
    # 颜色设置
    palette = {
        "base": "#36648B",   # 蓝色
        "fft": "#FF6F61",    # 珊瑚色
        "conv2d": "#2E8B57"  # 海绿色
    }
    
    # 从base模型选择维度
    base_embeddings = embeddings_dict.get('base')
    if base_embeddings is None or base_embeddings.shape[0] == 0:
        print("错误：'base'模型嵌入为空，无法选择维度。")
        return

    hidden_size = base_embeddings.shape[1]
    print(f"选择维度中，嵌入总维度大小：{hidden_size}")
    
    # 计算方差
    print("计算各维度方差...")
    base_variances = np.var(base_embeddings, axis=0)
    
    # 计算熵
    print("计算各维度熵...")
    base_entropies = np.array([calculate_entropy_hist(base_embeddings[:, d], n_bins=n_bins_entropy) 
                              for d in tqdm(range(hidden_size), desc="计算熵")])
    
    # 获取方差和熵排名前n_top_criteria的维度索引
    top_var_indices = set(np.argsort(base_variances)[-n_top_criteria:])
    top_ent_indices = set(np.argsort(base_entropies)[-n_top_criteria:])

    # 找到交集
    intersection_indices = list(top_var_indices.intersection(top_ent_indices))
    print(f"方差和熵高维度的交集大小: {len(intersection_indices)}")
    
    # 根据方差和熵的组合得分对交集维度排序
    # 使用方差和熵的归一化值之和作为得分
    if intersection_indices:
        # 归一化方差和熵以便合理组合
        norm_var = (base_variances - np.min(base_variances)) / (np.max(base_variances) - np.min(base_variances))
        norm_ent = (base_entropies - np.min(base_entropies)) / (np.max(base_entropies) - np.min(base_entropies))
        
        # 计算交集维度的综合得分
        combined_scores = [(i, norm_var[i] + norm_ent[i]) for i in intersection_indices]
        # 按得分降序排序
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择得分最高的n_dims_to_select个维度
        selected_indices = [idx for idx, _ in combined_scores[:n_dims_to_select]]
        print(selected_indices)
    else:
        # 如果交集为空，则回退到按方差选择
        print("警告：方差和熵的交集为空，回退到方差选择")
        selected_indices = sorted(list(top_var_indices))[:n_dims_to_select]
    
    # 确保我们有所需的维度数量
    final_n_dims = min(len(selected_indices), n_dims_to_select)
    selected_indices = sorted(selected_indices[:final_n_dims])  # 按维度索引排序
    
    print(f"选择了以下{len(selected_indices)}个维度用于可视化: {selected_indices}")
    all_dimension_data = []
    
    for model_type in model_types:
        embeddings = embeddings_dict.get(model_type)
        if embeddings is None or embeddings.shape[0] == 0:
            print(f"跳过{model_type}，因为嵌入为空。")
            continue
            
        models_present.append(model_type)
        
        try:
            for dim_idx in selected_indices:
                dim_values = embeddings[:, dim_idx]
                for val in dim_values:
                    all_dimension_data.append({
                        'Value': val, 
                        'Dimension': f"Dim {dim_idx}", 
                        'ModelType': model_type
                    })
            print(f"已收集{model_type}模型的维度数据。")
        except Exception as e:
            print(f"收集{model_type}的维度数据时出错: {e}")
    
    if not all_dimension_data:
        print("没有数据可绘制。")
        return
    df = pd.DataFrame(all_dimension_data)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    try:
        # 为每个模型创建新的DataFrame，添加L2范数
        kde_data = []
        
        for model_type in models_present:
            embeddings = embeddings_dict.get(model_type)
            if embeddings is not None and embeddings.shape[0] > 0:
                # 计算每个embedding的L2范数
                norms = np.linalg.norm(embeddings, axis=1)
                for norm in norms:
                    kde_data.append({
                        'Norm': norm,
                        'ModelType': model_type
                    })
        
        if kde_data:
            kde_df = pd.DataFrame(kde_data)
            
            # 绘制KDE图
            sns.kdeplot(data=kde_df, x='Norm', hue='ModelType', 
                        hue_order=models_present,
                        palette=palette,
                        fill=True,
                        alpha=0.35,        
                        common_norm=False,  
                        warn_singular=False)
                        
            ax.set_title('Density Distribution of Embeddings L2 Norm')
            ax.set_xlabel("L2 Norm")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "norm_distribution_kde.png")
            plt.savefig(plot_path, dpi=300)
            plt.close(fig)
            print(f"L2范数KDE分布图保存至 {plot_path}")
    
    except Exception as e:
        print(f"绘制KDE图时出错: {e}")
        import traceback
        traceback.print_exc()


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
        sst2_splits = load_sst2()
        train_df = sst2_splits['train']
        if len(train_df) < args.num_samples:
            print(f"Warning: Requested {args.num_samples} samples, but training set only has {len(train_df)}. Using all training samples.")
            args.num_samples = len(train_df)
        sampled_df = train_df.sample(n=args.num_samples, random_state=args.seed)
        sentences = sampled_df['text'].tolist() 
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

    plot_tsne(embeddings_dict, args.output_dir,
              perplexity=args.tsne_perplexity, n_iter=args.tsne_iterations, random_state=args.seed)
    plot_similarity_heatmap(embeddings_dict, args.output_dir, sample_size=args.heatmap_sample_size)
    plot_dimension_stats(embeddings_dict, args.output_dir)
    plot_dimension_violins(embeddings_dict, args.output_dir)
    plot_kde_distributions(embeddings_dict, args.output_dir)
    print("\nEmbedding visualization script finished.")

if __name__ == "__main__":
    main()