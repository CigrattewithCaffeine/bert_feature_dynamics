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
    """单个模型特征的简单可视化
    
    Args:
        features_dict: 字典，键为层索引，值为特征矩阵
        labels: 数据标签数组
        save_dir: 保存图像的路径
        epoch: 训练轮次
        model_name: 模型名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 为不同的标签设置不同的颜色 - 使用色盲友好的颜色
    color_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
    unique_labels = np.unique(labels)
    
    # 创建图表网格
    num_layers = len(features_dict)
    rows = int(np.ceil(np.sqrt(num_layers)))
    cols = int(np.ceil(num_layers / rows))
    
    plt.figure(figsize=(5*cols, 4*rows),num=None)
    
    for layer, features in features_dict.items():
        # 执行PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        
        # 绘制散点图
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
        if layer % cols == 0:  # 左侧第一列添加y轴标签
            plt.ylabel("PCA Component 2")
        if layer >= (rows-1) * cols:  # 底部一行添加x轴标签
            plt.xlabel("PCA Component 1")
        
        #explained_var = pca.explained_variance_ratio_
        #plt.legend(title=f"Explained: {explained_var[0]:.2f}, {explained_var[1]:.2f}")
        
        # 添加网格和移除上右边框
        plt.grid(alpha=0.3)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图表
    title = f"{model_name}_epoch_{epoch}" if epoch is not None and model_name is not None else "features"
    plt.savefig(os.path.join(save_dir, f"{title}.png"), dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {os.path.join(save_dir, f'{title}.png')}")
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
    對保存的 CLS 特徵向量進行 PCA 降維 + 可視化，使用真實標籤展示資料點。
    修改为按epoch批量处理，避免内存爆炸，并保存 explained variance
    """
    if feature_dir is None:
        if model_name is None:
            raise ValueError("當 feature_dir 為 None 時，必須提供 model_name")
        saved_root = "saved_features"
        candidates = [d for d in os.listdir(saved_root) if d.startswith(model_name)]
        if not candidates:
            raise FileNotFoundError(f"未找到與 model_name 匹配的特徵目錄: {model_name}")
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
                print(f"跳过 {fname}: 形状不是二维")
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

    print(f"已保存 PCA explained variance 至: {os.path.join(output_dir, 'pca_explained_variance.json')}")
    print(f"所有可视化图像已保存到: {output_dir}")

def create_summary_visualization(visualizations_metadata, color_palette, output_dir, max_samples=None):
    """
    创建汇总可视化图，横向按不同epoch排列，纵向按不同layer排列
    修改为按需加载数据，而不是一次性加载所有数据
    
    Args:
        visualizations_metadata: 可视化元数据列表
        color_palette: 颜色调色板
        output_dir: 输出目录
        max_samples: 最大样本数量
    """
    if not visualizations_metadata:
        print("没有可视化数据用于创建汇总图")
        return
    
    # 获取所有独特的层和epoch
    layers = sorted(list(set([v['layer'] for v in visualizations_metadata])))
    epochs = sorted(list(set([v['epoch'] for v in visualizations_metadata])))
    
    n_layers = len(layers)
    n_epochs = len(epochs)
    
    model_name = visualizations_metadata[0].get('model_name', '')
    model_prefix = f"{model_name}_" if model_name else ""
    
    # 创建一个大图像用于汇总所有可视化
    fig = plt.figure(figsize=(n_epochs * 3, n_layers * 2.5), num=None)
    
    # 创建网格规范
    gs = gridspec.GridSpec(n_layers, n_epochs, figure=fig)
    
    # 为每个层和epoch组合创建子图
    for i, layer in enumerate(layers):
        for j, epoch in enumerate(epochs):
            # 查找匹配的可视化元数据
            vis_metadata = None
            for v in visualizations_metadata:
                if v['layer'] == layer and v['epoch'] == epoch:
                    vis_metadata = v
                    break
            
            if vis_metadata is None:
                continue
            
            # 创建子图
            ax = fig.add_subplot(gs[i, j])
            
            # 按需加载数据
            features = np.load(vis_metadata['pca_data_file'])
            if max_samples is not None:
                features = features[:max_samples]
            
            # 加载标签
            if vis_metadata['labels_file'] and os.path.exists(vis_metadata['labels_file']):
                labels = np.load(vis_metadata['labels_file'])
                if max_samples is not None:
                    labels = labels[:max_samples]
            else:
                # 如果没有标签，回退到默认分组
                n_samples = features.shape[0]
                labels = np.zeros(n_samples, dtype=int)
                labels[n_samples//2:] = 1
            
            # 应用PCA降维
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features)
            
            # 获取唯一标签
            unique_labels = np.array(vis_metadata['unique_labels'])
            
            # 绘制数据点
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                color_idx = idx % len(color_palette)
                ax.scatter(
                    features_pca[mask, 0],
                    features_pca[mask, 1],
                    c=color_palette[color_idx],
                    alpha=0.6,  # 透明度为0.6
                    edgecolors='w',
                    linewidth=0.5,  # 边框粗细为0.5，汇总图中更细些
                    s=20  # 小点以便清晰显示
                )
            
            # 设置坐标轴标签
            if i == n_layers - 1:  # 最后一行
                ax.set_xlabel("PC1", fontsize=8)
            if j == 0:  # 第一列
                ax.set_ylabel("PC2", fontsize=8)
            
            # 设置标题
            ax.set_title(f'Layer{layer},Epoch{epoch}', fontsize=8)
            
            # 添加网格
            ax.grid(alpha=0.3)
            
            # 移除边框但保留刻度值
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # 减小刻度标签大小
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            # 清除不再需要的数据以释放内存
            del features, labels, features_pca
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model_prefix}summary_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 按照更高分辨率再保存一份
    save_path_hires = os.path.join(output_dir, f'{model_prefix}summary_visualization_hires.png')
    plt.savefig(save_path_hires, dpi=1200, bbox_inches='tight')
    
    plt.close()
    print(f"汇总可视化已保存到: {save_path}")


def create_comparison_frame(model_dirs, output_dir, model_names=None):
    """
    为多个不同模型创建对比可视化
    """
    # 自動推測模型資料夾
    resolved_dirs = []
    for name in model_dirs:
        if os.path.exists(name):
            resolved_dirs.append(name)
        else:
            saved_root = "saved_features"
            candidates = [d for d in os.listdir(saved_root) if d.startswith(name)]
            if not candidates:
                raise FileNotFoundError(f"未找到匹配的模型資料夾: {name}")
            resolved_dirs.append(os.path.join(saved_root, sorted(candidates)[-1]))

    model_dirs = resolved_dirs
    os.makedirs(output_dir, exist_ok=True)

    if model_names is None:
        model_names = [os.path.basename(d) for d in model_dirs]
    
    if model_names is None:
        model_names = [os.path.basename(d) for d in model_dirs]
    
    # 从每个模型目录获取特征文件
    all_layer_epoch_pairs = set()
    
    for i, model_dir in enumerate(model_dirs):
        # 获取每个模型目录下的特征文件
        feature_pattern = os.path.join(model_dir, "layer*_epoch*.npy")
        if model_names[i]:
            feature_pattern = os.path.join(model_dir, f"{model_names[i]}_layer*_epoch*_features.npy")
        
        feature_files = glob.glob(feature_pattern)
        
        for f in feature_files:
            basename = os.path.basename(f)
            parts = basename.split('_')
            
            # 提取层和轮次
            if model_names[i]:
                # 如 "ModelName_layer0_epoch0_features.npy"
                layer = int(parts[1][5:])
                epoch = int(parts[2][5:])
            else:
                # 如 "layer0_epoch0_features.npy"
                layer = int(parts[0][5:])
                epoch = int(parts[1][5:])
            
            all_layer_epoch_pairs.add((layer, epoch))
    
    # 色盲友好的颜色
    color_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
    
    # 对每个层和 epoch 组合创建对比图
    for layer, epoch in tqdm(sorted(all_layer_epoch_pairs), desc="创建对比图"):
        plt.figure(figsize=(5 * len(model_dirs), 5), num=None)
        
        for i, model_dir in enumerate(model_dirs):
            model_name = model_names[i] if model_names else os.path.basename(model_dir)
            
            # 加载特征
            if model_name:
                file_path = os.path.join(model_dir, f"{model_name}_layer{layer}_epoch{epoch}_features.npy")
                labels_path = os.path.join(model_dir, f"{model_name}_epoch{epoch}_labels.npy")
            else:
                file_path = os.path.join(model_dir, f"layer{layer}_epoch{epoch}_features.npy")
                labels_path = os.path.join(model_dir, f"epoch{epoch}_labels.npy")
                
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            features = np.load(file_path)
            
            # 尝试加载标签
            if os.path.exists(labels_path):
                labels = np.load(labels_path)
                print(f"使用真实标签: {labels_path}")
            else:
                # 如果没有标签，回退到默认分组
                n_samples = features.shape[0]
                labels = np.zeros(n_samples, dtype=int)
                labels[n_samples//2:] = 1
                print(f"未找到标签文件，使用默认标签划分: {labels_path}")
            
            # PCA 降维
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features)
            
            # 获取唯一标签
            unique_labels = np.unique(labels)
            
            # 绘制子图
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
            
            # 设置标题和坐标轴标签
            plt.title(f'{model_name}, Layer {layer}, Epoch {epoch}', fontsize=12)
            plt.xlabel(f"PC1 (var: {pca.explained_variance_ratio_[0]:.2f})", fontsize=10)
            plt.ylabel(f"PC2 (var: {pca.explained_variance_ratio_[1]:.2f})", fontsize=10)
            
            # 添加图例
            plt.legend(loc='best', fontsize=8)
            
            # 添加网格
            plt.grid(alpha=0.3)
            
            # 移除上右边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # 处理完后释放内存
            del features, labels, features_pca
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(output_dir, f'comparison_layer{layer}_epoch{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"所有模型对比图已保存到: {output_dir}")

# NOTE: 目前未使用，保留給 future probing.py 使用
def extract_features(model, dataloader, device):
    """提取所有层的CLS特征和标签"""
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
            
            # 首次初始化特征列表
            if not all_features:
                for i in range(len(hidden_states)):
                    all_features[i] = []
            
            # 收集各层的CLS特征
            for layer, layer_output in enumerate(hidden_states):
                cls_vec = layer_output[:, 0, :]  # [batch_size, hidden_size]
                all_features[layer].append(cls_vec.cpu().numpy())
            
            # 收集标签
            all_labels.append(batch["labels"].cpu().numpy())
    
    # 合并批次
    for layer in all_features:
        all_features[layer] = np.vstack(all_features[layer])
    
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='特征可视化工具')
    parser.add_argument('--feature_dir', type=str, default=None, required=False, help='输入特征目录')
    parser.add_argument('--output_dir', type=str, default=None, required=False, help='输出图像目录')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to visualize')
    parser.add_argument('--compare', action='store_true', help='是否生成模型对比')
    parser.add_argument('--dirs', nargs='+', default=[], help='比较的多个模型目录')
    parser.add_argument('--names', nargs='+', default=[], help='比较的多个模型名称')
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.dirs:
            print("错误: 进行模型比较需要提供 --dirs 参数!")
            exit(1)
        create_comparison_frame(args.dirs, args.output_dir, args.names if args.names else None)
    else:
        run_feature_visualization(
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_samples=args.max_samples
        )