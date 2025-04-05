import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def run_feature_dynamics_analysis(
    feature_dir: str,
    output_dir: str,
    n_clusters: int = 2,
    pca_dim: int = 2,
    figsize=(6, 6),
    random_state: int = 42,
):
    """
    对保存的 CLS 特征向量进行 PCA 降维 + 聚类 + 可视化。
    Args:
        feature_dir: 存储 npy 特征向量的目录。
        output_dir: 图像输出目录。
        n_clusters: KMeans 聚类类别数。
        pca_dim: 降维维度。
        figsize: 图像大小。
        random_state: 聚类随机种子。
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".npy")])

    for fname in feature_files:
        path = os.path.join(feature_dir, fname)
        features = np.load(path)  # shape: [num_samples, hidden_size]
        if features.ndim != 2:
            print(f"Skip {fname}: shape not 2D.")
            continue

        # PCA
        pca = PCA(n_components=pca_dim, random_state=random_state)
        features_pca = pca.fit_transform(features)

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(features_pca)

        # 绘图
        plt.figure(figsize=figsize)
        colors = ["#D62728", "#1F77B4"]  # 红蓝色盲友好配色
        for cluster in range(n_clusters):
            indices = cluster_labels == cluster
            plt.scatter(
                features_pca[indices, 0],
                features_pca[indices, 1],
                c=colors[cluster],
                label=f"Cluster {cluster}",
                alpha=0.7,
                edgecolors="k",
                s=50,
            )
        plt.title(fname.replace(".npy", ""), fontsize=12)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()

        save_name = fname.replace(".npy", ".png")
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path)
        plt.close()

    print(f"所有图像已保存至: {output_dir}")