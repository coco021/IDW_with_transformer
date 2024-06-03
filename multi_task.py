"""
新增mask_token：用于mask操作的特殊嵌入向量。
shuffle_head：用于预测被mask掉点的特征值（经纬度+降水量）的线性层。
新增函数dual_task_forward：接受mask_indices参数，用于mask掉特定点的特征。
"""


import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

# 示例数据
data = np.array([
    [120.1, 30.2, 100],
    [120.2, 30.3, 110],
    [121.1, 31.2, 90],
    # 添加更多数据点
])

# 设定参数
n_patches = 10  # 定义patch数量
num_samples = 2  # 定义样本数量（假设我们的数据集中有两个样本）

# 示例数据集（每个样本包含若干点）
data_samples = [data for _ in range(num_samples)]


# 数据预处理函数
def preprocess_data(data_samples, n_patches):
    patches_all_samples = []
    relative_positions_all_samples = []
    patch_centers_all_samples = []

    for data in data_samples:
        kmeans = KMeans(n_clusters=n_patches)
        patch_labels = kmeans.fit_predict(data[:, :2])  # 仅使用经纬度信息进行聚类

        patches = []
        patch_centers = []
        for i in range(n_patches):
            patch = data[patch_labels == i]
            patches.append(patch)
            patch_center = patch[:, :2].mean(axis=0)
            patch_centers.append(patch_center)

        max_patch_size = max(len(patch) for patch in patches)
        padded_patches = []
        relative_positions = []

        for patch, center in zip(patches, patch_centers):
            padding = np.zeros((max_patch_size - len(patch), data.shape[1]))
            padded_patch = np.vstack((patch, padding))
            padded_patches.append(padded_patch)

            relative_pos = np.zeros((max_patch_size, 2))
            relative_pos[:len(patch), :] = patch[:, :2] - center
            relative_positions.append(relative_pos)

        patches_all_samples.append(np.array(padded_patches))
        relative_positions_all_samples.append(np.array(relative_positions))
        patch_centers_all_samples.append(np.array(patch_centers))

    patches_array = np.array(patches_all_samples)
    relative_positions_array = np.array(relative_positions_all_samples)
    patch_centers_array = np.array(patch_centers_all_samples)

    return patches_array, relative_positions_array, patch_centers_array


# 预处理数据
patches_array, relative_positions_array, patch_centers_array = preprocess_data(data_samples, n_patches)

# 将numpy数组转换为tensor
patches_tensor = torch.tensor(patches_array, dtype=torch.float32)
relative_positions_tensor = torch.tensor(relative_positions_array, dtype=torch.float32)
patch_centers_tensor = torch.tensor(patch_centers_array, dtype=torch.float32).unsqueeze(2)


# 定义双任务Transformer模型
class DualTaskTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_patches, max_patch_size):
        super(DualTaskTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.max_patch_size = max_patch_size

        # Linear projection of flattened patches
        self.projection = nn.Linear(data.shape[1], embed_dim)

        # Positional embedding
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.relative_pos_embed = nn.Linear(2, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers for geo and precip prediction
        self.geo_head = nn.Linear(embed_dim, 2)  # 经纬度
        self.precip_head = nn.Linear(embed_dim, 1)  # 降水量

        # Dual task: mask and shuffle
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.shuffle_head = nn.Linear(embed_dim, 3)  # 预测mask掉的点的值（经纬度+降水量）

    def forward(self, x, patch_centers, relative_positions):
        B, N, T, _ = x.shape  # B: 批量大小, N: patch数量, T: 每个patch中的点数, _: 特征维度

        # Flatten patches and apply linear projection
        x = x.view(B * N, T, -1)
        x = self.projection(x)

        # Add positional embedding
        patch_pos_embedding = self.patch_pos_embed.expand(B, -1, -1)
        relative_pos_embedding = self.relative_pos_embed(relative_positions.view(B * N, T, -1))
        x = x + patch_pos_embedding.view(B, N, 1, -1).expand(B, N, T, -1).reshape(B * N, T, -1) + relative_pos_embedding

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate patch features (e.g., by mean)
        x = x.mean(dim=1)

        # Geo and precip predictions
        geo_output = self.geo_head(x)
        value_output = self.precip_head(x)

        return geo_output, value_output

    def dual_task_forward(self, x, patch_centers, relative_positions, mask_indices=None):
        B, N, T, _ = x.shape  # B: 批量大小, N: patch数量, T: 每个patch中的点数, _: 特征维度

        # Flatten patches and apply linear projection
        x = x.view(B * N, T, -1)
        x = self.projection(x)

        # Add positional embedding
        patch_pos_embedding = self.patch_pos_embed.expand(B, -1, -1)
        relative_pos_embedding = self.relative_pos_embed(relative_positions.view(B * N, T, -1))
        x = x + patch_pos_embedding.view(B, N, 1, -1).expand(B, N, T, -1).reshape(B * N, T, -1) + relative_pos_embedding

        if mask_indices is not None:
            x[mask_indices] = self.mask_token

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate patch features (e.g., by mean)
        x = x.mean(dim=1)

        # Geo and precip predictions
        geo_output = self.geo_head(x)
        value_output = self.precip_head(x)

        # Masked value prediction
        masked_pred = self.shuffle_head(x[mask_indices]) if mask_indices is not None else None

        return geo_output, value_output, masked_pred


# 模型参数
embed_dim = 64
num_heads = 8
num_layers = 6

# 初始化模型
model = DualTaskTransformer(embed_dim, num_heads, num_layers, n_patches, max_patch_size)

# 示例前向传播
geo_output, value_output, masked_pred = model.dual_task_forward(patches_tensor, patch_centers_tensor,
                                                                 relative_positions_tensor, mask_indices=[(0, 0, 0)])
print("Geo Output:", geo_output)
print("Precip Output:", value_output)
print("Masked Prediction:", masked_pred)
