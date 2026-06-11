# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dplearn
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn


class BiAutoencoder(nn.Module):
    def __init__(self, input_size=4, cnn_channels=16, hidden_size=128, num_layers=2, latent_dim=64):
        super().__init__()
        # 1. 1D-CNN 前置提取器 (物理特征扫描仪)
        # kernel=5, padding=2 保证序列长度 300 进 300 出
        # 自动提取波形的高频斜率、毛刺和突变
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=5, padding=2), # padding = (kernel_size - 1) // 2
            nn.LeakyReLU(0.1)
        )
        
        # Bi-LSTM 编码器
        self.encoder = nn.LSTM(input_size + cnn_channels, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        # 3. 隐空间降维 (混合池化: Max + Mean)
        # hidden_size * 2(双向) * 2(两种池化拼接) -> 转化为 64 维特征
        self.fc_reduce = nn.Linear(hidden_size * 2 * 2, latent_dim) 
        
        # 使用 128 维 LSTM 展开，最后用全连接层输出 4 个物理通道
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.decoder_fc = nn.Linear(hidden_size * 2, input_size)

    def encode(self, x):
        # x 形状: [batch, 300, 4]
        # CNN 需要的形状是 [batch, channel, seq_len]，所以做转置
        x_cnn_in = x.permute(0, 2, 1) 
        cnn_out = self.cnn(x_cnn_in).permute(0, 2, 1) # [batch, 300, 16]
        x_combined = torch.cat([x, cnn_out], dim=2) # 维度变成 [batch, 300, 4 + 16]
        # 将原始波形 x 和 CNN 提取的特征拼接在一起
        # 这样 LSTM 既不会丢失涡旋的原始相位，也能看到激波的梯度特征
        out, _ = self.encoder(x_combined) # out: [batch, 300, 256]
        
        # 混合池化 (Mixed Pooling)：既抓物理尖峰(Max)，又抓背景趋势(Mean)
        pooled_max, _ = torch.max(out, dim=1)
        pooled_avg = torch.mean(out, dim=1)
        pooled_concat = torch.cat([pooled_max, pooled_avg], dim=1) # [batch, 512]
        
        latent_z = self.fc_reduce(pooled_concat) # [batch, 64]
        return latent_z

    def forward(self, x):
        seq_len = x.size(1)
        
        # 编码获取低维物理流形
        latent_z = self.encode(x) # [batch, 64]
        # 沿时间维度复制隐变量，给解码器提供全局上下文
        z_rep = latent_z.unsqueeze(1).repeat(1, seq_len, 1) # [batch, 300, 64]
        # 解码并重构
        dec_out, _ = self.decoder_lstm(z_rep) # [batch, 300, 256]
        reconstructed = self.decoder_fc(dec_out) # [batch, 300, 4]
        
        return reconstructed
    


# %%
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

class PhysicalPredictor:
    def __init__(self, model_path, proto_emb_path, thresholds_path, target_pts_path='target_pts.npy', device='cuda', distance_metric='euclidean'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.target_pts = int(np.load(target_pts_path)[0])
        self.distance_metric = distance_metric  # 'euclidean' or 'cosine'
        
        # 1. 加载模型结构并装载权重
        # 确保这里的参数与 BiAutoencoder 定义时完全一致
        self.model = BiAutoencoder(input_size=4, cnn_channels=16, 
                                   hidden_size=128, num_layers=2, latent_dim=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 2. 加载范本中心向量 (dict: {name: array})
        self.proto_embs = np.load(proto_emb_path, allow_pickle=True).item()
        
        # 3. 加载每个类别的专属动态阈值 (dict: {name: float})
        self.thresholds = np.load(thresholds_path, allow_pickle=True).item()
        
        print(f"Successfully loaded {len(self.proto_embs)} prototypes and their thresholds.")
        print(f"Distance metric: {self.distance_metric}")

    def _preprocess_single_df(self, df):
        """完全复刻训练时的标准化和插值逻辑"""
        # 提取指定的 4 个通道
        seq_raw = df[['B', 'b_z', 'b_max', 'b_min']].values
        
        # --- 步骤 A: 标准化 ---
        B = seq_raw[:, 0]
        perturb = seq_raw[:, 1:]
        
        # B 通道中心化
        B_centerd = B - np.mean(B)
        max_B = np.abs(B_centerd).max()
        B_norm = B_centerd / (max_B + 1e-9)
        
        # 扰动三分量全局标准化
        max_abs = np.abs(perturb).ravel().max()
        perturb_norm = perturb / max_abs if max_abs > 0 else perturb
        
        seq_processed = np.column_stack([B_norm, perturb_norm])
        
        # --- 步骤 B: 插值对齐 ---
        x_old = np.arange(len(seq_processed))
        x_new = np.linspace(0, len(seq_processed) - 1, self.target_pts)
        seq_inter = np.zeros((self.target_pts, 4))
        for i in range(4):
            seq_inter[:, i] = np.interp(x_new, x_old, seq_processed[:, i])
            
        return seq_inter

    def encode(self, df):
        """编码单个样本，返回 64 维嵌入向量"""
        seq_inter = self._preprocess_single_df(df)
        input_tensor = torch.tensor(seq_inter, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode(input_tensor).cpu().numpy().flatten()
        return emb

    def predict(self, df, return_embedding=False):
        """输入一个 df，返回预测类别（含 neither 判定）和距离详情。
           若 return_embedding=True，额外返回 embedding 向量（避免重复 encode）"""
        emb = self.encode(df)
        
        best_match = None
        min_dist = float('inf')
        all_details = {}

        for cls_name, p_emb in self.proto_embs.items():
            if self.distance_metric == 'cosine':
                norm_emb = np.linalg.norm(emb)
                norm_p = np.linalg.norm(p_emb)
                cos_sim = np.dot(emb, p_emb) / (norm_emb * norm_p + 1e-9)
                dist = 1.0 - cos_sim
            else:
                dist = np.linalg.norm(emb - p_emb)
            all_details[cls_name] = {
                'distance': dist,
                'threshold': self.thresholds.get(cls_name, 2.0)
            }
            if dist < min_dist:
                min_dist = dist
                best_match = cls_name
        
        final_label = best_match
        is_neither = False
        
        if min_dist > all_details[best_match]['threshold']:
            final_label = 'neither'
            is_neither = True
                
        if return_embedding:
            return final_label, min_dist, all_details, is_neither, emb
        return final_label, min_dist, all_details, is_neither


# %%
workspace = '..' 
predictor = PhysicalPredictor(
    model_path=os.path.join(workspace, 'bi_model.pth'), 
    proto_emb_path=os.path.join(workspace, 'proto_emb.npy'),
    thresholds_path=os.path.join(workspace, 'thresholds.npy'),
    target_pts_path=os.path.join(workspace, 'target_pts.npy'),
)

# 加载数据
test_df = pd.read_parquet(os.path.join(workspace, 'trainset', '59914.parquet'))

# 获取预测
label, dist, details, is_neither = predictor.predict(test_df)

# --- 结果展示 ---
print(f"【最终判定】: {label.upper()}")
print(f"【最近类别】: {list(details.keys())[np.argmin([v['distance'] for v in details.values()])]} ")
print(f"【最近距离】: {dist:.4f} vs 该类阈值: {details[min(details.items(), key=lambda x: x[1]['distance'])[0]]['threshold']:.4f}")
print("-" * 50)
print("各类别匹配详情:")
for cls, info in details.items():
    status = "<- MATCH" if cls == label else ""
    print(f" - {cls:<12}: Dist={info['distance']:.4f} | Threshold={info['threshold']:.4f} {status}")

if is_neither:
    print(f"\n⚠️ 警告：该样本虽最像{list(details.keys())[np.argmin([v['distance'] for v in details.values()])]}，但距离超过了该类的动态半径，判定为 'neither'。")

# 绘图确认
fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(test_df['B'], color='black')
ax[0].set_ylabel('B (Total)')
ax[1].plot(test_df[['b_z', 'b_max', 'b_min']])
ax[1].set_ylabel('Perturbations')
plt.suptitle(f'Prediction Result: {label}\nMin Dist: {dist:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# # run_batch_predictions — 从 trainset_* 随机抽 fraction 比例预测

# %%
import random
from tqdm import tqdm
import torch

def run_batch_predictions(predictor, data_dir, fraction=0.1, seed=42, batch_log=200, clear_cache_every=500):
    """
    从 data_dir 随机抽取 fraction 比例的 parquet 文件做预测（带进度条）。
    返回 test_data_raw, test_files, test_embeddings, predictions
    
    修复: 不再重复调用 encode()，每 clear_cache_every 个样本释放一次 GPU 缓存
    """
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.parquet')])
    n_select = max(1, int(len(all_files) * fraction))
    random.seed(seed)
    selected = sorted(random.sample(all_files, n_select))
    
    test_data_raw = []
    test_files_list = []
    test_embeddings = []
    predictions_list = []
    
    pbar = tqdm(enumerate(selected), total=n_select, desc="Batch predicting")
    for i, f in pbar:
        file_path = os.path.join(data_dir, f)
        df = pd.read_parquet(file_path, engine='pyarrow')
        
        seq_raw = df[['B', 'b_z', 'b_max', 'b_min']].values
        test_data_raw.append(seq_raw)
        test_files_list.append(f)
        
        # 只做一次 GPU 推理: predict() 内部已调用 encode()
        label, dist, details, is_neither, emb = predictor.predict(df, return_embedding=True)
        predictions_list.append(label)
        test_embeddings.append(emb)
        
        # 定期释放 GPU 缓存，防止碎片累积
        if (i + 1) % clear_cache_every == 0:
            torch.cuda.empty_cache()
        
        if (i + 1) % batch_log == 0:
            pbar.set_postfix_str(f"{i+1}/{n_select} GPU cache cleared")

    torch.cuda.empty_cache()
    test_embeddings = np.array(test_embeddings)
    
    print(f"\nDone: {n_select} samples from {data_dir}")
    stats = pd.Series(predictions_list).value_counts()
    for cls, count in stats.items():
        print(f"  {cls:<15}: {count:<6} ({count/n_select*100:>5.1f}%)")
    
    return test_data_raw, test_files_list, test_embeddings, predictions_list

# --- 运行批量预测 ---
data_dir = os.path.join(workspace, 'testset')

test_data_raw, test_files, test_embeddings, predictions = run_batch_predictions(
    predictor, data_dir, fraction=0.2, seed=123
)
train_size = 0   # 预测 notebook 无训练/验证划分
val_size = 0
final_proto_emb = predictor.proto_embs

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_distance_histograms(test_embeddings, predictions, proto_emb, thresholds, bin_size=0.5, max_dist=10):
    """
    为每个种类绘制距离分布直方图（3x3 固定布局）
    """
    unique_classes = [cls for cls in proto_emb.keys() if cls != 'neither']

    cols = 3
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    
    bins = np.arange(0, max_dist + bin_size, bin_size)

    for i, cls in enumerate(unique_classes):
        if i >= rows * cols:
            break
        r, c = i // cols, i % cols
        
        indices = [idx for idx, pred in enumerate(predictions) if pred == cls]
        
        if not indices:
            axes[r, c].text(0.5, 0.5, f'No samples for {cls}', ha='center')
            axes[r, c].set_title(f"Class: {cls}")
            continue
            
        class_embs = test_embeddings[indices]
        center_vec = proto_emb[cls]
        dists = np.linalg.norm(class_embs - center_vec, axis=1)
        
        axes[r, c].hist(dists, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

        mean_d = np.mean(dists)
        axes[r, c].axvline(mean_d, color='red', linestyle='--', label=f'Mean: {mean_d:.2f}')
        
        axes[r, c].set_title(f"Class: {cls.upper()} (N={len(indices)})", fontsize=14, fontweight='bold')
        if r == rows - 1:
            axes[r, c].set_xlabel("Distance to Prototype")
        else:
            axes[r, c].set_xticklabels([])
        if c == 0:
            axes[r, c].set_ylabel("Frequency")
        axes[r, c].set_xlim(0, max_dist)
        axes[r, c].grid(axis='y', alpha=0.3)
        axes[r, c].legend()

    for j in range(len(unique_classes), rows * cols):
        r, c = j // cols, j % cols
        axes[r, c].axis('off')

    plt.suptitle(f"Sample Distance Distribution per Class", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()


plot_distance_histograms(
    test_embeddings=test_embeddings, 
    predictions=predictions, 
    proto_emb=final_proto_emb,
    thresholds=predictor.thresholds,
    bin_size=0.1,
    max_dist=20
)


# %% [markdown]
# ### t-SNE 可视化

# %%
from sklearn.manifold import TSNE

# 1. 提取所有原型 embedding
unique_classes = list(final_proto_emb.keys())
proto_vecs = np.array([final_proto_emb[cls] for cls in unique_classes])

# 2. 合并数据并降维
all_vecs = np.vstack([test_embeddings, proto_vecs])
print("Running t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
vecs_2d = tsne.fit_transform(all_vecs)

test_2d = vecs_2d[:len(test_embeddings)]
proto_2d = vecs_2d[len(test_embeddings):]

# 3. 颜色映射
colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'olive']
color_map = {cls: colors_list[i % len(colors_list)] for i, cls in enumerate(unique_classes)}
color_map['neither'] = 'lightgrey'

point_colors = [color_map.get(p, 'lightgrey') for p in predictions]

# 4. 绘图
plt.figure(figsize=(14, 10))
plt.scatter(test_2d[:, 0], test_2d[:, 1], c=point_colors, alpha=0.3, s=4, label='Classified Samples')

for i, cls in enumerate(unique_classes):
    plt.scatter(proto_2d[i, 0], proto_2d[i, 1],
                color=color_map[cls], linewidth=1,
                s=150, label=f'Proto: {cls}', marker='*')
    plt.text(proto_2d[i, 0] + 1, proto_2d[i, 1] + 1,
             cls, fontsize=12, fontweight='bold', color=color_map[cls])

plt.title('t-SNE Visualization: Predicted Clusters vs Prototypes (Test Set)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


# %%
def plot_class_samples(target_class, total_count=100, per_fig=10, seed=42):
    """
    随机抽取指定预测类别的测试集样本 (固定随机种子保证可复现)
    """
    import random
    rng = random.Random(seed)

    class_indices = [i for i, pred in enumerate(predictions) if pred == target_class]

    if not class_indices:
        print(f"在预测结果中没有找到类别: {target_class}")
        return

    target_proto_center = final_proto_emb.get(target_class)
    n_sample = min(total_count, len(class_indices))
    selected_subset = rng.sample(class_indices, n_sample)
    
    for i in range(0, len(selected_subset), per_fig):
        batch_indices = selected_subset[i : i + per_fig]
        current_n = len(batch_indices)

        fig, axes = plt.subplots(2, current_n, figsize=(5 * current_n, 8))
        fig.suptitle(f"Class: {target_class.upper()} | Random {i+1}-{i+current_n} of {n_sample} (seed={seed})",
                     fontsize=16, fontweight='bold', y=1.05)

        for j, idx in enumerate(batch_indices):
            ax_top = axes[0, j] if current_n > 1 else axes[0]
            ax_btm = axes[1, j] if current_n > 1 else axes[1]
            
            seq = test_data_raw[idx]
            file_name = test_files[idx]

            if target_proto_center is not None:
                sample_emb = test_embeddings[idx]
                dist_to_proto = np.linalg.norm(sample_emb - target_proto_center)
                dist_str = f"Dist: {dist_to_proto:.4f} | "
            else:
                dist_str = ""
            
            B = seq[:, 0]
            b_z = seq[:, 1]
            b_max = seq[:, 2]
            b_min = seq[:, 3]
            
            b_sum_sq = b_z**2 + b_max**2 + b_min**2
            max_idx = np.argmax(b_sum_sq)
            min_B_idx = np.argmin(B)
            
            ax_top.plot(B, color='black', linewidth=1.5)
            ax_top.axhline(np.mean(B), color='gray', linestyle='--', alpha=0.6)
            ax_top.axvline(max_idx, color='red', linestyle='--', alpha=0.4)
            ax_top.axvline(min_B_idx, color='green', linestyle='--', alpha=0.4)
            
            ax_top.set_title(f"File: {file_name}\n{dist_str}Idx: {idx}", fontsize=11)
            
            if j == 0: ax_top.set_ylabel('B', fontweight='bold')
            
            ax_btm.plot(b_z, label='b_z', color='blue', alpha=0.8)
            ax_btm.plot(b_max, label='b_max', color='red', alpha=0.8)
            ax_btm.plot(b_min, label='b_min', color='green', alpha=0.8)
            ax_btm.axvline(max_idx, color='red', linestyle='--', alpha=0.4)
            ax_btm.axhline(0, color='gray', linestyle='--', alpha=0.6)
            ax_btm.axvline(min_B_idx, color='green', linestyle='--', alpha=0.4)
            
            if j == 0: ax_btm.set_ylabel('b (Perturbations)', fontweight='bold')
            if j == current_n - 1: ax_btm.legend(loc='upper right', fontsize='small')
            ax_btm.set_xlabel('Time Step')

        plt.tight_layout()
        plt.show()


# 绘制 shock 类的随机 100 个样本 (seed=42 保证可复现)
# plot_class_samples(target_class='shock', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='sheet', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='hole', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='soliton', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
plot_class_samples(target_class='c vortex', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='l vortex', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='l vortex chain', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='vortex chain', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='alfven dis', total_count=100, per_fig=10, seed=42)
# print("-------------------------------")
# plot_class_samples(target_class='neither', total_count=100, per_fig=10, seed=42)


# %%
def plot_text(l, n):
    """随机抽查测试集样本（从索引 l 开始，每页 n 个）"""
    selected_indices = list(range(l, min(n + l, len(test_data_raw))))
    fig, ax = plt.subplots(2, n, figsize=(5 * n, 8))
    
    for j, idx in enumerate(selected_indices):
        seq = test_data_raw[idx]
        file_name = test_files[idx]
        
        B = seq[:, 0]
        b_z = seq[:, 1]
        b_max = seq[:, 2]
        b_min = seq[:, 3]
        
        b_sum_sq = b_z**2 + b_max**2 + b_min**2
        max_idx = np.argmax(b_sum_sq)
        
        ax[0, j].plot(B, color='black')
        ax[0, j].axhline(np.mean(B), color='gray', linestyle='--')
        ax[0, j].axvline(max_idx, color='red', linestyle='--', alpha=0.5)
        ax[0, j].set_ylabel('B')
        
        ax[0, j].set_title(f"File: {file_name}\nIdx: {idx} | Pred: {predictions[idx]}", 
                           fontsize=11, fontweight='bold')
        
        ax[1, j].plot(b_z, label='b_z', color='blue')
        ax[1, j].plot(b_max, label='b_max', color='red')
        ax[1, j].plot(b_min, label='b_min', color='green')
        ax[1, j].axvline(max_idx, color='red', linestyle='--', alpha=0.5)
        ax[1, j].axhline(0, color='gray', linestyle='--')
        
        if j == n - 1:
            ax[1, j].legend(loc='upper right', fontsize='small')
            
        ax[1, j].set_xlabel('Time Step')
        ax[1, j].set_ylabel('b')

    plt.tight_layout()
    plt.show()


for i in range(0, min(100, len(test_data_raw)), 10):
    plot_text(i, 10)
