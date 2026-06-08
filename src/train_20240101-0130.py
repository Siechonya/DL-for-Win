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

# %% [markdown]
# 已经实现的研究摘要背景：
# > 空间等离子体湍流的演化特征一直是空间物理学的重要研究课题。以往研究多聚焦于太阳风环境，而对火星空间等离子体湍流的全球演化规律缺乏系统认知。本研究联合使用 2023 至 2024 年间 MAVEN 与“天问一号”的高分辨率磁场观测数据，从功率谱密度与相干性结构全球分布的视角，对火星空间湍流展开了深入研究。研究采用小波变换方法计算 PSD，并以质子回旋频率 (𝑓𝑐𝑝) 为界，分别提取了能量级联的 MHD 尺度 (< 𝑓𝑐𝑝) 和动理学尺度 (> 𝑓𝑐𝑝) 的能谱特征。统计结果表明，火星磁鞘存在明显的惯性区缺失，其 MHD 尺度的能谱斜率中位数仅为-0.5。此外，本研究在亚离子尺度上，通过小波反变换与间歇性结构提取算法，实现了对火星空间内 6 种典型相干性结构的自动提取与归类，并绘制了其全球分布图谱。观测发现，阿尔芬涡旋、磁洞和孤子在磁鞘中大量存在，但在空间分布上呈现显著差异：阿尔芬涡旋倾向于分布在磁堆积边界附近，而磁洞与孤子则更靠近舷激波。同时，研究在火星磁尾区域观测到大量激波结构，揭示了该现象可能与 O+/O+2 重离子流片的局地脉冲特征有关。最后，针对磁鞘中阿尔芬涡旋易被误判为电流片的现象，本文指出电流片的分类判据仍有待进一步优化。本研究为理解火星空间等离子体湍流的跨尺度能量演化及相干性结构的动力学过程提供了观测证据。
#
# 本代码目的：使用深度学习实现对火星空间等离子体相干性结构的自动归类。分类目标为 6 种不同的相干性结构（见samples_clean文件夹），注意样本包含一半左右的噪声样本，不属于任何一类。

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import copy
import itertools # 用于循环迭代范本

# %%
# === 检测并设置计算设备 (GPU / CPU) ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(path):
    files = [f for f in os.listdir(path) if f.endswith('.parquet')]
    data_processed = []
    data_raw = []
    for f in files:
        file_path = os.path.join(path, f)
        df = pd.read_parquet(file_path, engine='pyarrow')
        seq_raw = df[['B', 'b_z', 'b_max', 'b_min']].values
        data_raw.append(seq_raw)
        
        B = seq_raw[:, 0]
        b_z = seq_raw[:, 1]
        b_max = seq_raw[:, 2]
        b_min = seq_raw[:, 3]
        
        B_centerd = B - np.mean(B)
        max_B = np.abs(B_centerd).max()
        B_norm = B_centerd / (max_B + 1e-9)
        
        perturb = np.column_stack([b_z, b_max, b_min])
        max_abs = np.abs(perturb).ravel().max()
        if max_abs > 0:
            perturb_norm = perturb / max_abs
        else:
            perturb_norm = perturb
            
        seq_processed = np.column_stack([B_norm, perturb_norm])
        data_processed.append(seq_processed)
    return data_processed, data_raw, files

def augment_prototypes(prototypes_dict):
    aug_dict = {}
    for cls, seqs in prototypes_dict.items():
        aug_list = []
        for seq in seqs:
            aug_list.append(seq)
            aug_list.append(seq[::-1])
        aug_dict[cls] = aug_list
    return aug_dict

def preprocess_sequences(data_list, target_pts=300):
    """
    统一使用线性插值将序列放缩到 target_pts 个点。
    无论原序列长短，均做全局等比例缩放，避免局部拉伸破坏物理梯度和频率特征。
    """
    processed_data = []
    for seq in data_list:
        seq = np.array(seq)
        x_old = np.arange(len(seq))
        x_new = np.linspace(0, len(seq) - 1, target_pts)
        seq_inter = np.zeros((target_pts, seq.shape[1]))
        for i in range(seq.shape[1]):
            seq_inter[:, i] = np.interp(x_new, x_old, seq[:, i])
        processed_data.append(seq_inter)
    return np.array(processed_data)


# %%
class TimeSeriesDataset(Dataset):
    def __init__(self, data, phys_features=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.phys_features = phys_features # 新增

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.phys_features is not None:
            return self.data[idx], self.phys_features[idx]
        return self.data[idx]
    
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

    def forward(self, x, return_embedding=False):
        seq_len = x.size(1)

        latent_z = self.encode(x)  # [batch, 64]
        z_rep = latent_z.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, 300, 64]
        dec_out, _ = self.decoder_lstm(z_rep)  # [batch, 300, 256]
        reconstructed = self.decoder_fc(dec_out)  # [batch, 300, 4]

        if return_embedding:
            return reconstructed, latent_z
        return reconstructed


# %%
class PrototypeDataset(Dataset):
    def __init__(self, prototypes_dict, phys_features_dict):
        self.data = []
        self.labels = []
        self.phys_features = []
        self.class_to_idx = {cls: i for i, cls in enumerate(prototypes_dict.keys())}
        
        for cls, seqs in prototypes_dict.items():
            self.data.extend(seqs)
            self.labels.extend([self.class_to_idx[cls]] * len(seqs))
            self.phys_features.extend(phys_features_dict[cls])
            
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.phys_features = torch.tensor(np.array(self.phys_features), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.phys_features[idx]


# %%
import torch

def extract_physical_features_batch(data_batch, device):
    # data_batch: [N, 300, 4]
    B = data_batch[:, :, 0]
    bz = data_batch[:, :, 1]
    bmax = data_batch[:, :, 2]
    bmin = data_batch[:, :, 3]
    N = data_batch.size(0)
    batch_indices = torch.arange(N, device=device)
    
    # =========================================================================
    # 0. 基础导数预计算 (统一计算，方便后续提取)
    # =========================================================================
    dot_B = torch.zeros_like(B)
    dot_B[:, :-1] = torch.abs(B[:, 1:] - B[:, :-1])
    dot_B[:, -1] = dot_B[:, -2]
    
    dot_bz = torch.zeros_like(bz)
    dot_bz[:, :-1] = torch.abs(bz[:, 1:] - bz[:, :-1])
    dot_bz[:, -1] = dot_bz[:, -2]
    
    dot_bmax = torch.zeros_like(bmax)
    dot_bmax[:, :-1] = torch.abs(bmax[:, 1:] - bmax[:, :-1])
    dot_bmax[:, -1] = dot_bmax[:, -2]

    # 1. 核心门控指标：压缩性指标 (决定后续特征是否激活)
    bz_sq_max = torch.max(bz**2, dim=1)[0]
    bperp_sq_max = torch.max(bmax**2 + bmin**2, dim=1)[0]
    comp_index = torch.sqrt(bz_sq_max / (bperp_sq_max + 1e-6))
    
    # 定义物理门控掩码 (仅保留阿尔芬结构的硬门控)
    mask_alfven = comp_index < 1.0     # 阿尔芬结构门控
    mask_comp = comp_index > 0.5     # 压缩性结构门控

    def get_abs_skewness(x):
        """计算序列的绝对偏度：E[(x-mu)^3] / sigma^3"""
        mu = torch.mean(x, dim=1, keepdim=True)
        sigma = torch.std(x, dim=1, keepdim=True)
        # 计算三阶标准矩
        skew = torch.mean(((x - mu) / (sigma + 1e-6))**3, dim=1)
        return torch.abs(skew)

    # =========================================================================
    # A. 阿尔芬结构专属判据 (保留 mask_alfven 门控)
    # =========================================================================
    
    # (0) 极化比: max(|b_min|) / max(|b_max|)
    max_abs_bmin = torch.max(torch.abs(bmin), dim=1)[0]
    max_abs_bmax = torch.max(torch.abs(bmax), dim=1)[0]
    raw_pol_ratio = max_abs_bmin / (max_abs_bmax + 1e-6)
    pol_ratio = torch.where(mask_alfven, raw_pol_ratio, torch.zeros_like(raw_pol_ratio))
    
    # (4) b_max 和 b_min 的最大互相关性
    def get_max_corr_pair(x, y):
        N_pts = x.size(1)
        x_norm = (x - x.mean(dim=1, keepdim=True)) 
        y_norm = (y - y.mean(dim=1, keepdim=True))
        pad_size = N_pts * 2
        X_freq = torch.fft.rfft(x_norm, n=pad_size, dim=1)
        Y_freq = torch.fft.rfft(y_norm, n=pad_size, dim=1)
        corr_freq = X_freq * torch.conj(Y_freq)
        cross_corr = torch.fft.irfft(corr_freq, n=pad_size, dim=1)
        x_energy = torch.sqrt(torch.sum(x_norm**2, dim=1) + 1e-8)
        y_energy = torch.sqrt(torch.sum(y_norm**2, dim=1) + 1e-8)
        max_corr = torch.max(torch.abs(cross_corr), dim=1)[0]
        return max_corr / (x_energy * y_energy + 1e-8)
    raw_corr_bmax_bmin = get_max_corr_pair(bmax, bmin)
    # 增加双重条件：极化比>0.2 且 属于阿尔芬结构门控
    condition_corr = (pol_ratio > 0.2) & mask_alfven
    corr_bmax_bmin = torch.where(condition_corr, raw_corr_bmax_bmin, torch.zeros_like(raw_corr_bmax_bmin))
    
    # (5) b_max 的自相关相位
    import torch.nn.functional as F
    def get_generalized_freq(x, device):
        N = x.size(1)
        # 1. 去均值标准化
        x_norm = x - x.mean(dim=1, keepdim=True)
        # 2. 计算自相关
        pad_size = N * 2
        X_freq = torch.fft.rfft(x_norm, n=pad_size, dim=1)
        autocorr = torch.fft.irfft(X_freq * torch.conj(X_freq), n=pad_size, dim=1)[:, :N]
        autocorr = autocorr / (autocorr[:, 0:1] + 1e-8)
        # 3. 寻找所有“极大值点” (Local Maxima)
        # 通过比较相邻点找到所有波峰
        is_max = (autocorr[:, 1:-1] > autocorr[:, :-2]) & (autocorr[:, 1:-1] > autocorr[:, 2:])
        # 4. 阈值筛选 + 第一个显著峰
        # 找“第一个”相关性超过 0.4 且具有显著性的峰
        # 如果没有任何峰超过阈值，它就是单周期 (freq=1)
        freq_est = torch.ones(x.size(0), device=device) # 默认频率为 1
        for i in range(x.size(0)):
            # 找到该样本所有的极大值索引
            peak_indices = torch.where(is_max[i])[0] + 1 
            # 过滤掉靠近原点（Lag < 20）的干扰点
            valid_peaks = peak_indices[peak_indices > 20]
            if len(valid_peaks) > 0:
                # 在这些峰里，找到相关系数最高的那个
                best_peak_idx = valid_peaks[torch.argmax(autocorr[i, valid_peaks])]
                best_corr_val = autocorr[i, best_peak_idx]
                # 如果这个最强峰的相关性足够高（ > 0.4），则承认它
                if best_corr_val > 0.4:
                    freq_est[i] = N / best_peak_idx.float()
        return freq_est
    dom_freq = get_generalized_freq(bmax, device)
    dom_freq = torch.where(mask_alfven, dom_freq, torch.zeros_like(dom_freq))

    
    # (8)(9) B最小时，dot_bmax 凸起的程度和 b_max 的大小
    def calc_sheet_reversal_criterion(B_full, b_max, search_range=100):
        """
        B_full: [Batch, Length] - 总磁场强度 B
        b_max:  [Batch, Length] - 主翻转分量
        search_range: 向两侧搜索极值的最大距离
        """
        batch_size, seq_len = B_full.shape
        device = B_full.device
        # 1. 找到 B 全局最小点索引
        idx_min_B = torch.argmin(B_full, dim=1)
        # 2. 检查中心窗口 (+/- 10) 是否发生变号
        range_tensor = torch.arange(seq_len, device=device).unsqueeze(0)
        center_mask = (range_tensor >= (idx_min_B.unsqueeze(1) - 10)) & \
                      (range_tensor <= (idx_min_B.unsqueeze(1) + 10))
        signs = torch.sign(b_max)
        win_signs = torch.where(center_mask, signs, signs[torch.arange(batch_size), idx_min_B].unsqueeze(1))
        has_flip = (torch.max(win_signs, dim=1).values != torch.min(win_signs, dim=1).values)
        left_extrema_val = torch.zeros(batch_size, device=device)
        right_extrema_val = torch.zeros(batch_size, device=device)
        # 3. 寻找向外延伸的第一个局部极值
        for i in range(batch_size):
            if not has_flip[i]: continue
            center = idx_min_B[i]
            # --- 右侧搜索：从 center 向右 ---
            r_end = min(seq_len, center + search_range + 1)
            r_segment = b_max[i, center : r_end]
            if r_segment.numel() > 1:
                # 计算相邻点的差值 (类似导数)
                diffs_r = torch.diff(r_segment)
                signs_r = torch.sign(diffs_r)
                # 寻找导数符号发生变化的位置 (即出现极值或平缓区)
                changes_r = torch.nonzero(signs_r[:-1] != signs_r[1:]).squeeze(1)
                if len(changes_r) > 0:
                    # +1 是因为 diff 会使得索引偏移，取变号后的那个点作为极值点
                    right_extrema_val[i] = r_segment[changes_r[0] + 1]
                else:
                    # 如果在 search_range 内单调递增/递减没有极值，则取边界点
                    right_extrema_val[i] = r_segment[-1]
            elif r_segment.numel() == 1:
                right_extrema_val[i] = r_segment[0]
            # --- 左侧搜索：从 center 向左 ---
            l_start = max(0, center - search_range)
            l_segment = b_max[i, l_start : center + 1]
            if l_segment.numel() > 1:
                # 关键：将左侧切片反转，使其物理意义变为“从 center 向左延伸”
                rev_l_segment = torch.flip(l_segment, dims=[0])
                diffs_l = torch.diff(rev_l_segment)
                signs_l = torch.sign(diffs_l)
                changes_l = torch.nonzero(signs_l[:-1] != signs_l[1:]).squeeze(1)
                
                if len(changes_l) > 0:
                    left_extrema_val[i] = rev_l_segment[changes_l[0] + 1]
                else:
                    left_extrema_val[i] = rev_l_segment[-1]
            elif l_segment.numel() == 1:
                left_extrema_val[i] = l_segment[0]
        # 最终判定：变号且两翼（第一个局部）极值绝对值均 > 0.5
        is_current_sheet = has_flip & (torch.abs(left_extrema_val) > 0.5) & (torch.abs(right_extrema_val) > 0.5)
        # 返回得分：如果是电流片，得分 = 1.0 - |b_max在B最小时的值|
        val_at_min = b_max[batch_indices, idx_min_B]
        score = 1.0 - torch.abs(val_at_min)
        score = torch.where(is_current_sheet, score, torch.zeros_like(score))
        return score
    idx_min_B = torch.argmin(B, dim=1)
    peakiness_dot_bmax = dot_bmax[batch_indices, idx_min_B]
    peakiness_dot_bmax = torch.where(mask_alfven, peakiness_dot_bmax, torch.zeros_like(peakiness_dot_bmax))
    b_max_flipscore = calc_sheet_reversal_criterion(B, bmax, search_range=100)
    b_max_flipscore = torch.where(mask_alfven, b_max_flipscore, torch.zeros_like(b_max_flipscore))

    # (17) b_max梯度的偏度
    diff_bmax = bmax[:, 1:] - bmax[:, :-1]
    abs_skew_grad_bmax = get_abs_skewness(diff_bmax)
    abs_skew_grad_bmax = torch.where(mask_alfven, abs_skew_grad_bmax, torch.zeros_like(abs_skew_grad_bmax))


    # =========================================================================
    # B. 压缩性结构专属判据 (移除 mask_comp 硬截断，保留物理连续性)
    # =========================================================================

    # (2) b_z和B 扰动凹陷或凸起程度
    idx_max_bz = torch.argmax(torch.abs(bz), dim=1)
    bz_dip = bz[batch_indices, idx_max_bz]
    idx_max_B = torch.argmax(torch.abs(B), dim=1)
    B_dip = B[batch_indices, idx_max_B]
    B_dip = torch.where(mask_comp, B_dip, torch.zeros_like(B_dip))

    # (6) 激波判据：b_z最大值的绝对值减最小值的绝对值
    b_z_max_ = torch.max(bz, dim=1)[0]
    b_z_min_ = torch.min(bz, dim=1)[0]
    R_jump = torch.abs(b_z_max_) - torch.abs(b_z_min_) # 接近0：shock；接近1：soliton；接近-1：hole
    R_jump = torch.where(mask_comp, R_jump, torch.zeros_like(R_jump))

    # (7) 渐近不对称比: 跃变前后B均值差的绝对值 / 局部噪声 (激波>>1, 磁洞/孤子≈0)
    dB = torch.abs(B[:, 1:] - B[:, :-1])
    ramp = torch.argmax(dB, dim=1)  # [N], 0..298
    idx = torch.arange(300, device=device).unsqueeze(0)  # [1, 300]
    ramp_col = ramp.unsqueeze(1)  # [N, 1]
    up_mask = (idx >= ramp_col - 50) & (idx <= ramp_col - 15)
    dn_mask = (idx >= ramp_col + 15) & (idx <= ramp_col + 50)
    B_up_mean = (B * up_mask.float()).sum(dim=1) / (up_mask.sum(dim=1).float() + 1e-6)
    B_dn_mean = (B * dn_mask.float()).sum(dim=1) / (dn_mask.sum(dim=1).float() + 1e-6)
    jump = torch.abs(B_dn_mean - B_up_mean)
    # 计算局部std: E[X^2] - E[X]^2
    B_up_std = torch.sqrt(((B**2 * up_mask.float()).sum(dim=1) / (up_mask.sum(dim=1).float() + 1e-6) - B_up_mean**2).clamp(min=0))
    B_dn_std = torch.sqrt(((B**2 * dn_mask.float()).sum(dim=1) / (dn_mask.sum(dim=1).float() + 1e-6) - B_dn_mean**2).clamp(min=0))
    noise = torch.maximum(B_up_std, B_dn_std)
    asym_B = torch.where(mask_comp, jump / (noise + 1e-6), torch.zeros_like(jump))

    # (10) dot_B 的全局峰度 (Kurtosis)
    mean_dot_B = torch.mean(dot_B, dim=1, keepdim=True)
    std_dot_B = torch.std(dot_B, dim=1, keepdim=True)
    kurt_dot_B = torch.mean(((dot_B - mean_dot_B) / (std_dot_B + 1e-6))**4, dim=1) / 10.0
    kurt_dot_B = torch.where(mask_comp, kurt_dot_B, torch.zeros_like(kurt_dot_B))

    # (11) dot_bz 的全局峰度 (Kurtosis)
    mean_dot_bz = torch.mean(dot_bz, dim=1, keepdim=True)
    std_dot_bz = torch.std(dot_bz, dim=1, keepdim=True)
    kurt_dot_bz = torch.mean(((dot_bz - mean_dot_bz) / (std_dot_bz + 1e-6))**4, dim=1) / 10.0
    kurt_dot_bz = torch.where(mask_comp, kurt_dot_bz, torch.zeros_like(kurt_dot_bz))

    # (12)(13) b_z和b_max穿过 ±0.5 的次数 (反映震荡结构的复杂程度)
    def calc_criterion_16(bz, threshold=0.5):
        """
        bz: [Batch, Length] 的张量
        计算 bz 穿过 threshold 和 -threshold 的总次数并除以 4
        """
        # 1. 计算穿过 +0.5 的次数
        # 当相邻两个点的符号不同时，代表发生了一次穿越
        cross_plus = torch.diff((bz > threshold).int(), dim=1).abs().sum(dim=1)
        # 2. 计算穿过 -0.5 的次数
        cross_minus = torch.diff((bz < -threshold).int(), dim=1).abs().sum(dim=1)
        # 3. 求和并归一化
        # 对于一个标准的双极性脉冲 (0 -> 1 -> -1 -> 0):
        # 穿过 0.5 两次 (上一、下一)，穿过 -0.5 两次 (下一、上一)，总计 4 次。4/4 = 1.0
        complexity_index = (cross_plus + cross_minus).float() / 4.0
        score = torch.exp(-(complexity_index-1)**2 / 0.3 **2) # 距离标准值1越远，得分越低
        return score
    complexity_index_bz = calc_criterion_16(bz)
    complexity_index_bmax = calc_criterion_16(bmax)

    # (14) B 场与 tanh 模板的最大相关性
    def get_max_corr_template(x, y_template):
        """
        计算 batch x 与单个模板 y_template 之间的最大互相关性 (位移无关)
        """
        N_pts = x.size(1)
        # 模板扩展到 batch 大小
        y = y_template.expand(x.size(0), -1)
        x_norm = (x - x.mean(dim=1, keepdim=True)) 
        y_norm = (y - y.mean(dim=1, keepdim=True))
        pad_size = N_pts * 2
        X_freq = torch.fft.rfft(x_norm, n=pad_size, dim=1)
        Y_freq = torch.fft.rfft(y_norm, n=pad_size, dim=1)
        corr_freq = X_freq * torch.conj(Y_freq)
        cross_corr = torch.fft.irfft(corr_freq, n=pad_size, dim=1)
        x_energy = torch.sqrt(torch.sum(x_norm**2, dim=1) + 1e-8)
        y_energy = torch.sqrt(torch.sum(y_norm**2, dim=1) + 1e-8)
        # 使用 abs 是为了同时兼容正向和反向的波形 (+/- 符号)
        max_corr = torch.max(torch.abs(cross_corr), dim=1)[0]
        return max_corr / (x_energy * y_energy + 1e-8)
    t1 = torch.linspace(-100, 100, 300, device=device)
    tanh_template = torch.tanh(t1).unsqueeze(0) # [1, 300]
    corr_shock_B = get_max_corr_template(B, tanh_template)
    corr_shock_B = torch.where(mask_comp, corr_shock_B, torch.zeros_like(corr_shock_B))

    # (15)(16) 梯度（斜率）的偏度 (反映跳变的方向性)
    # 激波的斜率分布是单向极值（极度偏斜），震荡结构的斜率分布是对称的（偏度近0）
    diff_B = B[:, 1:] - B[:, :-1]
    diff_bz = bz[:, 1:] - bz[:, :-1]
    abs_skew_grad_B = get_abs_skewness(diff_B)
    abs_skew_grad_bz = get_abs_skewness(diff_bz)
    # 采用压缩性门控
    abs_skew_grad_B = torch.where(mask_comp, abs_skew_grad_B, torch.zeros_like(abs_skew_grad_B))
    abs_skew_grad_bz = torch.where(mask_comp, abs_skew_grad_bz, torch.zeros_like(abs_skew_grad_bz))

    return torch.stack([
        pol_ratio,             # 0
        comp_index,            # 1
        bz_dip,                # 2
        B_dip,                 # 3
        corr_bmax_bmin,        # 4
        dom_freq,              # 5
        R_jump,                # 6
        asym_B,                # 7
        peakiness_dot_bmax,    # 8
        b_max_flipscore,       # 9
        kurt_dot_B,            # 10
        kurt_dot_bz,           # 11
        complexity_index_bz,   # 12
        complexity_index_bmax, # 13
        corr_shock_B,          # 14
        abs_skew_grad_B,       # 15
        abs_skew_grad_bz,      # 16
        abs_skew_grad_bmax     # 17
    ], dim=1)


# %%
def physical_contrastive_loss(embeddings, labels, phys_features, margin=1.0, feat_weights=None):
    N = embeddings.size(0)
    device = embeddings.device

    # phys_features 是从 DataLoader 直接传进来的 [N, feature_dim]
    # 只需做标准化和 cdist
    feat_mean = phys_features.mean(dim=0, keepdim=True)
    feat_std = phys_features.std(dim=0, keepdim=True) + 1e-6
    phys_features_norm = (phys_features - feat_mean) / feat_std

    # 应用特征权重 (权重乘在归一化特征上, sqrt 因为后续用 L2 距离)
    if feat_weights is not None:
        phys_features_norm = phys_features_norm * feat_weights.to(device).sqrt()

    # 计算物理差异矩阵 D_phys 和隐空间距离矩阵 d_ij
    phys_diff_matrix = torch.cdist(phys_features_norm, phys_features_norm, p=2)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    # --- 构建掩码 ---
    # 标签矩阵：只有当两个样本都有标签且标签相同时为 1
    # 注意：无标签样本 labels 为 -1
    has_label = (labels >= 0).unsqueeze(1) & (labels >= 0).unsqueeze(0)
    same_label = (labels.unsqueeze(1) == labels.unsqueeze(0)) & has_label
    diff_label = (labels.unsqueeze(1) != labels.unsqueeze(0)) & has_label

    # 物理相似性掩码：即使没标签，如果物理差异很小，也视为“准同类”
    phys_similar = (phys_diff_matrix < 0.8) 
    phys_dissimilar = (phys_diff_matrix > 4.0)

    mask = torch.eye(N, device=device)

    # 正样本对损失
    # 1. 标签明确相同且物理特征相近 2. 或者虽然没标签但物理特征极度接近
    pos_mask = (same_label | (~has_label & phys_similar)) * (1 - mask)
    if pos_mask.sum() > 0:
        pos_loss = (pos_mask * (dist_matrix**2 + 0.2 * phys_diff_matrix)).sum() / pos_mask.sum()
    else:
        pos_loss = torch.tensor(0.0, device=device)

    # 负样本对损失
    # 1. 标签明确不同 2. 或者虽然没标签但物理特征差异极大
    neg_mask = (diff_label | (~has_label & phys_dissimilar)) * (1 - mask)
    if neg_mask.sum() > 0:
        dynamic_margin = margin + 0.5 * phys_diff_matrix # 物理差异越大，排斥越狠
        neg_loss = (neg_mask * torch.clamp(dynamic_margin - dist_matrix, min=0)**2).sum() / neg_mask.sum()
    else:
        neg_loss = torch.tensor(0.0, device=device)

    return pos_loss + neg_loss


# %%
def calc_invariant_mse(pred, target, max_shift=50):
    B, L, C = pred.shape
    target_flipped = torch.flip(target, dims=[1])
    best_loss = torch.full((B, C), float('inf'), device=pred.device)
    pred_t = pred.transpose(1, 2)
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            p_shifted_t = pred_t
        elif shift > 0:
            p_shifted_t = F.pad(pred_t[:, :, :-shift], (shift, 0), mode='replicate')
        else:
            s = -shift
            p_shifted_t = F.pad(pred_t[:, :, s:], (0, s), mode='replicate')
        p_shifted = p_shifted_t.transpose(1, 2)
        mse_normal = F.mse_loss(p_shifted, target, reduction='none').mean(dim=1)
        mse_flipped = F.mse_loss(p_shifted, target_flipped, reduction='none').mean(dim=1)
        current_min = torch.min(mse_normal, mse_flipped)
        best_loss = torch.min(best_loss, current_min)
    return best_loss.mean(dim=0)


# %%

def train_autoencoder(model, train_dataloader, val_dataloader, proto_dataloader, device,
                      epochs=100, lr=0.001, patience=10, best_model_path=None,
                      max_lambda_contrastive=0.1, step_lambda_contrastive=0.01, start_lambda_contrastive=0.0, max_shift=20,
                      feat_weights=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, min_lr=1e-4)
    loss_weights = torch.tensor([0.5, 2.0, 1.0, 1.0]).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []
    all_train_loss_list = [[] for _ in range(5)]  # B, bz, bmax, bmin, contrastive
    all_val_loss_list = [[] for _ in range(5)]
    best_model_wts = copy.deepcopy(model.state_dict())
    
    epoch_bar = tqdm(range(epochs), desc='Overall Progress')
    
    for epoch in epoch_bar:
        current_lambda = min(max_lambda_contrastive, step_lambda_contrastive * epoch + start_lambda_contrastive)

        # --- 训练阶段 ---
        model.train()
        total_train_loss, total_train_con = 0, 0
        total_train_errs = np.zeros(4)
        
        proto_iter = itertools.cycle(proto_dataloader)
        
        for batch_item in train_dataloader:
            x, x_phys = batch_item
            x, x_phys = x.to(device), x_phys.to(device)
            
            p_data, p_labels, p_phys = next(proto_iter)
            p_data, p_labels, p_phys = p_data.to(device), p_labels.to(device), p_phys.to(device)
            
            optimizer.zero_grad()
            
            # --- A. 重建 ---
            output = model(x)
            mse_per_channel = calc_invariant_mse(output, x, max_shift=max_shift)
            weighted_errs = (mse_per_channel * loss_weights) / 4.0

            # --- B. 编码 (对比路径独立 encode，避免 MSE 梯度干扰对比梯度) ---
            z_x = model.encode(x)
            z_p = model.encode(p_data)
            
            loss_rec = (1 - current_lambda) * weighted_errs.sum()
            
            # --- C. 对比路径 ---
            combined_emb = torch.cat([z_x, z_p], dim=0)
            combined_phys = torch.cat([x_phys, p_phys], dim=0)
            x_labels = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            combined_labels = torch.cat([x_labels, p_labels], dim=0)
            loss_con = physical_contrastive_loss(combined_emb, combined_labels, combined_phys, feat_weights=feat_weights)

            weighted_con = current_lambda * loss_con
            loss = loss_rec + weighted_con            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_errs += weighted_errs.detach().cpu().numpy()
            total_train_con += weighted_con.item()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        t_errs = total_train_errs / len(train_dataloader)
        t_con = total_train_con / len(train_dataloader)
        train_loss_list.append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        total_val_rec_loss, total_val_con_loss = 0, 0
        val_errs_detailed = np.zeros(4)
        val_proto_iter = itertools.cycle(proto_dataloader)
        
        with torch.no_grad():
            for v_batch_item in val_dataloader:
                vx, vx_phys = v_batch_item
                vx, vx_phys = vx.to(device), vx_phys.to(device)
                
                v_output = model(vx)
                v_mse_per_channel = calc_invariant_mse(v_output, vx, max_shift=max_shift)
                v_weighted_errs = (v_mse_per_channel * loss_weights) / 4.0
                total_val_rec_loss += v_weighted_errs.sum().item()
                val_errs_detailed += v_weighted_errs.cpu().numpy()

                vp_data, vp_labels, vp_phys = next(val_proto_iter)
                vp_data, vp_labels, vp_phys = vp_data.to(device), vp_labels.to(device), vp_phys.to(device)

                vz_x = model.encode(vx)
                vz_p = model.encode(vp_data)
                
                v_comb_emb = torch.cat([vz_x, vz_p], dim=0)
                v_comb_phys = torch.cat([vx_phys, vp_phys], dim=0)
                vx_labels = torch.full((vx.size(0),), -1, dtype=torch.long, device=device)
                v_comb_labels = torch.cat([vx_labels, vp_labels], dim=0)
                v_loss_con = physical_contrastive_loss(v_comb_emb, v_comb_labels, v_comb_phys, feat_weights=feat_weights)
                total_val_con_loss += v_loss_con.item()

        avg_val_rec = (1 - current_lambda) * total_val_rec_loss / len(val_dataloader)
        avg_val_con = current_lambda * (total_val_con_loss / len(val_dataloader))
        avg_val_loss = avg_val_rec + avg_val_con
        v_errs = val_errs_detailed / len(val_dataloader)
        val_loss_list.append(avg_val_loss)
        
        t_rec_total = (1 - current_lambda) * t_errs.sum()
        v_rec_total = (1 - current_lambda) * v_errs.sum()
        summary = (
            f"TRA: {avg_train_loss:.4f} | Rec: {t_rec_total:.4f} + Con: {t_con:.4f} || "
            f"VAL: {avg_val_loss:.4f} | Rec: {v_rec_total:.4f} + Con: {avg_val_con:.4f}"
        )
        epoch_bar.set_postfix_str(summary)

        for i in range(4):
            all_train_loss_list[i].append((1 - current_lambda) * t_errs[i])
            all_val_loss_list[i].append((1 - current_lambda) * v_errs[i])
        all_train_loss_list[4].append(t_con)
        all_val_loss_list[4].append(avg_val_con)

        # 早停：以总误差为准，但仅在 current_lambda 达到最大值后才触发
        if current_lambda >= max_lambda_contrastive:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_wts = copy.deepcopy(model.state_dict())
                if best_model_path:
                    torch.save(best_model_wts, best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                epoch_bar.write(f"Early stopping at epoch {epoch+1}.")
                break

        scheduler.step(avg_val_loss)

    model.load_state_dict(best_model_wts)
    return train_loss_list, val_loss_list, all_train_loss_list, all_val_loss_list


# %% [markdown]
# ## 主程序

# %%
workspace = '..\\'
trainset_path = os.path.join(workspace, 'trainset')
samples_path = os.path.join(workspace, 'samples_clean')

# --- 1. 加载数据 ---
data_all_processed, data_all_raw, data_all_files = load_data(trainset_path)

# --- 同步随机打乱 ---
combined = list(zip(data_all_processed, data_all_raw, data_all_files))
import random
seed = 42
random.seed(seed) 
random.shuffle(combined)
data_all_processed, data_all_raw, data_all_files = zip(*combined)
data_all_processed = list(data_all_processed)
data_all_raw = list(data_all_raw)
data_all_files = list(data_all_files)

print(f"Total dataset has {len(data_all_files)} files. Data has been shuffled.")

# --- 2. 执行 8:1:1 划分 ---
total_samples = len(data_all_processed)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)

train_data_processed = data_all_processed[:train_size]
val_data_processed = data_all_processed[train_size:train_size+val_size]
test_data_processed = data_all_processed[train_size+val_size:]
test_data_raw = data_all_raw[train_size+val_size:]
test_files = data_all_files[train_size+val_size:]

# --- 3. 加载并增强范本 (Prototypes) ---
classes = ['sheet', 'vortex chain', 'c vortex', 'l vortex', 'hole', 'soliton', 'shock', 'alfen dis']
prototypes_processed_raw = {}
for cls in classes:
    cls_path = os.path.join(samples_path, cls)
    if not os.path.exists(cls_path): continue
    data_p, _, _ = load_data(cls_path)
    prototypes_processed_raw[cls] = data_p
prototypes_processed = augment_prototypes(prototypes_processed_raw)

# --- 4. 确定最大长度并执行预处理 ---
target_pts = 300
print(f"Executing Preprocessing (Interpolation) to {target_pts} points...")
X_train_pad = preprocess_sequences(train_data_processed, target_pts)
X_val_pad = preprocess_sequences(val_data_processed, target_pts)
X_test_pad = preprocess_sequences(test_data_processed, target_pts)

prototypes_pad = {}
for cls, seqs in prototypes_processed.items():
    prototypes_pad[cls] = preprocess_sequences(seqs, target_pts)

print("Pre-calculating physical features for training set...")
X_train_tensor = torch.tensor(X_train_pad, dtype=torch.float32).to(device)
train_phys_feats = extract_physical_features_batch(X_train_tensor, device).cpu()
X_val_tensor = torch.tensor(X_val_pad, dtype=torch.float32).to(device)
val_phys_feats = extract_physical_features_batch(X_val_tensor, device).cpu()
prototypes_phys_feats = {}
for cls, seqs in prototypes_pad.items():
    seqs_tensor = torch.tensor(seqs, dtype=torch.float32).to(device)
    prototypes_phys_feats[cls] = extract_physical_features_batch(seqs_tensor, device).cpu().numpy()

# 5. 构建 Dataset
train_dataset = TimeSeriesDataset(X_train_pad, phys_features=train_phys_feats)
val_dataset = TimeSeriesDataset(X_val_pad, phys_features=val_phys_feats)
proto_dataset = PrototypeDataset(prototypes_pad, prototypes_phys_feats)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=False)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, pin_memory=False)
proto_dataloader = DataLoader(proto_dataset, batch_size=128, shuffle=True, pin_memory=False)

# --- 6. 训练 ---
model = BiAutoencoder(input_size=4, cnn_channels=16, hidden_size=128, num_layers=2, latent_dim=64).to(device)
print(f"Starting Autoencoder training on {device}...")
torch.cuda.empty_cache()
# 特征权重 (索引见CLAUDE.md特征表)
feat_weights = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

train_loss_list, val_loss_list, all_train_loss_list, all_val_loss_list = train_autoencoder(
    model, train_dataloader, val_dataloader, proto_dataloader, device,
    epochs=100, lr=0.005, patience=10, max_lambda_contrastive=0.05, step_lambda_contrastive=0, start_lambda_contrastive=0.05, max_shift=50,
    feat_weights=feat_weights
)

# %% [markdown]
# loss = (1 - λ) × data_mse  +  λ × contrastive

# %% [markdown]
# ## loss曲线

# %%
# 保存loss图片
# 同一轮训练的所有输出保存在同一个时间戳文件夹
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
img_dir = os.path.join(workspace, 'output', 'images', timestamp)
os.makedirs(img_dir, exist_ok=True)

fig = plt.figure(figsize=(8, 4))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
loss_path = os.path.join(img_dir, 'loss_all.png')
fig.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.show()

all_train_loss_array = np.array(all_train_loss_list)
all_val_loss_array = np.array(all_val_loss_list)

fig = plt.figure(figsize=(8, 4))
all_train_mse_loss = np.sum([all_train_loss_array[i] for i in range(4)], axis=0)
plt.plot(all_train_mse_loss, label='Train Loss MSE')
plt.plot(all_train_loss_array[0], label='Train Loss B')
plt.plot(all_train_loss_array[1], label='Train Loss b_z')
plt.plot(all_train_loss_array[2], label='Train Loss b_max')
plt.plot(all_train_loss_array[3], label='Train Loss b_min')
plt.plot(all_train_loss_array[4], label='Train Loss Contrastive')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
loss_path = os.path.join(img_dir, 'loss_train.png')
fig.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(8, 4))
all_val_mse_loss = np.sum([all_val_loss_array[i] for i in range(4)], axis=0)
plt.plot(all_val_mse_loss, label='Validation Loss MSE')
plt.plot(all_val_loss_array[0], label='Validation Loss B')
plt.plot(all_val_loss_array[1], label='Validation Loss b_z')
plt.plot(all_val_loss_array[2], label='Validation Loss b_max')
plt.plot(all_val_loss_array[3], label='Validation Loss b_min')
plt.plot(all_val_loss_array[4], label='Validation Loss Contrastive')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
loss_path = os.path.join(img_dir, 'loss_val.png')
fig.savefig(loss_path, dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## 预测

# %%
from scipy.spatial.distance import cdist, pdist, squareform
from torch.utils.data import DataLoader, TensorDataset

def test_clustering(model, test_data_preprocessed, prototypes_preprocessed_dict, device, n_std=2.0, batch_size=256):
    """
    自适应聚类测试逻辑：
    1. 提取范本中心
    2. 分批次提取测试集 Embedding (防止 GPU 显存溢出)
    3. 全量分配最近邻
    4. 针对每一类测试样本的分布计算动态阈值
    5. 过滤出 neither
    6. 输出范本中心距离矩阵 (模块 3)
    """
    model.eval()
    
    # --- 1. 提取范本中心 (仅作为锚点) ---
    proto_emb = {}
    class_names = list(prototypes_preprocessed_dict.keys())
    for cls in class_names:
        data_matrix = prototypes_preprocessed_dict[cls]
        with torch.no_grad():
            # 范本数量通常不多，直接转 tensor。如果范本也很多，同样可以加 batch
            seq_tensor = torch.tensor(np.array(data_matrix), dtype=torch.float32).to(device)
            embs = model.encode(seq_tensor).cpu().numpy()
            proto_emb[cls] = np.mean(embs, axis=0)

    # --- 2. 提取测试集 Embedding (防 OOM 核心修改：分批处理) ---
    print(f"Extracting test embeddings in batches of {batch_size}...")
    test_tensor = torch.tensor(np.array(test_data_preprocessed), dtype=torch.float32)
    test_dataset = TensorDataset(test_tensor)
    # 注意：shuffle=False 必须严格保证，否则预测标签和输入数据会对不上！
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_embs = []
    with torch.no_grad():
        for batch in test_loader:
            batch_data = batch[0].to(device)
            emb = model.encode(batch_data).cpu().numpy()
            test_embs.append(emb)
    test_embs = np.vstack(test_embs)  # 拼接回 (N_test, 64) 的矩阵

    # --- 3. 强制最近邻分配 (初步归类) ---
    centers_matrix = np.array([proto_emb[cls] for cls in class_names])
    # cdist 计算 (N_test, N_proto) 的所有距离
    dist_matrix = cdist(test_embs, centers_matrix, metric='euclidean') 
    
    initial_idx = np.argmin(dist_matrix, axis=1)        # 每个样本最像哪个范本
    all_nearest_dists = np.min(dist_matrix, axis=1)     # 到最近范本的距离

    # --- 4. 基于每一类测试样本的分布计算阈值 ---
    class_thresholds = {}
    print("\n" + "-" * 55)
    print(f"{'Class':<15} | {'Test_Mean':<10} | {'Test_Std':<10} | {'Threshold':<10}")
    print("-" * 55)

    for i, cls in enumerate(class_names):
        # 找出所有初步归为该类的测试样本距离
        this_class_dists = all_nearest_dists[initial_idx == i]
        
        if len(this_class_dists) > 0:
            m = np.mean(this_class_dists)
            s = np.std(this_class_dists)
            # 计算该类的专属阈值：基于测试集表现
            t = m + n_std * s
            class_thresholds[cls] = t
            print(f"{cls:<15} | {m:<10.4f} | {s:<10.4f} | {t:<10.4f}")
        else:
            class_thresholds[cls] = 0.0
            print(f"{cls:<15} | No samples assigned.")

    # --- 5. 最终判定 (二次过滤) ---
    final_predictions = []
    for i in range(len(test_embs)):
        best_cls = class_names[initial_idx[i]]
        d = all_nearest_dists[i]
        
        # 只有在测试集自身的“势力范围”内的才保留
        if d <= class_thresholds[best_cls]:
            final_predictions.append(best_cls)
        else:
            final_predictions.append('neither')

    # --- 6. 模块三：范本中心两两之间的距离 ---
    print("\n" + "-" * 35)
    print("Prototype Centers Distance Matrix:")
    # pdist 计算两两距离，squareform 将其转为方阵
    dist_matrix_centers = squareform(pdist(centers_matrix, metric='euclidean'))
    dist_df = pd.DataFrame(dist_matrix_centers, index=class_names, columns=class_names)
    print(dist_df.round(2))
    print("-" * 35 + "\n")
            
    return final_predictions, proto_emb, class_thresholds


# %%
model = model.to(device)
predictions, final_proto_emb, thresholds = test_clustering(model, X_test_pad, prototypes_pad, device, n_std=1) # 加方差会不准
torch.save(model.state_dict(), os.path.join(workspace, 'bi_model.pth')) 
np.save(os.path.join(workspace, 'proto_emb.npy'), final_proto_emb) 
np.save(os.path.join(workspace, 'target_pts.npy'), np.array([target_pts]))
np.save(os.path.join(workspace, 'thresholds.npy'), thresholds)

# 保存输出到 output/train_result/{timestamp}/
result_dir = os.path.join(workspace, 'output', 'train_result', timestamp)
os.makedirs(result_dir, exist_ok=True)

print("-" * 35)
import pandas as pd
stats = pd.Series(predictions).value_counts()
total = len(predictions)
lines = []
for cls, count in stats.items():
    percentage = (count / total) * 100
    line = f"Cluster: {cls:<15} | Num: {count:<6} | Percent: {percentage:>6.2f}%"
    print(line)
    lines.append(line)
print("-" * 35)
print(f"Total Samples: {total}")
lines.append("-" * 35)
lines.append(f"Total Samples: {total}")

# 写入文件
result_path = os.path.join(result_dir, 'classification.txt')
with open(result_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"\nClassification results saved to: {result_path}")

# %% [markdown]
# ### TSNE图

# %%
from sklearn.manifold import TSNE

def plot_tsne(model, test_loader, prototypes_pad, predictions, device):
    """
    predictions: test_clustering 函数返回的预测结果列表
    """
    model.eval()
    all_embeddings = []
    
    # 1. 提取测试集的 Embedding
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            emb = model.encode(batch).cpu().numpy()
            all_embeddings.append(emb)
    
    # 2. 提取范本的 Embedding
    proto_embs = []
    unique_protos = list(prototypes_pad.keys())
    for cls_name in unique_protos:
        seqs = prototypes_pad[cls_name]
        seqs_tensor = torch.tensor(seqs, dtype=torch.float32).to(device)
        with torch.no_grad():
            emb = model.encode(seqs_tensor).cpu().numpy()
            proto_embs.append(emb)
    
    # 合并数据并运行 t-SNE
    test_embs = np.vstack(all_embeddings)
    all_vecs = np.vstack([test_embs, np.vstack(proto_embs)])
    
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
    vecs_2d = tsne.fit_transform(all_vecs)
    
    test_2d = vecs_2d[:len(test_embs)]
    proto_2d = vecs_2d[len(test_embs):]
    
    # --- 3. 建立颜色映射表 ---
    # 定义类别颜色
    colors_list = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    color_map = {cls: colors_list[i % len(colors_list)] for i, cls in enumerate(unique_protos)}
    color_map['neither'] = 'lightgrey' # 没认出来的设为灰色

    # 根据预测结果生成每个点的颜色列表
    point_colors = [color_map.get(p, 'lightgrey') for p in predictions]

    # 4. 绘图
    plt.figure(figsize=(14, 10))
    
    # 绘制测试集样本
    plt.scatter(test_2d[:, 0], test_2d[:, 1], c=point_colors, alpha=0.3, s=4, label='Classified Samples')
    
    # 绘制范本点：大星星，高饱和度
    start_idx = 0
    for i, cls_name in enumerate(unique_protos):
        num_samples = len(prototypes_pad[cls_name])
        end_idx = start_idx + num_samples
        
        cls_color = color_map[cls_name]
        
        # 绘制该类的所有范本点 (通常是 K 个)
        plt.scatter(proto_2d[start_idx:end_idx, 0], proto_2d[start_idx:end_idx, 1], 
                    color=cls_color, linewidth=1,
                    s=150, label=f'Proto: {cls_name}', marker='*')
        
        # 在第一个范本点旁边标注文字
        plt.text(proto_2d[start_idx, 0]+1, proto_2d[start_idx, 1]+1, 
                 cls_name, fontsize=12, fontweight='bold', color=cls_color)
        
        start_idx = end_idx

    plt.title('t-SNE Visualization: Predicted Clusters vs Prototypes')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    tsne_path = os.path.join(img_dir, 'tsne.png')
    plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
    plt.show()

test_dataloader = DataLoader(TimeSeriesDataset(X_test_pad), batch_size=128, shuffle=False, num_workers=0)
plot_tsne(model.to(device), test_dataloader, prototypes_pad, predictions, device) 


# %% [markdown]
# ### 离每个范本中心最近的 K 个原始波形

# %%
import numpy as np
import matplotlib.pyplot as plt

def verify_top_k_samples(test_embeddings, test_data_raw, test_files, proto_embs_dict,
                         predictions=None, thresholds=None, k=5, save=False, set='test'):
    """
    可视化测试集中离每个范本中心最近的 K 个原始波形。
    标题显示: 模型预测类别 vs 最近邻类别，红色=不一致。
    """
    all_class_names = list(proto_embs_dict.keys())
    all_centers = np.array([proto_embs_dict[name] for name in all_class_names])

    # 构建索引→预测类别的映射 (如果提供了 predictions)
    pred_map = None
    if predictions is not None:
        pred_map = {i: predictions[i] for i in range(len(predictions))}

    for cls_name, proto_center in proto_embs_dict.items():
        distances = np.linalg.norm(test_embeddings - proto_center, axis=1)
        closest_indices = np.argsort(distances)[:k]
        
        fig, axes = plt.subplots(2, k, figsize=(5*k, 8))
        fig.suptitle(f'Top {k} Matches for Prototype Class: {cls_name.upper()}', 
                     fontsize=18, fontweight='bold', y=1.02)
        
        for j, idx in enumerate(closest_indices):
            sample_emb = test_embeddings[idx]
            all_dists = np.linalg.norm(all_centers - sample_emb, axis=1)
            nearest_cls_idx = np.argmin(all_dists)
            nearest_cls_name = all_class_names[nearest_cls_idx]
            
            # 模型实际预测类别
            if pred_map is not None:
                model_pred = pred_map[idx]  # 可能是 'neither'
            else:
                model_pred = None

            seq = test_data_raw[idx]
            file_num = test_files[idx]
            B = seq[:, 0]
            b_z = seq[:, 1]
            b_max = seq[:, 2]
            b_min = seq[:, 3]
            
            b_sum_sq = b_z**2 + b_max**2 + b_min**2
            max_idx = np.argmax(b_sum_sq)
            min_idx = np.argmin(B)
            
            ax_top = axes[0, j] if k > 1 else axes[0]
            ax_btm = axes[1, j] if k > 1 else axes[1]

            ax_top.plot(B, color='black', linewidth=1.5)
            ax_top.axhline(np.mean(B), color='gray', linestyle='--', alpha=0.6)
            ax_top.axvline(max_idx, color='red', linestyle='--', alpha=0.4)
            ax_top.axvline(min_idx, color='green', linestyle='--', alpha=0.4)

            # 标题: 模型预测 vs 最近邻
            if model_pred is not None:
                pred_str = f"Pred: {model_pred} | NN: {nearest_cls_name}"
            else:
                pred_str = f"NN: {nearest_cls_name}"
            
            # 红色 = 模型的预测结果与当前原型不一致
            title_color = 'black' if (model_pred == cls_name) else 'red'
            ax_top.set_title(f'Dist: {distances[idx]:.3f} - {file_num}\n{pred_str}', 
                             fontsize=11, color=title_color)
            
            if j == 0:
                ax_top.set_ylabel('B', fontsize=12, fontweight='bold')
            
            ax_btm.plot(b_z, label='b_z', color='blue', alpha=0.8)
            ax_btm.plot(b_max, label='b_max', color='red', alpha=0.8)
            ax_btm.plot(b_min, label='b_min', color='green', alpha=0.8)
            ax_btm.axvline(min_idx, color='green', linestyle='--', alpha=0.4)
            ax_btm.axvline(max_idx, color='red', linestyle='--', alpha=0.4)
            ax_btm.axhline(0, color='gray', linestyle='--', alpha=0.6)
            
            if j == 0:
                ax_btm.set_ylabel('b (Perturbations)', fontsize=12, fontweight='bold')
            if j == k - 1:
                ax_btm.legend(loc='upper right', fontsize='small')
            
            ax_btm.set_xlabel('Time Step')

        plt.tight_layout()
        if save:
            save_path = os.path.join(img_dir, f'{set}_{cls_name}.png')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()



# %%
import torch
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

def classify_by_thresholds(embeddings, proto_embs_dict, thresholds):
    """根据最近邻+阈值对 embeddings 分类，返回 predictions 列表"""
    class_names = list(proto_embs_dict.keys())
    centers = np.array([proto_embs_dict[cls] for cls in class_names])
    dists = cdist(embeddings, centers, metric='euclidean')
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dists = np.min(dists, axis=1)
    
    predictions = []
    for i in range(len(embeddings)):
        cls = class_names[nearest_idx[i]]
        if cls in thresholds and nearest_dists[i] <= thresholds[cls]:
            predictions.append(cls)
        else:
            predictions.append('neither')
    return predictions


def extract_embeddings_sequentially(model, padded_data, device, batch_size=128):
    model.eval()
    temp_dataset = TimeSeriesDataset(padded_data)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
    
    all_embs = []
    with torch.no_grad():
        for batch in temp_loader:
            emb = model.encode(batch.to(device)).cpu().numpy()
            all_embs.append(emb)
    return np.vstack(all_embs)

# --- 准备所有数据集的 Raw 数据和文件名 ---
train_data_raw = data_all_raw[:train_size]
train_files = data_all_files[:train_size]

val_data_raw = data_all_raw[train_size : train_size + val_size]
val_files = data_all_files[train_size : train_size + val_size]

# --- 顺序提取所有集合的 Embeddings ---
print("Extracting aligned embeddings for all splits...")
train_embs = extract_embeddings_sequentially(model, X_train_pad, device)
val_embs = extract_embeddings_sequentially(model, X_val_pad, device)
test_embs = extract_embeddings_sequentially(model, X_test_pad, device)

# --- 计算类中心 ---
proto_embs_dict = {}
for cls_name, seqs in prototypes_pad.items():
    seqs_tensor = torch.tensor(seqs, dtype=torch.float32).to(device)
    with torch.no_grad():
        emb = model.encode(seqs_tensor).cpu().numpy()
        proto_embs_dict[cls_name] = np.mean(emb, axis=0)

# --- 为 train/val 生成 predictions (test 已在主程序中通过 test_clustering 生成) ---
train_preds = classify_by_thresholds(train_embs, proto_embs_dict, thresholds)
val_preds = classify_by_thresholds(val_embs, proto_embs_dict, thresholds)

# --- 执行可视化 ---

print("\n--- Visualizing TOP-K Matches in TRAINING SET ---")
verify_top_k_samples(
    test_embeddings=train_embs, 
    test_data_raw=train_data_raw, 
    test_files=train_files,
    proto_embs_dict=proto_embs_dict,
    predictions=train_preds,
    k=10
)

print("\n--- Visualizing TOP-K Matches in VALIDATION SET ---")
verify_top_k_samples(
    test_embeddings=val_embs, 
    test_data_raw=val_data_raw, 
    test_files=val_files,
    proto_embs_dict=proto_embs_dict,
    predictions=val_preds,
    k=10
)

print("\n--- Visualizing TOP-K Matches in TEST SET ---")
verify_top_k_samples(
    test_embeddings=test_embs, 
    test_data_raw=test_data_raw, 
    test_files=test_files,
    proto_embs_dict=proto_embs_dict,
    predictions=predictions,
    k=10,
    save=True, 
    set='test'
)


# %% [markdown]
# ### 每个种类绘制距离分布直方图

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_distance_histograms(test_embeddings, predictions, proto_emb, bin_size=0.5, max_dist=10):
    """
    为每个种类绘制距离分布直方图
    test_embeddings: 测试集的 Embedding 矩阵 (N, 64)
    predictions: test_clustering 返回的预测列表
    proto_emb: test_clustering 返回的各类别中心字典
    bin_size: 直方图分辨率 (默认 0.5)
    max_dist: 绘图显示的最大距离门槛 (默认 10)
    """
    # 1. 提取所有有效的类别名（排除 'neither'）
    unique_classes = [cls for cls in proto_emb.keys() if cls != 'neither']
    num_classes = len(unique_classes)
    
    # 设置子图布局 (根据类别数量自动调整，每行最多 3 个图)
    cols = 3
    rows = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()
    
    # 设置统一的 bins
    bins = np.arange(0, max_dist + bin_size, bin_size)

    for i, cls in enumerate(unique_classes):
        # 2. 找到预测为该类别的所有样本索引
        indices = [idx for idx, pred in enumerate(predictions) if pred == cls]
        
        if not indices:
            axes[i].text(0.5, 0.5, f'No samples for {cls}', ha='center')
            axes[i].set_title(f"Class: {cls}")
            continue
            
        # 3. 计算这些样本到该类中心的距离
        class_embs = test_embeddings[indices]
        center_vec = proto_emb[cls]
        dists = np.linalg.norm(class_embs - center_vec, axis=1)
        
        # 4. 绘图
        axes[i].hist(dists, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 统计一些关键信息标注在图上
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        axes[i].axvline(mean_d, color='red', linestyle='--', label=f'Mean: {mean_d:.2f}')
        
        axes[i].set_title(f"Class: {cls.upper()} (N={len(indices)})", fontsize=14, fontweight='bold')
        axes[i].set_xlabel("Distance to Prototype")
        axes[i].set_ylabel("Frequency")
        axes[i].set_xlim(0, max_dist)
        axes[i].grid(axis='y', alpha=0.3)
        axes[i].legend()

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Sample Distance Distribution per Class", fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()


# 之前必须已经运行了：
# predictions, final_proto_emb = test_clustering(...)
# test_embs = model.encode(...) 得到的嵌入向量

plot_distance_histograms(
    test_embeddings=test_embs, 
    predictions=predictions, 
    proto_emb=final_proto_emb, 
    bin_size=0.5,   # 分辨率
    max_dist=20     # 最大显示距离
)


# %% [markdown]
# ## 抽查结果

# %% [markdown]
# ### 绘制指定类别的测试集样本

# %%
def plot_class_samples(target_class, start_offset, total_count=100, per_fig=10):
    """
    target_class: 指定要绘制的类别名称 (如 'vortex chain')
    start_offset: 从该类别的第几个样本开始
    total_count: 总共绘制多少个样本
    per_fig: 每张画布画几个样本
    """
    # 1. 筛选出所有属于 target_class 的索引
    class_indices = [i for i, pred in enumerate(predictions) if pred == target_class]
    
    # 2. 检查是否有足够的样本
    if not class_indices:
        print(f"在预测结果中没有找到类别: {target_class}")
        return
    
    # 获取该类别的范本中心向量
    target_proto_center = final_proto_emb.get(target_class)
    
    # 3. 截取指定的范围 (n 到 n+100)
    selected_subset = class_indices[start_offset : start_offset + total_count]
    
    if not selected_subset:
        print(f"起始索引 {start_offset} 超出了类别 {target_class} 的样本总数 ({len(class_indices)})")
        return

    # 4. 分组绘图，每组 per_fig (10) 个
    for i in range(0, len(selected_subset), per_fig):
        batch_indices = selected_subset[i : i + per_fig]
        current_n = len(batch_indices)
        
        # 创建画布：2行，current_n 列
        fig, axes = plt.subplots(2, current_n, figsize=(5 * current_n, 8))
        fig.suptitle(f"Class: {target_class.upper()} | Samples {start_offset + i} to {start_offset + i + current_n}", 
                     fontsize=16, fontweight='bold', y=1.05)

        for j, idx in enumerate(batch_indices):
            # 处理 current_n 为 1 的情况
            ax_top = axes[0, j] if current_n > 1 else axes[0]
            ax_btm = axes[1, j] if current_n > 1 else axes[1]
            
            seq = test_data_raw[idx]
            file_name = test_files[idx]
            
            # --- 计算该样本到所属范本中心的欧氏距离 ---
            sample_emb = test_embs[idx]
            dist_to_proto = np.linalg.norm(sample_emb - target_proto_center)
            
            B = seq[:, 0]
            b_z = seq[:, 1]
            b_max = seq[:, 2]
            b_min = seq[:, 3]
            
            # 计算扰动最强位置
            b_sum_sq = b_z**2 + b_max**2 + b_min**2
            max_idx = np.argmax(b_sum_sq)
            min_B_idx = np.argmin(B)
            
            # --- 第一行：总磁场 B ---
            ax_top.plot(B, color='black', linewidth=1.5)
            ax_top.axhline(np.mean(B), color='gray', linestyle='--', alpha=0.6)
            ax_top.axvline(max_idx, color='red', linestyle='--', alpha=0.4)
            ax_top.axvline(min_B_idx, color='green', linestyle='--', alpha=0.4)
            
            global_idx = train_size + val_size + idx
            # 在标题中加入距离信息
            ax_top.set_title(f"File: {file_name}\nDist: {dist_to_proto:.4f} | Global Idx: {global_idx}", fontsize=11)
            
            if j == 0: ax_top.set_ylabel('B', fontweight='bold')
            
            # --- 第二行：扰动三分量 b ---
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

print("-------------- sheet -----------------")
# plot_class_samples(target_class='sheet', start_offset=0, total_count=100, per_fig=10)
print("-------------- shock -----------------")
plot_class_samples(target_class='shock', start_offset=0, total_count=100, per_fig=10)
print("-------------- soliton -----------------")
# plot_class_samples(target_class='soliton', start_offset=0, total_count=100, per_fig=10)
print("-------------- hole -----------------")
# plot_class_samples(target_class='hole', start_offset=0, total_count=100, per_fig=10)

# %% [markdown]
# ### 随机抽查测试集样本

# %%
def plot_text(l, n):
    # 确保 test_files 也是可访问的
    selected_indices = list(range(l, min(n+l, len(test_data_raw))))
    fig, ax = plt.subplots(2, n, figsize=(5*n, 8))
    
    for j, idx in enumerate(selected_indices):
        seq = test_data_raw[idx]
        file_name = test_files[idx]  # 获取对应的文件名
        
        B = seq[:, 0]
        b_z = seq[:, 1]
        b_max = seq[:, 2]
        b_min = seq[:, 3]
        
        b_sum_sq = b_z**2 + b_max**2 + b_min**2
        max_idx = np.argmax(b_sum_sq)
        
        # --- 第一行：总磁场 B ---
        ax[0, j].plot(B, color='black')
        ax[0, j].axhline(np.mean(B), color='gray', linestyle='--')
        ax[0, j].axvline(max_idx, color='red', linestyle='--', alpha=0.5)
        ax[0, j].set_ylabel('B')
        
        # 在标题里加入文件名和预测结果
        global_idx = train_size + val_size + idx
        # \n 是换行，避免标题太长挤在一起
        ax[0, j].set_title(f"File: {file_name}\nIdx: {global_idx} | Pred: {predictions[idx]}", 
                           fontsize=11, fontweight='bold')
        
        # --- 第二行：扰动三分量 b ---
        ax[1, j].plot(b_z, label='b_z', color='blue')
        ax[1, j].plot(b_max, label='b_max', color='red')
        ax[1, j].plot(b_min, label='b_min', color='green')
        ax[1, j].axvline(max_idx, color='red', linestyle='--', alpha=0.5)
        ax[1, j].axhline(0, color='gray', linestyle='--')
        
        if j == n - 1: # 只在最后一个子图显示图例，防止遮挡
            ax[1, j].legend(loc='upper right', fontsize='small')
            
        ax[1, j].set_xlabel('Time Step')
        ax[1, j].set_ylabel('b')

    plt.tight_layout()
    plt.show()

for i in range(2000, 2100, 10):
    plot_text(i, 10)
