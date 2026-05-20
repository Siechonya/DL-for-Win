import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # 自动切换为 notebook 模式
import torch.nn.functional as F
import copy
import itertools # 用于循环迭代范本

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
        
        B_mean = np.mean(B)
        B_norm = (B - B_mean) / (B_mean + 1e-9)  * 2  # 放大 B 的数值范围，增强其在训练中的影响力
        
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



import numpy as np

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
    
class PrototypeDataset(Dataset):
    def __init__(self, prototypes_dict):
        self.data = []
        self.labels = []
        # 将分类名称映射为整数 ID
        self.label_map = {cls_name: i for i, cls_name in enumerate(prototypes_dict.keys())}
        for cls_name, seqs in prototypes_dict.items():
            for seq in seqs:
                self.data.append(seq)
                self.labels.append(self.label_map[cls_name])
                
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
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
    
    # (2) 极化比: max(|b_min|) / max(|b_max|)
    max_abs_bmin = torch.max(torch.abs(bmin), dim=1)[0]
    max_abs_bmax = torch.max(torch.abs(bmax), dim=1)[0]
    raw_pol_ratio = max_abs_bmin / (max_abs_bmax + 1e-6)
    pol_ratio = torch.where(mask_alfven, raw_pol_ratio, torch.zeros_like(raw_pol_ratio))
    
    # # (4) b_max 和 b_min 的最大互相关性
    # def get_max_corr_pair(x, y):
    #     N_pts = x.size(1)
    #     x_norm = (x - x.mean(dim=1, keepdim=True)) 
    #     y_norm = (y - y.mean(dim=1, keepdim=True))
    #     pad_size = N_pts * 2
    #     X_freq = torch.fft.rfft(x_norm, n=pad_size, dim=1)
    #     Y_freq = torch.fft.rfft(y_norm, n=pad_size, dim=1)
    #     corr_freq = X_freq * torch.conj(Y_freq)
    #     cross_corr = torch.fft.irfft(corr_freq, n=pad_size, dim=1)
    #     x_energy = torch.sqrt(torch.sum(x_norm**2, dim=1) + 1e-8)
    #     y_energy = torch.sqrt(torch.sum(y_norm**2, dim=1) + 1e-8)
    #     max_corr = torch.max(torch.abs(cross_corr), dim=1)[0]
    #     return max_corr / (x_energy * y_energy + 1e-8)
    # raw_corr_bmax_bmin = get_max_corr_pair(bmax, bmin)
    # # 增加双重条件：极化比>0.2 且 属于阿尔芬结构门控
    # condition_corr = (pol_ratio > 0.2) & mask_alfven
    # corr_bmax_bmin = torch.where(condition_corr, raw_corr_bmax_bmin, torch.zeros_like(raw_corr_bmax_bmin))
    
    # # (5) b_max 的自相关相位
    # import torch.nn.functional as F
    # def get_generalized_freq(x, device):
    #     N = x.size(1)
    #     # 1. 去均值标准化
    #     x_norm = x - x.mean(dim=1, keepdim=True)
    #     # 2. 计算自相关
    #     pad_size = N * 2
    #     X_freq = torch.fft.rfft(x_norm, n=pad_size, dim=1)
    #     autocorr = torch.fft.irfft(X_freq * torch.conj(X_freq), n=pad_size, dim=1)[:, :N]
    #     autocorr = autocorr / (autocorr[:, 0:1] + 1e-8)
    #     # 3. 寻找所有“极大值点” (Local Maxima)
    #     # 通过比较相邻点找到所有波峰
    #     is_max = (autocorr[:, 1:-1] > autocorr[:, :-2]) & (autocorr[:, 1:-1] > autocorr[:, 2:])
    #     # 4. 阈值筛选 + 第一个显著峰
    #     # 找“第一个”相关性超过 0.4 且具有显著性的峰
    #     # 如果没有任何峰超过阈值，它就是单周期 (freq=1)
    #     freq_est = torch.ones(x.size(0), device=device) # 默认频率为 1
    #     for i in range(x.size(0)):
    #         # 找到该样本所有的极大值索引
    #         peak_indices = torch.where(is_max[i])[0] + 1 
    #         # 过滤掉靠近原点（Lag < 20）的干扰点
    #         valid_peaks = peak_indices[peak_indices > 20]
    #         if len(valid_peaks) > 0:
    #             # 在这些峰里，找到相关系数最高的那个
    #             best_peak_idx = valid_peaks[torch.argmax(autocorr[i, valid_peaks])]
    #             best_corr_val = autocorr[i, best_peak_idx]
    #             # 如果这个最强峰的相关性足够高（ > 0.4），则承认它
    #             if best_corr_val > 0.4:
    #                 freq_est[i] = N / best_peak_idx.float()
    #     return freq_est
    # dom_freq = get_generalized_freq(bmax, device)
    # dom_freq = torch.where(mask_alfven, dom_freq, torch.zeros_like(dom_freq))

    
    # # (9) B最小时，dot_bmax 凸起的程度和 b_max 的大小
    # def calc_sheet_reversal_criterion(B_full, b_max, search_range=100):
    #     """
    #     B_full: [Batch, Length] - 总磁场强度 B
    #     b_max:  [Batch, Length] - 主翻转分量
    #     search_range: 向两侧搜索极值的最大距离
    #     """
    #     batch_size, seq_len = B_full.shape
    #     device = B_full.device
    #     # 1. 找到 B 全局最小点索引
    #     idx_min_B = torch.argmin(B_full, dim=1)
    #     # 2. 检查中心窗口 (+/- 10) 是否发生变号
    #     range_tensor = torch.arange(seq_len, device=device).unsqueeze(0)
    #     center_mask = (range_tensor >= (idx_min_B.unsqueeze(1) - 10)) & \
    #                   (range_tensor <= (idx_min_B.unsqueeze(1) + 10))
    #     signs = torch.sign(b_max)
    #     win_signs = torch.where(center_mask, signs, signs[torch.arange(batch_size), idx_min_B].unsqueeze(1))
    #     has_flip = (torch.max(win_signs, dim=1).values != torch.min(win_signs, dim=1).values)
    #     left_extrema_val = torch.zeros(batch_size, device=device)
    #     right_extrema_val = torch.zeros(batch_size, device=device)
    #     # 3. 寻找向外延伸的第一个局部极值
    #     for i in range(batch_size):
    #         if not has_flip[i]: continue
    #         center = idx_min_B[i]
    #         # --- 右侧搜索：从 center 向右 ---
    #         r_end = min(seq_len, center + search_range + 1)
    #         r_segment = b_max[i, center : r_end]
    #         if r_segment.numel() > 1:
    #             # 计算相邻点的差值 (类似导数)
    #             diffs_r = torch.diff(r_segment)
    #             signs_r = torch.sign(diffs_r)
    #             # 寻找导数符号发生变化的位置 (即出现极值或平缓区)
    #             changes_r = torch.nonzero(signs_r[:-1] != signs_r[1:]).squeeze(1)
    #             if len(changes_r) > 0:
    #                 # +1 是因为 diff 会使得索引偏移，取变号后的那个点作为极值点
    #                 right_extrema_val[i] = r_segment[changes_r[0] + 1]
    #             else:
    #                 # 如果在 search_range 内单调递增/递减没有极值，则取边界点
    #                 right_extrema_val[i] = r_segment[-1]
    #         elif r_segment.numel() == 1:
    #             right_extrema_val[i] = r_segment[0]
    #         # --- 左侧搜索：从 center 向左 ---
    #         l_start = max(0, center - search_range)
    #         l_segment = b_max[i, l_start : center + 1]
    #         if l_segment.numel() > 1:
    #             # 关键：将左侧切片反转，使其物理意义变为“从 center 向左延伸”
    #             rev_l_segment = torch.flip(l_segment, dims=[0])
    #             diffs_l = torch.diff(rev_l_segment)
    #             signs_l = torch.sign(diffs_l)
    #             changes_l = torch.nonzero(signs_l[:-1] != signs_l[1:]).squeeze(1)
                
    #             if len(changes_l) > 0:
    #                 left_extrema_val[i] = rev_l_segment[changes_l[0] + 1]
    #             else:
    #                 left_extrema_val[i] = rev_l_segment[-1]
    #         elif l_segment.numel() == 1:
    #             left_extrema_val[i] = l_segment[0]
    #     # 最终判定：变号且两翼（第一个局部）极值绝对值均 > 0.5
    #     is_current_sheet = has_flip & (torch.abs(left_extrema_val) > 0.5) & (torch.abs(right_extrema_val) > 0.5)
    #     # 返回得分：如果是电流片，得分 = 1.0 - |b_max在B最小时的值|
    #     val_at_min = b_max[batch_indices, idx_min_B]
    #     score = 1.0 - torch.abs(val_at_min)
    #     score = torch.where(is_current_sheet, score, torch.zeros_like(score))
    #     return score
    # idx_min_B = torch.argmin(B, dim=1)
    # peakiness_dot_bmax = dot_bmax[batch_indices, idx_min_B]
    # peakiness_dot_bmax = torch.where(mask_alfven, peakiness_dot_bmax, torch.zeros_like(peakiness_dot_bmax))
    # b_max_flipscore = calc_sheet_reversal_criterion(B, bmax, search_range=100)
    # b_max_flipscore = torch.where(mask_alfven, b_max_flipscore, torch.zeros_like(b_max_flipscore))

    # # （16）b_max梯度的偏度
    # diff_bmax = bmax[:, 1:] - bmax[:, :-1]
    # abs_skew_grad_bmax = get_abs_skewness(diff_bmax)
    # abs_skew_grad_bmax = torch.where(mask_alfven, abs_skew_grad_bmax, torch.zeros_like(abs_skew_grad_bmax))


    # # =========================================================================
    # # B. 压缩性结构专属判据 (移除 mask_comp 硬截断，保留物理连续性)
    # # =========================================================================

    # # (3) b_z 扰动凹陷或凸起程度
    # idx_max_bz = torch.argmax(torch.abs(bz), dim=1)
    # bz_dip = bz[batch_indices, idx_max_bz]

    # # (6) 激波指标: b_z 斜率(差分)绝对值的最大值
    # max_grad_bz = torch.max(torch.abs(bz[:, 1:] - bz[:, :-1]), dim=1)[0]

    # # (8) 激波判据：b_z最大值的绝对值减最小值的绝对值
    # b_z_max_ = torch.max(bz, dim=1)[0]
    # b_z_min_ = torch.min(bz, dim=1)[0]
    # R_jump = 1 - (torch.abs(b_z_max_) - torch.abs(b_z_min_)) # 越接近1说明越跳变
    # R_jump = torch.where(mask_comp, R_jump, torch.zeros_like(R_jump))

    # # (10) dot_B 的全局峰度 (Kurtosis)
    # mean_dot_B = torch.mean(dot_B, dim=1, keepdim=True)
    # std_dot_B = torch.std(dot_B, dim=1, keepdim=True)
    # kurt_dot_B = torch.mean(((dot_B - mean_dot_B) / (std_dot_B + 1e-6))**4, dim=1) / 10.0
    # kurt_dot_B = torch.where(mask_comp, kurt_dot_B, torch.zeros_like(kurt_dot_B))

    # # (11) 在 dot_bz 最大值附近 (±10个点) 的积分 (局部能量聚集度)
    # idx_max_dot_bz = torch.argmax(dot_bz, dim=1)
    # offsets = torch.arange(-10, 11, device=device).view(1, -1) # [1, 21]
    # window_idx_dot_bz = torch.clamp(idx_max_dot_bz.view(-1, 1) + offsets, 0, 299)
    # int_dot_bz_window = torch.gather(dot_bz, 1, window_idx_dot_bz).sum(dim=1)
    # int_dot_bz_window = torch.where(mask_comp, int_dot_bz_window, torch.zeros_like(int_dot_bz_window))

    # # (12) dot_bz 的全局峰度 (Kurtosis)
    # mean_dot_bz = torch.mean(dot_bz, dim=1, keepdim=True)
    # std_dot_bz = torch.std(dot_bz, dim=1, keepdim=True)
    # kurt_dot_bz = torch.mean(((dot_bz - mean_dot_bz) / (std_dot_bz + 1e-6))**4, dim=1) / 10.0
    # kurt_dot_bz = torch.where(mask_comp, kurt_dot_bz, torch.zeros_like(kurt_dot_bz))

    # # (13) b_z和b_max穿过 ±0.5 的次数 (反映震荡结构的复杂程度)
    # def calc_criterion_16(bz, threshold=0.5):
    #     """
    #     bz: [Batch, Length] 的张量
    #     计算 bz 穿过 threshold 和 -threshold 的总次数并除以 4
    #     """
    #     # 1. 计算穿过 +0.5 的次数
    #     # 当相邻两个点的符号不同时，代表发生了一次穿越
    #     cross_plus = torch.diff((bz > threshold).int(), dim=1).abs().sum(dim=1)
    #     # 2. 计算穿过 -0.5 的次数
    #     cross_minus = torch.diff((bz < -threshold).int(), dim=1).abs().sum(dim=1)
    #     # 3. 求和并归一化
    #     # 对于一个标准的双极性脉冲 (0 -> 1 -> -1 -> 0):
    #     # 穿过 0.5 两次 (上一、下一)，穿过 -0.5 两次 (下一、上一)，总计 4 次。4/4 = 1.0
    #     complexity_index = (cross_plus + cross_minus).float() / 4.0
    #     score = torch.exp(-(complexity_index-1)**2 / 0.3 **2) # 距离标准值1越远，得分越低
    #     return score
    # complexity_index_bz = calc_criterion_16(bz)
    # complexity_index_bmax = calc_criterion_16(bmax)

    # # (14) B 场与 tanh 模板的最大相关性
    # def get_max_corr_template(x, y_template):
    #     """
    #     计算 batch x 与单个模板 y_template 之间的最大互相关性 (位移无关)
    #     """
    #     N_pts = x.size(1)
    #     # 模板扩展到 batch 大小
    #     y = y_template.expand(x.size(0), -1)
    #     x_norm = (x - x.mean(dim=1, keepdim=True)) 
    #     y_norm = (y - y.mean(dim=1, keepdim=True))
    #     pad_size = N_pts * 2
    #     X_freq = torch.fft.rfft(x_norm, n=pad_size, dim=1)
    #     Y_freq = torch.fft.rfft(y_norm, n=pad_size, dim=1)
    #     corr_freq = X_freq * torch.conj(Y_freq)
    #     cross_corr = torch.fft.irfft(corr_freq, n=pad_size, dim=1)
    #     x_energy = torch.sqrt(torch.sum(x_norm**2, dim=1) + 1e-8)
    #     y_energy = torch.sqrt(torch.sum(y_norm**2, dim=1) + 1e-8)
    #     # 使用 abs 是为了同时兼容正向和反向的波形 (+/- 符号)
    #     max_corr = torch.max(torch.abs(cross_corr), dim=1)[0]
    #     return max_corr / (x_energy * y_energy + 1e-8)
    # t1 = torch.linspace(-100, 100, 300, device=device)
    # tanh_template = torch.tanh(t1).unsqueeze(0) # [1, 300]
    # corr_shock_B = get_max_corr_template(B, tanh_template)
    # corr_shock_B = torch.where(mask_comp, corr_shock_B, torch.zeros_like(corr_shock_B))

    # # (15) 梯度（斜率）的偏度 (反映跳变的方向性)
    # # 激波的斜率分布是单向极值（极度偏斜），震荡结构的斜率分布是对称的（偏度近0）
    # diff_B = B[:, 1:] - B[:, :-1]
    # diff_bz = bz[:, 1:] - bz[:, :-1]
    # abs_skew_grad_B = get_abs_skewness(diff_B)
    # abs_skew_grad_bz = get_abs_skewness(diff_bz)
    # # 采用压缩性门控
    # abs_skew_grad_B = torch.where(mask_comp, abs_skew_grad_B, torch.zeros_like(abs_skew_grad_B))
    # abs_skew_grad_bz = torch.where(mask_comp, abs_skew_grad_bz, torch.zeros_like(abs_skew_grad_bz))

    return torch.stack([
        pol_ratio,
        comp_index,
        # bz_dip,
        # corr_bmax_bmin,
        # dom_freq,
        # max_grad_bz,
        # R_jump,
        # peakiness_dot_bmax,
        # b_max_flipscore,
        # kurt_dot_B,
        # int_dot_bz_window,
        # kurt_dot_bz,
        # complexity_index_bz,
        # complexity_index_bmax,
        # corr_shock_B,
        # abs_skew_grad_B,
        # abs_skew_grad_bz,
        # abs_skew_grad_bmax
    ], dim=1)

def physical_contrastive_loss(embeddings, labels, phys_features, margin=2.0):
    N = embeddings.size(0)
    device = embeddings.device

    # phys_features 是从 DataLoader 直接传进来的 [N, feature_dim]
    # 只需做标准化和 cdist
    feat_mean = phys_features.mean(dim=0, keepdim=True)
    feat_std = phys_features.std(dim=0, keepdim=True) + 1e-6
    phys_features_norm = (phys_features - feat_mean) / feat_std

    # 2. 计算物理置信度权重
    # 计算每个样本物理向量的范数，代表它偏离“平均噪声背景”(原点)的程度
    # 混沌结构的判据得分通常都很低且接近均值，其范数会较小
    # phys_saliency = torch.norm(phys_features_norm, dim=1) 
    # 使用 sigmoid 或 tanh 将其映射到 [0, 1]，作为样本的“结构置信度”
    # confidence = torch.tanh(phys_saliency)
    # 构造成对权重矩阵：只有当两个样本都是“高质量结构”时，权重才高：Weight_ij = conf_i * conf_j
    # conf_weight_matrix = confidence.unsqueeze(1) * confidence.unsqueeze(0)

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
    phys_similar = (phys_diff_matrix < 0.5) 
    phys_dissimilar = (phys_diff_matrix > 3.0)

    mask = torch.eye(N, device=device)

    # 正样本对损失
    # 1. 标签明确相同 2. 或者虽然没标签但物理特征极度接近
    pos_mask = (same_label | (~has_label & phys_similar)) * (1 - mask)
    if pos_mask.sum() > 0:
        # weighted_pos_dist = pos_mask * conf_weight_matrix * (dist_matrix**2 + 0.2 * phys_diff_matrix)
        # pos_loss = weighted_pos_dist.sum() / (pos_mask * conf_weight_matrix).sum()
        pos_loss = (pos_mask * (dist_matrix**2 + 0.2 * phys_diff_matrix)).sum() / pos_mask.sum()
    else:
        pos_loss = torch.tensor(0.0, device=device)

    # 负样本对损失
    # 1. 标签明确不同 2. 或者虽然没标签但物理特征差异极大
    neg_mask = (diff_label | (~has_label & phys_dissimilar)) * (1 - mask)
    if neg_mask.sum() > 0:
        # 同样对负样本应用权重，避免混沌样本被错误地推向远方
        dynamic_margin = margin + 0.5 * phys_diff_matrix # 物理差异越大，排斥越狠
        # weighted_neg_dist = neg_mask * conf_weight_matrix * torch.clamp(dynamic_margin - dist_matrix, min=0)**2
        # neg_loss = weighted_neg_dist.sum() / (neg_mask * conf_weight_matrix).sum()
        neg_loss = (neg_mask * torch.clamp(dynamic_margin - dist_matrix, min=0)**2).sum() / neg_mask.sum()
    else:
        neg_loss = torch.tensor(0.0, device=device)

    return pos_loss + neg_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import copy
import numpy as np

def calc_invariant_mse(pred, target, max_shift=20):
    """
    计算具有平移和翻转不变性的 MSE Loss。
    pred, target shape: [Batch, Length, Channels]
    max_shift: 允许的最大左右平移步数 (过大容易让模型“走捷径”乱匹配)
    """
    B, L, C = pred.shape
    # 1. 生成时间轴左右翻转的 Target
    target_flipped = torch.flip(target, dims=[1]) 

    # 记录每个样本、每个通道目前的最小 Loss，初始化为无穷大
    best_loss = torch.full((B, C), float('inf'), device=pred.device)
    
    # 为了使用 F.pad，我们需要把通道维度放到前面 -> [B, C, Length]
    pred_t = pred.transpose(1, 2)

    # 2. 遍历所有允许的平移步数
    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            p_shifted_t = pred_t
        elif shift > 0:
            # 向右平移：切掉右边溢出的 shift 个点，左边空缺用边缘值(replicate)复制填充
            p_shifted_t = F.pad(pred_t[:, :, :-shift], (shift, 0), mode='replicate')
        else:
            # 向左平移：切掉左边溢出的 -shift 个点，右边空缺用边缘值复制填充
            s = -shift
            p_shifted_t = F.pad(pred_t[:, :, s:], (0, s), mode='replicate')
            
        # 转回 [B, Length, C]
        p_shifted = p_shifted_t.transpose(1, 2)

        # 3. 分别计算与 正常Target 和 翻转Target 的逐点 MSE (在 Length 维度求平均)
        mse_normal = F.mse_loss(p_shifted, target, reduction='none').mean(dim=1)
        mse_flipped = F.mse_loss(p_shifted, target_flipped, reduction='none').mean(dim=1)

        # 4. 找出当前 shift 下，正向和翻转中较小的误差
        current_min = torch.min(mse_normal, mse_flipped)
        
        # 5. 更新全局最小误差
        best_loss = torch.min(best_loss, current_min)

    # 返回 Batch 内各样本平均后的通道 Loss -> shape: [C]
    return best_loss.mean(dim=0)


def train_autoencoder(model, train_dataloader, val_dataloader, proto_dataloader, device, 
                      epochs=100, lr=0.001, patience=10, best_model_path=None, 
                      max_lambda_contrastive=0.1, step_lambda_contrastive=0.01, start_lambda_contrastive=0.0, max_shift=20):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, min_lr=1e-4)
    
    # 损失函数权重
    loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_list = []
    val_loss_list = []
    all_train_loss_list = [[] for _ in range(5)]
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
        batch_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1} Train', leave=False)
        
        for batch_item in batch_bar:
            x, x_phys = batch_item
            x, x_phys = x.to(device), x_phys.to(device)
            
            p_data, p_labels, p_phys = next(proto_iter)
            p_data, p_labels, p_phys = p_data.to(device), p_labels.to(device), p_phys.to(device)
            
            optimizer.zero_grad()
            
            # --- A. 重建路径 ---
            output = model(x)
            
            # 使用平移翻转不变性 MSE Loss
            # 这会返回 shape 为 [4] 的 tensor，对应 4 个 channel 的 loss
            mse_per_channel = calc_invariant_mse(output, x, max_shift=max_shift)
            
            # 应用通道权重 (除以4是为了取平均，保持量级一致)
            weighted_errs = (mse_per_channel * loss_weights) / 4.0
            loss_rec = (1 - current_lambda) * weighted_errs.sum()
            
            # --- B. 对比路径 ---
            combined_data = torch.cat([x, p_data], dim=0)
            combined_emb = model.encode(combined_data)
            combined_phys = torch.cat([x_phys, p_phys], dim=0)
            
            x_labels = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            combined_labels = torch.cat([x_labels, p_labels], dim=0)
            
            loss_con = physical_contrastive_loss(combined_emb, combined_labels, combined_phys)

            weighted_con = current_lambda * loss_con
            # loss = loss_rec + weighted_con    
            loss = loss_rec # --------------------------------------------------------------------------------------------        
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
                
                # 重建损失 (同样应用平移翻转不变性 MSE)
                v_output = model(vx)
                v_mse_per_channel = calc_invariant_mse(v_output, vx, max_shift=max_shift)
                
                v_weighted_errs = (v_mse_per_channel * loss_weights) / 4.0
                total_val_rec_loss += v_weighted_errs.sum().item()
                val_errs_detailed += v_weighted_errs.cpu().numpy()
            
                # 对比损失
                vp_data, vp_labels, vp_phys = next(val_proto_iter)
                vp_data, vp_labels, vp_phys = vp_data.to(device), vp_labels.to(device), vp_phys.to(device)
                
                v_comb_data = torch.cat([vx, vp_data], dim=0)
                v_comb_emb = model.encode(v_comb_data)
                v_comb_phys = torch.cat([vx_phys, vp_phys], dim=0)
                
                vx_labels = torch.full((vx.size(0),), -1, dtype=torch.long, device=device)
                v_comb_labels = torch.cat([vx_labels, vp_labels], dim=0)
                
                v_loss_con = physical_contrastive_loss(v_comb_emb, v_comb_labels, v_comb_phys)
                total_val_con_loss += v_loss_con.item()

        avg_val_rec = (1-current_lambda) * total_val_rec_loss / len(val_dataloader)
        avg_val_con = (total_val_con_loss / len(val_dataloader)) * current_lambda
        # avg_val_loss = avg_val_rec + avg_val_con
        avg_val_loss = avg_val_rec # --------------------------------------------------------------------------------------------
        v_errs = val_errs_detailed / len(val_dataloader)
        val_loss_list.append(avg_val_loss)
        
        summary = (
            f"TRA: {avg_train_loss:.4f} = {t_errs[0]:.4f}+{t_errs[1]:.4f}+{t_errs[2]:.4f}+{t_errs[3]:.4f} + {t_con:.4f} || "
            f"VAL: {avg_val_loss:.4f} = {v_errs[0]:.4f}+{v_errs[1]:.4f}+{v_errs[2]:.4f}+{v_errs[3]:.4f} + {avg_val_con:.4f}"
        )
        epoch_bar.set_postfix_str(summary)

        for i in range(4):
            all_train_loss_list[i].append(t_errs[i])
            all_val_loss_list[i].append(v_errs[i])
        all_train_loss_list[4].append(t_con)
        all_val_loss_list[4].append(avg_val_con)

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


def test_clustering(model, test_data_preprocessed, prototypes_preprocessed_dict, device, fallback_threshold=0.5):
    """
    测试数据和范本数据都已经在外部预处理好了，直接转 tensor 推理。
    """
    model.eval()
    proto_emb = {}
    class_thresholds = {}
    
    # 1. 计算范本的嵌入向量及每一类的自适应阈值
    for cls, data_matrix in prototypes_preprocessed_dict.items():
        emb_list = []
        for seq_padded in data_matrix:
            # 增加 batch 维度并送到 GPU
            seq_tensor = torch.tensor(seq_padded, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 计算完毕后拉回 CPU 供 numpy 计算
                emb = model.encode(seq_tensor).cpu().numpy().flatten()
            
            norm = np.linalg.norm(emb)
            emb_list.append(emb / norm if norm > 0 else emb)
            
        if emb_list:
            mean_emb = np.mean(emb_list, axis=0)
            center_norm = np.linalg.norm(mean_emb)
            center_vec = mean_emb / center_norm if center_norm > 0 else mean_emb
            proto_emb[cls] = center_vec
            
            internal_sims = [np.dot(e, center_vec) for e in emb_list]
            if len(internal_sims) > 1:
                adaptive_t = np.mean(internal_sims) - np.std(internal_sims)
                class_thresholds[cls] = np.clip(adaptive_t, 0.3, 0.85)
            else:
                class_thresholds[cls] = fallback_threshold
            
            print(f"Class [{cls}]: Adaptive Threshold set to {class_thresholds[cls]:.3f} (based on {len(data_matrix)} samples)")
        else:
            proto_emb[cls] = np.zeros(256) 
            class_thresholds[cls] = fallback_threshold

    predictions = []
    
    # 2. 对测试数据进行分类
    for seq_padded in test_data_preprocessed:
        seq_tensor = torch.tensor(seq_padded, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = model.encode(seq_tensor).cpu().numpy().flatten()
            
        norm = np.linalg.norm(emb)
        emb_normalized = emb / norm if norm > 0 else emb
        
        max_sim = -float('inf')
        best_match = None
        for cls, p_emb in proto_emb.items():
            sim = np.dot(emb_normalized, p_emb)
            if sim > max_sim:
                max_sim = sim
                best_match = cls
        
        if best_match and max_sim >= class_thresholds[best_match]:
            predictions.append(best_match)
        else:
            predictions.append('neither')
        
    return predictions, proto_emb
if __name__ == "__main__":
    workspace = '..\\'
    trainset_path = os.path.join(workspace, 'trainset_20240401-0414')
    samples_path = os.path.join(workspace, 'samples_clean')

    # --- 1. 加载数据 ---
    # 这里返回的三个列表是原始顺序
    data_all_processed, data_all_raw, data_all_files = load_data(trainset_path)
    
    # --- 同步随机打乱 ---
    # 将处理后的数据、原始波形数据、文件名打包在一起
    combined = list(zip(data_all_processed, data_all_raw, data_all_files))
    
    # 使用固定的随机数种子（Seed），保证以后想复现实验时，测试集里的样本是不变的
    import random
    random.seed(42) 
    random.shuffle(combined)
    
    # 解包回三个对齐的列表
    data_all_processed, data_all_raw, data_all_files = zip(*combined)
    
    # 转换为 list 方便后续切片操作
    data_all_processed = list(data_all_processed)
    data_all_raw = list(data_all_raw)
    data_all_files = list(data_all_files)
    
    print(f"Total dataset has {len(data_all_files)} files. Data has been shuffled.")

    # --- 2. 执行 8:1:1 划分 ---
    total_samples = len(data_all_processed)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    
    # 此时切出来的 processed 和 raw 索引是严格一一对应的
    train_data_processed = data_all_processed[:train_size]
    val_data_processed = data_all_processed[train_size:train_size+val_size]
    test_data_processed = data_all_processed[train_size+val_size:]
    
    # 这里的 test_data_raw 就是绘图要用的原始波形
    test_data_raw = data_all_raw[train_size+val_size:]
    test_files = data_all_files[train_size+val_size:] # 记下文件名，方便查错

    # --- 3. 加载并增强范本 (Prototypes) ---
    # (这部分逻辑保持不变)
    classes = ['sheet', 'vortex chain', 'c vortex', 'l vortex', 'hole', 'soliton', 'shock']
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
    
    # 注意：为了后续 verify_top_k_samples 的索引对齐，
    # 在提取 test_embeddings 时必须保证 DataLoader 的顺序不乱
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True) 
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True) 
    proto_dataloader = DataLoader(proto_dataset, batch_size=64, shuffle=True)

    # --- 6. 训练 ---
    model = BiAutoencoder(input_size=4, cnn_channels=16, hidden_size=128, num_layers=2, latent_dim=64).to(device)
    print(f"Starting Autoencoder training on {device}...")
    torch.cuda.empty_cache()
    train_loss_list, val_loss_list, all_train_loss_list, all_val_loss_list = train_autoencoder(
        model, train_dataloader, val_dataloader, proto_dataloader, device,
        epochs=100, lr=0.005, patience=5, max_lambda_contrastive=0.05, step_lambda_contrastive=0.01, start_lambda_contrastive=0.05, max_shift=50
    )