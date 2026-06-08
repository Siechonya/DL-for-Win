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
    
    # 定义物理门控掩码 (软门控: 中间压缩性结构[0.5~1]两组特征全激活)
    mask_alfven = comp_index < 1       # 阿尔芬结构门控
    mask_comp = comp_index > 0.5  # 压缩性结构门控

    def get_abs_skewness(x):
        mu = torch.mean(x, dim=1, keepdim=True)
        sigma = torch.std(x, dim=1, keepdim=True)
        sigma_safe = torch.clamp(sigma, min=0.05)  # 防噪声放大
        skew = torch.mean(((x - mu) / sigma_safe)**3, dim=1)
        return torch.abs(skew)

    # =========================================================================
    # A. 阿尔芬结构判据
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
    
    # (5) b_max 的自相关周期
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

    # （17）b_max梯度的偏度
    diff_bmax = bmax[:, 1:] - bmax[:, :-1]
    abs_skew_grad_bmax = get_abs_skewness(diff_bmax)
    abs_skew_grad_bmax = torch.where(mask_alfven, abs_skew_grad_bmax, torch.zeros_like(abs_skew_grad_bmax))


    # =========================================================================
    # B. 压缩性结构判据
    # =========================================================================

    # (3) b_z和B 扰动凹陷或凸起程度
    idx_max_bz = torch.argmax(torch.abs(bz), dim=1)
    bz_dip = bz[batch_indices, idx_max_bz]
    idx_max_B = torch.argmax(torch.abs(B), dim=1)
    B_dip = B[batch_indices, idx_max_B]
    B_dip = torch.where(mask_comp, B_dip, torch.zeros_like(B_dip))

    # (6) 激波指标: b_z 斜率(差分)绝对值的最大值
    max_grad_bz = torch.max(torch.abs(bz[:, 1:] - bz[:, :-1]), dim=1)[0]

    # (7) 激波判据：b_z最大值的绝对值减最小值的绝对值
    b_z_max_ = torch.max(bz, dim=1)[0]
    b_z_min_ = torch.min(bz, dim=1)[0]
    R_jump = torch.abs(b_z_max_) - torch.abs(b_z_min_) # 接近0：shock；接近1：soliton；接近-1：hole
    R_jump = torch.where(mask_comp, R_jump, torch.zeros_like(R_jump))

    def get_abs_kurtosis(x):
        mu = torch.mean(x, dim=1, keepdim=True)
        sigma = torch.std(x, dim=1, keepdim=True)
        sigma_safe = torch.clamp(sigma, min=0.05)  # 防止噪声放大
        kurt = torch.mean( ((x - mu) / sigma_safe)**4, dim=1) / 10 # shock 的峰度太大，除以10进行压缩
        return kurt
    
    # (10) dot_B 的全局峰度 (Kurtosis)
    kurt_dot_B = get_abs_kurtosis(dot_B)
    kurt_dot_B = torch.where(mask_comp, kurt_dot_B, torch.zeros_like(kurt_dot_B))

    # (11) dot_bz 的全局峰度 (Kurtosis)
    kurt_dot_bz = get_abs_kurtosis(dot_bz)
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
    def get_max_corr_template(x, y_template, max_shift=50):
        """
        计算 batch x 与单个模板 y_template 之间的最大互相关性 (位移无关, 限制在 ±max_shift)
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
        # 限制搜索范围为 ±max_shift，避免无关结构通过极端位移获得虚假高相关
        pos_indices = torch.arange(0, max_shift + 1, device=x.device)
        neg_indices = torch.arange(pad_size - max_shift, pad_size, device=x.device)
        valid_indices = torch.cat([pos_indices, neg_indices])
        # 使用 abs 是为了同时兼容正向和反向的波形 (+/- 符号)
        max_corr = torch.max(torch.abs(cross_corr[:, valid_indices]), dim=1)[0]
        return max_corr / (x_energy * y_energy + 1e-8)
    t1 = torch.linspace(-100, 100, 300, device=device)
    tanh_template = torch.tanh(t1).unsqueeze(0) # [1, 300]
    corr_shock_B = get_max_corr_template(B, tanh_template, max_shift=50)
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
        pol_ratio,
        comp_index,
        bz_dip,
        B_dip,
        corr_bmax_bmin,
        dom_freq,
        max_grad_bz,
        R_jump,
        peakiness_dot_bmax,
        b_max_flipscore,
        kurt_dot_B,
        kurt_dot_bz,
        complexity_index_bz,
        complexity_index_bmax,
        corr_shock_B,
        abs_skew_grad_B,
        abs_skew_grad_bz,
        abs_skew_grad_bmax
    ], dim=1)