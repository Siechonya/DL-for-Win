# Plan: `calibrate_thresholds` 修复

## 改动范围

两个文件（代码完全一致）：

- `src/train.py` — Bi-LSTM pipeline
- `src/train_cnn_transformer.py` — Transformer pipeline

每个文件需要改动同一处：`calibrate_thresholds` 函数。

**工作流**：编辑 `.py` → jupytext 同步 `.ipynb`（按 CLAUDE.md 规范）。

---

## Fix 1: 更新 docstring（两文件相同）

在 `def calibrate_thresholds(...)` 后，把现有的一行 docstring 替换为完整版：

```python
def calibrate_thresholds(model, data_preprocessed, prototypes_preprocessed_dict, device,
                         n_std=1.0, batch_size=256, robust=True,
                         margin_quantile=0.5, threshold_quantile=0.8,
                         min_calibration_samples=20):
    """用大数据集 (train+val) 计算原型中心和各类阈值。

    Args:
        n_std: 倍率，仅当 threshold_quantile=None 时生效
        robust: True=median+MAD×1.4826, False=mean+std；仅当 threshold_quantile=None 时生效
        margin_quantile: 高置信样本的 margin 分位数阈值 (margin = dist2nd - dist1st)；
                         保留 margin >= 此分位数的样本
        threshold_quantile: 非 None 时直接用 confident_dists 的此分位数作为阈值，
                            为 None 时回退到 robust (median+MAD) 或 classical (mean+std)
        min_calibration_samples: 高置信样本数不足此值时回退到全部分配样本

    Returns:
        proto_embs: dict[class_name → center_vector (64,)]
        class_thresholds: dict[class_name → float]
    """
```

---

## Fix 2: 修复死参数 + 表头误导（两文件相同）

### 2a — 当 `threshold_quantile` 生效时，改表头列名

找到打印表头的两行：

```python
print(f"{'Class':<15} | {'N':<6} | {'N conf':<6} | {'Margin':<10} | {'Median':<10} | {'Scale':<10} | {'Threshold':<10}")
print("-" * 90)
```

替换为根据 `threshold_quantile` 动态选择：

```python
if threshold_quantile is not None:
    q_label = f"Q{threshold_quantile*100:.0f}"
    stat_hdr = f"{q_label:<10} | {'Threshold':<10}"
else:
    stat_hdr = f"{'Median':<10} | {'Scale':<10} | {'Threshold':<10}"
print(f"{'Class':<15} | {'N':<6} | {'N conf':<6} | {'Margin':<10} | {stat_hdr}")
print("-" * 90)
```
> 注：`q_label` 如 `Q80` 表示 80th 分位数。

### 2b — 修改循环内的阈值计算分支

当前代码：
```python
            if robust:
                m = np.median(confident_dists)
                s = np.median(np.abs(confident_dists - m)) * 1.4826
            else:
                m = np.mean(confident_dists)
                s = np.std(confident_dists)
            if threshold_quantile is None:
                t = m + n_std * s
            else:
                t = np.quantile(confident_dists, threshold_quantile)
```

改为先判断 `threshold_quantile`，仅在回退分支中计算 m/s（避免无效计算）：

```python
            if threshold_quantile is not None:
                t = np.quantile(confident_dists, threshold_quantile)
                m = np.median(confident_dists)  # 仅供参考打印
                s = np.median(np.abs(confident_dists - m)) * 1.4826
            elif robust:
                m = np.median(confident_dists)
                s = np.median(np.abs(confident_dists - m)) * 1.4826
                t = m + n_std * s
            else:
                m = np.mean(confident_dists)
                s = np.std(confident_dists)
                t = m + n_std * s
```

### 2c — 修改打印行

替换：
```python
            print(
                f"{cls:<15} | {len(this_class_dists):<6} | {len(confident_dists):<6} | "
                f"{margin_cut:<10.4f} | {m:<10.4f} | {s:<10.4f} | {t:<10.4f}"
            )
```

为动态选择格式：
```python
            n_total = len(this_class_dists)
            n_conf = len(confident_dists)
            if threshold_quantile is not None:
                print(
                    f"{cls:<15} | {n_total:<6} | {n_conf:<6} | "
                    f"{margin_cut:<10.4f} | {t:<10.4f} | {t:<10.4f}"
                )
            else:
                print(
                    f"{cls:<15} | {n_total:<6} | {n_conf:<6} | "
                    f"{margin_cut:<10.4f} | {m:<10.4f} | {s:<10.4f} | {t:<10.4f}"
                )
```
> 注意：分位数模式下，`m`/`s` 仅供调试用，`Threshold` 列就是分位数值。如需保留 m/s 可见性，可扩展列数，但 plan 保持简洁——只输出最终阈值。

---

## Fix 3: 对齐顶部分隔线宽度

找到 `calibrate_thresholds` 中 `print("\n" + "-" * 55)`（约在 class_thresholds 循环前的那行），改为：

```python
print("\n" + "-" * 90)
```

---

## 验证

改完后对每个文件执行：

```powershell
# 1. 语法检查
python -c "import py_compile; py_compile.compile('src/train.py', doraise=True)"

# 2. jupytext 同步
& 'D:\Anaconda\envs\jupytext\python.exe' -m jupytext --sync src/train.py
& 'D:\Anaconda\envs\jupytext\python.exe' -m jupytext --sync src/train_cnn_transformer.py

# 3. JSON 合法性
python -c "import json; json.load(open('src/train.ipynb')); json.load(open('src/train_cnn_transformer.ipynb')); print('OK')"
```

---

## 不改的部分

- `classify_with_thresholds` — 无需改动，前向兼容
- `test_clustering` (legacy) — 保持不动
- `margin_quantile=0.5` + `min_calibration_samples=20` 导致的小类全额回退行为 — 属设计权衡，非 bug；如需调参另开 plan
