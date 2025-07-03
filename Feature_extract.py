import numpy as np
from sklearn.decomposition import PCA, NMF
import pywt
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, pearsonr

# ======================
# 特征提取函数
# ======================

def pca_feature_extraction(
    data: np.ndarray,
    n_components: int = 5
) -> np.ndarray:
    """
    对输入数据 (n_samples, n_pixels, n_bands) 中的所有光谱做 PCA 降维。
    返回形状 (n_samples, n_pixels, n_components)。
    """
    n_samples, n_pixels, n_bands = data.shape
    flat_data = data.reshape((-1, n_bands))

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(flat_data)

    return transformed.reshape((n_samples, n_pixels, n_components))


def nmf_feature_extraction(
    data: np.ndarray,
    n_components: int = 5,
    init: str = 'nndsvda',
    max_iter: int = 200
) -> np.ndarray:
    """
    对输入数据做 NMF 降维，返回形状 (n_samples, n_pixels, n_components)。
    """
    n_samples, n_pixels, n_bands = data.shape
    flat_data = data.reshape((-1, n_bands))

    model = NMF(n_components=n_components, init=init, max_iter=max_iter, random_state=0)
    W = model.fit_transform(flat_data)

    return W.reshape((n_samples, n_pixels, n_components))


def cwt_feature_extraction(
    data: np.ndarray,
    wavelet: str = 'mexh',
    scales: np.ndarray = np.arange(1, 6)
) -> np.ndarray:
    """
    对输入数据做连续小波变换 (CWT)，提取每个尺度最大绝对系数。
    返回形状 (n_samples, n_pixels, n_scales)。
    """
    n_samples, n_pixels, n_bands = data.shape
    n_scales = len(scales)
    features = np.zeros((n_samples, n_pixels, n_scales), dtype=float)

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            coeffs, _ = pywt.cwt(spectrum, scales, wavelet)
            features[i, j, :] = np.max(np.abs(coeffs), axis=1)

    return features


def spectral_derivative(
    data: np.ndarray,
    order: int = 1
) -> np.ndarray:
    """
    对输入数据计算一阶或二阶导数，返回形状 (n_samples, n_pixels, n_bands)。
    """
    if order not in [1, 2]:
        raise ValueError("order 只能为 1 或 2。")

    n_samples, n_pixels, n_bands = data.shape
    deriv = np.zeros_like(data)

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            d = np.diff(spectrum, n=order)
            pad_len = n_bands - d.shape[0]
            deriv[i, j, :] = np.concatenate([d, np.zeros(pad_len)])

    return deriv


def peak_feature_extraction(
    data: np.ndarray,
    n_peaks: int = 3,
    height: float = None,
    distance: int = None
) -> np.ndarray:
    """
    对输入数据进行峰值检测，提取峰高与峰位，返回 (n_samples, n_pixels, 2*n_peaks)。
    """
    n_samples, n_pixels, n_bands = data.shape
    features = np.zeros((n_samples, n_pixels, 2 * n_peaks), dtype=float)

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            peaks, _ = find_peaks(spectrum, height=height, distance=distance)
            if peaks.size == 0:
                continue
            heights = spectrum[peaks]
            idx_sorted = np.argsort(heights)[::-1][:n_peaks]
            sel = peaks[idx_sorted]
            for k, idxp in enumerate(sel):
                features[i, j, 2*k] = spectrum[idxp]
                features[i, j, 2*k+1] = idxp

    return features


def statistical_feature_extraction(
    data: np.ndarray
) -> np.ndarray:
    """
    计算统计特征：均值、方差、偏度、峰度，返回 (n_samples, n_pixels, 4)。
    """
    n_samples, n_pixels, _ = data.shape
    features = np.zeros((n_samples, n_pixels, 4), dtype=float)

    for i in range(n_samples):
        for j in range(n_pixels):
            spec = data[i, j, :].astype(float)
            features[i, j, 0] = spec.mean()
            features[i, j, 1] = spec.var()
            features[i, j, 2] = skew(spec)
            features[i, j, 3] = kurtosis(spec)

    return features

# ======================
# 新增：Lambert-Pearson 特征提取
# ======================
def lambert_pearson_feature_extraction(
    data: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.8,
    top_k: int = 3
) -> np.ndarray:
    """
    将原始强度比值 I0/Isample 转为吸光度，并基于 Pearson 相关系数筛选波段。
    支持按阈值或前 top_k 个绝对相关系数最高的波段提取特征。

    参数:
        data: 原始强度比值，shape (n_samples, n_pixels, n_bands)
        target: 浓度或标签，shape (n_samples, n_pixels) 或扁平 (n_samples*n_pixels,)
        threshold: Pearson 相关系数阈值（当 top_k=None 时使用）
        top_k: 若指定，则选取绝对相关系数最高的 top_k 个波段特征
    返回:
        selected: 筛选后特征，shape (n_samples, n_pixels, n_selected_bands)
    """
    # 1. 吸光度转换：A = log10(data)
    with np.errstate(divide='ignore', invalid='ignore'):
        absorbance = np.log10(data)
    absorbance = np.nan_to_num(absorbance)

    # 2. 扁平化计算相关系数
    n_samples, n_pixels, n_bands = absorbance.shape
    flat = absorbance.reshape(-1, n_bands)
    flat_target = target.reshape(-1)
    corrs = np.array([pearsonr(flat[:, i], flat_target)[0] for i in range(n_bands)])

    # 3. 根据 top_k 或 threshold 筛选
    if top_k is not None:
        idx_sorted = np.argsort(-np.abs(corrs))
        selected_idx = idx_sorted[:top_k]
    else:
        selected_idx = np.where(np.abs(corrs) >= threshold)[0]

    # 4. 重构特征
    selected_flat = flat[:, selected_idx]
    selected = selected_flat.reshape(n_samples, n_pixels, -1)
    return selected
