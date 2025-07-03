import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def baseline_correction_asls(
    data: np.ndarray,
    lam: float = 1e5,
    p: float = 0.001,
    niter: int = 10
) -> np.ndarray:
    """
    对输入数据 (n_samples, n_pixels, n_bands) 中每个光谱（长度 = n_bands）逐一进行 AsLS 基线校正。
    """
    n_samples, n_pixels, n_bands = data.shape
    corrected = np.zeros_like(data)

    # 二阶差分矩阵 D
    L = n_bands
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            w = np.ones(L)
            for _ in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.T.dot(D)
                z = spsolve(Z, w * spectrum)
                w = p * (spectrum > z) + (1 - p) * (spectrum < z)
            corrected[i, j, :] = spectrum - z

    return corrected


def savitzky_golay_smoothing(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    deriv: int = 0,
    delta: float = 1.0
) -> np.ndarray:
    """
    对输入数据应用 Savitzky-Golay 滤波：平滑或求导。
    """
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length 必须为奇数，且大于 polyorder。")

    n_samples, n_pixels, _ = data.shape
    result = np.zeros_like(data)

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            result[i, j, :] = savgol_filter(
                spectrum,
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv,
                delta=delta
            )

    return result


def standard_normal_variate(data: np.ndarray) -> np.ndarray:
    """
    对输入数据应用 SNV 变换: (spectrum - mean) / std。
    """
    n_samples, n_pixels, n_bands = data.shape
    snv = np.zeros_like(data)

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            mean = spectrum.mean()
            std = spectrum.std()
            if std == 0:
                snv[i, j, :] = spectrum - mean
            else:
                snv[i, j, :] = (spectrum - mean) / std

    return snv


def multiplicative_scatter_correction(data: np.ndarray) -> np.ndarray:
    """
    对输入数据进行 MSC 校正。
    """
    n_samples, n_pixels, n_bands = data.shape
    corrected = np.zeros_like(data)

    for i in range(n_samples):
        reference = data[i, :, :].mean(axis=0)
        for j in range(n_pixels):
            y = data[i, j, :].astype(float)
            a, b = np.polyfit(reference, y, deg=1)
            if a == 0:
                corrected[i, j, :] = y - b
            else:
                corrected[i, j, :] = (y - b) / a

    return corrected


def normalization_min_max(data: np.ndarray) -> np.ndarray:
    """
    Min-Max 归一化到 [0,1]。
    """
    n_samples, n_pixels, n_bands = data.shape
    normed = np.zeros_like(data)

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            min_val, max_val = spectrum.min(), spectrum.max()
            if max_val == min_val:
                normed[i, j, :] = 0
            else:
                normed[i, j, :] = (spectrum - min_val) / (max_val - min_val)

    return normed


def detrend_spectrum(data: np.ndarray) -> np.ndarray:
    """
    一阶多项式去趋势。
    """
    n_samples, n_pixels, n_bands = data.shape
    detrended = np.zeros_like(data)
    x = np.arange(n_bands)

    for i in range(n_samples):
        for j in range(n_pixels):
            spectrum = data[i, j, :].astype(float)
            a, b = np.polyfit(x, spectrum, deg=1)
            trend = a * x + b
            detrended[i, j, :] = spectrum - trend

    return detrended

# 新增复合预处理函数：

def snv_fd(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    delta: float = 1.0
) -> np.ndarray:
    """
    先做 SNV 变换，再对结果进行一阶导数 (Savitzky-Golay)。
    """
    snv_data = standard_normal_variate(data)
    # 一阶导数
    return savitzky_golay_smoothing(
        snv_data,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta
    )


def sg_fd(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    delta: float = 1.0
) -> np.ndarray:
    """
    Savitzky-Golay 一阶导数。
    """
    return savitzky_golay_smoothing(
        data,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta
    )


def msc_fd(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    delta: float = 1.0
) -> np.ndarray:
    """
    先做 MSC 校正，再对结果进行一阶导数 (Savitzky-Golay)。
    """
    msc_data = multiplicative_scatter_correction(data)
    return savitzky_golay_smoothing(
        msc_data,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta
    )


def plot_average_spectra(data: np.ndarray, labels: np.ndarray, title: str):
    """
    绘制各类别平均光谱曲线。
    """
    classes = np.unique(labels)
    n_bands = data.shape[2]
    wavelengths = np.arange(n_bands)

    plt.figure(figsize=(8, 6))
    for cls in classes:
        class_data = data[labels == cls, :, :]
        avg_spectrum = class_data.mean(axis=(0, 1))
        plt.plot(wavelengths, avg_spectrum, label=f"Class {cls}")

    plt.title(title)
    plt.xlabel("Band Index")
    plt.ylabel("Average Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
