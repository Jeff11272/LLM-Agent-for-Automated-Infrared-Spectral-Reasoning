#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_feature_extraction.py

Common spectral feature extraction methods. Input data format: (n_samples, n_pixels, n_bands).
Each function extracts features from each pixel spectrum (length = n_bands) for every sample.

Functions included:
    1. PCA feature extraction (dimensionality reduction)
    2. NMF feature extraction (dimensionality reduction)
    3. Continuous wavelet transform (CWT) coefficient extraction
    4. Spectral derivatives (first and second order)
    5. Peak feature extraction (peak detection)
    6. Statistical features (mean, variance, skewness, kurtosis)
    7. Lambert-Pearson feature extraction (based on Lambert-Beer law and Pearson correlation)

Author:
Date:
"""

import numpy as np
from sklearn.decomposition import PCA, NMF
import pywt
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, pearsonr

# ======================
# Feature extraction functions
# ======================

def pca_feature_extraction(
    data: np.ndarray,
    n_components: int = 5
) -> np.ndarray:
    """
    Apply PCA dimensionality reduction to all spectra in input data (n_samples, n_pixels, n_bands).
    Returns shape (n_samples, n_pixels, n_components).
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
    Apply NMF dimensionality reduction to input data. Returns shape (n_samples, n_pixels, n_components).
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
    Apply continuous wavelet transform (CWT) to input data, extract max absolute coefficient for each scale.
    Returns shape (n_samples, n_pixels, n_scales).
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
    Compute first or second order derivative for input data. Returns shape (n_samples, n_pixels, n_bands).
    """
    if order not in [1, 2]:
        raise ValueError("order must be 1 or 2.")

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
    Detect peaks in input data, extract peak heights and positions. Returns (n_samples, n_pixels, 2*n_peaks).
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
    Compute statistical features: mean, variance, skewness, kurtosis. Returns (n_samples, n_pixels, 4).
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
# Lambert-Pearson feature extraction
# ======================
def lambert_pearson_feature_extraction(
    data: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.8,
    top_k: int = 2
) -> np.ndarray:
    """
    Convert raw intensity ratio I0/Isample to absorbance, then select bands by Pearson correlation.
    Supports selection by threshold or top_k highest absolute correlations.

    Args:
        data: raw intensity ratio, shape (n_samples, n_pixels, n_bands)
        target: concentration or label, shape (n_samples, n_pixels) or flat (n_samples*n_pixels,)
        threshold: Pearson correlation threshold (used if top_k=None)
        top_k: if specified, select top_k bands with highest absolute correlation
    Returns:
        selected: selected features, shape (n_samples, n_pixels, n_selected_bands)
    """
    # 1. Absorbance conversion: A = log10(data)
    with np.errstate(divide='ignore', invalid='ignore'):
        absorbance = np.log10(data)
    absorbance = np.nan_to_num(absorbance)

    # 2. Flatten and compute correlation
    n_samples, n_pixels, n_bands = absorbance.shape
    flat = absorbance.reshape(-1, n_bands)
    flat_target = target.reshape(-1)
    corrs = np.array([pearsonr(flat[:, i], flat_target)[0] for i in range(n_bands)])

    # 3. Select by top_k or threshold
    if top_k is not None:
        idx_sorted = np.argsort(-np.abs(corrs))
        selected_idx = idx_sorted[:top_k]
    else:
        selected_idx = np.where(np.abs(corrs) >= threshold)[0]

    # 4. Reconstruct features
    selected_flat = flat[:, selected_idx]
    selected = selected_flat.reshape(n_samples, n_pixels, -1)
    return selected

from sklearn.cross_decomposition import PLSRegression
def Partial_Least_Squares(X, Y, n_components=4):
    """
    使用偏最小二乘（PLS）进行特征提取。

    参数:
        X (np.ndarray): 特征数据，形状为 (样本数, 特征数)。
        Y (np.ndarray): 目标变量，形状为 (样本数,) 或 (样本数, 目标变量数)。

    返回:
        np.ndarray: 提取的特征，形状为 (样本数, n_components)。
    """
    # 初始化 PLS 模型
    pls = PLSRegression(n_components=n_components)

    # 使用 PLS 提取特征
    pls.fit(X, Y)  # 这里的 fit 是为了计算 PLS 的投影矩阵
    X_transformed = pls.transform(X)  # 提取降维后的特征

    return X_transformed