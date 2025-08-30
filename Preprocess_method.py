# spectral_preprocessing.py

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
    Apply AsLS baseline correction to each spectrum (length = n_bands) in input data (n_samples, n_pixels, n_bands).
    """
    n_samples, n_pixels, n_bands = data.shape
    corrected = np.zeros_like(data)

    # Second-order difference matrix D
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
    Apply Savitzky-Golay filter to input data: smoothing or derivative.
    """
    if window_length % 2 == 0 or window_length <= polyorder:
        raise ValueError("window_length must be odd and greater than polyorder.")

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
    Apply SNV transformation to input data: (spectrum - mean) / std.
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
    Apply MSC correction to input data.
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
    Min-Max normalization to [0,1].
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
    First-order polynomial detrending.
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

# Composite preprocessing functions

def snv_fd(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    delta: float = 1.0
) -> np.ndarray:
    """
    Apply SNV transformation first, then first derivative (Savitzky-Golay).
    """
    snv_data = standard_normal_variate(data)
    # First derivative
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
    Savitzky-Golay first derivative.
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
    Apply MSC correction first, then first derivative (Savitzky-Golay).
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
    Plot average spectra for each class.
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