import numpy as np
import scipy as sp


def snr_from_psd(
    Syy_signal: np.ndarray,
    Syy_noise: np.ndarray,
    fs,
    nfft,
):
    power_noise = power_from_psd(Syy=Syy_noise, fs=fs, nfft=nfft)
    power_signal = power_from_psd(Syy=Syy_signal, fs=fs, nfft=nfft)
    snr = power_signal / power_noise
    return snr


def asc_from_psd(
    Syy_signal: np.ndarray,
    Syy_noise: np.ndarray,
    fs,
    nfft,
):
    Syy_meas = Syy_signal + Syy_noise
    Syy_noise_db = 10 * np.log10(Syy_noise)
    Syy_meas_db = 10 * np.log10(Syy_meas)
    asc = sp.integrate.simpson(y=Syy_meas_db - Syy_noise_db, dx=fs / nfft)
    return asc


def psd_from_time(
    y: np.ndarray, fs: float, nfft: int = None, nperseg: int = None, window: str = "flattop"
):
    """Calculates the psd of y using welches method.

    Args:
        y (np.ndarray): the time series data to calculate the psd of
        fs (float): sample rate in Hz
        nfft (int): length of fft (if longer then nperseg uses zero padding)
        nperseg (int, optional): Number of samples per segment. Defaults to fs this will result in windows being 1 second long  .
        window (str, optional): which window to use . Defaults to "flattop".

    Returns:
        f (np.ndarray): Array of sample frequencies.
        Pxx (np.ndarray): Power spectral density or power spectrum of y.
    """
    if nperseg is None:
        nperseg = int(fs)
    return sp.signal.welch(
        y, fs, noverlap=int(nperseg / 2), nperseg=nperseg, nfft=nfft, window=window
    )


def power_from_psd(Syy: np.ndarray, fs: float, nfft: int):
    return sp.integrate.simpson(y=Syy, dx=fs / nfft)
