from argparse import ArgumentError
from damnedlib.utils.checks import check_type
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import wavfile
from damnedlib.utils.checks import check_type
import damnedlib.signal.metrics as metrics
from damnedlib.utils.figures import save_fig


def run_quantitative_analysis(
    signal: Callable,
    snr_desired: float,
    custom_noise_bands: np.ndarray = None,
    custom_noise_gain: np.ndarray = None,
    custom_filter=None,
    fs=1000,
    T=5,
    nfft=4096,
    numtaps=129,
    generate_figures=True,
    save_png=False,
    save_tikz=False,
    file_prefix="",
    img_dir="../Attachments/img/",
    save_wav=False,
    wav_dir="../Attachments/data/",
):
    check_type(snr_desired, int, float)
    check_type(fs, int, float)
    check_type(T, int, float)
    check_type(nfft, int)
    check_type(numtaps, int)
    check_type(generate_figures, bool)
    check_type(save_png, bool)
    check_type(save_tikz, bool)

    generate_figures = save_png or save_tikz or generate_figures
    save_figures = save_png or save_tikz

    img_prefix = img_dir + file_prefix + " - "
    wav_prefix = wav_dir + file_prefix + "/"
    file_affix = f"_(SNR_desired={snr_desired:.2f})"

    ts = 1 / fs
    delta_f = fs / nfft
    t = np.arange(0, T, ts)
    f = np.arange(0, fs / 2 + delta_f, delta_f)

    if save_wav:
        wavfile.write(wav_prefix + "t" + file_affix + ".wav", fs, t)
        wavfile.write(wav_prefix + "f" + file_affix + ".wav", fs, f)

    snr = dict()
    snr_filtered = dict()
    asc = dict()

    if not custom_filter:
        b_filter, a_filter = design_signal_filter(4 * numtaps, fs)
    else:
        b_filter, a_filter = custom_filter

    if generate_figures:
        plot_filter(
            b=b_filter,
            fs=fs,
            title="Signal Filter",
            symbol="s",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
        )

    y_signal = signal(t)
    _, Syy_signal = metrics.psd_from_time(y=y_signal, fs=fs, nfft=nfft)
    power_signal = metrics.power_from_psd(Syy=Syy_signal, fs=fs, nfft=nfft)
    if generate_figures:
        plot_time(
            y=y_signal,
            t=t,
            title="Signal",
            symbol="s(t)",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
        )
        plot_power(
            Syy=Syy_signal,
            f=f,
            power=power_signal,
            title="Signal",
            symbol="S_S(f)",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
        )

    y_signal_filtered = sp.signal.lfilter(b_filter, a_filter, y_signal)
    _, Syy_signal_filtered = metrics.psd_from_time(y=y_signal_filtered, fs=fs, nfft=nfft)
    power_signal_filtered = metrics.power_from_psd(Syy=Syy_signal_filtered, fs=fs, nfft=nfft)
    if generate_figures:
        plot_time(
            y=y_signal_filtered,
            t=t,
            title="signal_filtered",
            symbol="\\tilde{s}(t)",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
        )
        plot_power(
            Syy=Syy_signal_filtered,
            f=f,
            power=power_signal_filtered,
            title="signal_filtered",
            symbol="S_{\\tilde{S}}(f)",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
        )

    if custom_noise_bands is None:
        eval_white_noise(
            fs=fs,
            T=T,
            y_signal=y_signal,
            Syy_signal=Syy_signal,
            power_signal=power_signal,
            snr_desired=snr_desired,
            nfft=nfft,
            file_prefix=file_prefix,
            file_affix=file_affix,
            save_png=save_png,
            save_tikz=save_tikz,
            generate_figures=generate_figures,
            t=t,
            f=f,
            b_filter=b_filter,
            numtaps=numtaps,
        )
        eval_lp_noise(
            fs=fs,
            T=T,
            y_signal=y_signal,
            Syy_signal=Syy_signal,
            power_signal=power_signal,
            snr_desired=snr_desired,
            nfft=nfft,
            file_prefix=file_prefix,
            file_affix=file_affix,
            save_png=save_png,
            save_tikz=save_tikz,
            generate_figures=generate_figures,
            t=t,
            f=f,
            b_filter=b_filter,
            numtaps=numtaps,
        )
        eval_hp_noise(
            fs=fs,
            T=T,
            y_signal=y_signal,
            Syy_signal=Syy_signal,
            power_signal=power_signal,
            snr_desired=snr_desired,
            nfft=nfft,
            file_prefix=file_prefix,
            file_affix=file_affix,
            save_png=save_png,
            save_tikz=save_tikz,
            generate_figures=generate_figures,
            t=t,
            f=f,
            b_filter=b_filter,
            numtaps=numtaps,
        )
    else:
        generate_noise_and_evaluate(
            bands=custom_noise_bands,
            gain=custom_noise_gain,
            name="custom",
            y_signal=y_signal,
            Syy_signal=Syy_signal,
            power_signal=power_signal,
            snr_desired=snr_desired,
            fs=fs,
            T=T,
            nfft=nfft,
            file_prefix=file_prefix,
            file_affix=file_affix,
            save_png=save_png,
            save_tikz=save_tikz,
            generate_figures=generate_figures,
            f=f,
            t=t,
            b_filter=b_filter,
            numtaps=numtaps,
        )


def design_signal_filter(numtaps, fs):
    bands = [0, 100, 110, fs / 2]
    gain = [1, 0.0001]
    b_filter = sp.signal.remez(numtaps, bands, gain, fs=fs)
    return b_filter, [1]


def plot_filter(
    b: np.ndarray,
    fs: float,
    title: str = None,
    symbol: str = None,
    filename: str = None,
    save_png: bool = False,
    save_tikz: bool = False,
    file_prefix: str = None,
    file_affix: str = None,
):
    if filename is None:
        filename = file_prefix + "_filter_" + file_affix
    w, h = sp.signal.freqz(b)
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("$f$ $[Hz]$")
    ax.set_ylabel(f"$H_{{{symbol}}}(f) [dB]$")
    ax.set_title(title.replace("_", "\_"))
    ax.plot(w / (2 * np.pi) * fs, 20 * np.log10(abs(h)))
    save_fig(fig=fig, filename=filename, save_png=save_png, save_tikz=save_tikz)
    return fig


def plot_time(
    y: np.ndarray,
    t: np.ndarray,
    title: str = None,
    symbol: str = None,
    filename: str = None,
    save_png: bool = False,
    save_tikz: bool = False,
    file_prefix: str = None,
    file_affix: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
):
    if filename is None:
        filename = file_prefix + title + "_time_" + file_affix
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel("$t$ $[s]$")
    ax.set_ylabel(f"${symbol}$ $[pt]$")
    ax.set_title(title.replace("_", "\_"))
    ax.plot(t, y)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    save_fig(fig=fig, filename=filename, save_png=save_png, save_tikz=save_tikz)
    return fig


def plot_power(
    Syy: np.ndarray,
    f: np.ndarray,
    power: float,
    title: str = None,
    symbol: str = None,
    filename: str = None,
    save_png: bool = None,
    save_tikz: bool = None,
    file_prefix: str = None,
    file_affix: str = None,
    xlim: tuple = None,
    ylim: tuple = None,
):
    if filename is None:
        filename = file_prefix + title + "_power_" + file_affix
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(f"$f$ $[Hz]$")
    ax.set_ylabel(f"${symbol}$ $[pT^2 / Hz]$")
    ax.semilogy(f, Syy)
    ax.set_title(title.replace("_", "\_") + f" $P = {power:.2f}\\;pT^2$")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    save_fig(fig=fig, filename=filename, save_png=save_png, save_tikz=save_tikz)
    return fig


def eval_white_noise(
    fs: float,
    T: float,
    y_signal: np.ndarray,
    Syy_signal: np.ndarray,
    power_signal: float,
    snr_desired: float,
    nfft: int,
    file_prefix: str,
    file_affix: str,
    save_png: bool,
    save_tikz: bool,
    generate_figures: bool,
    f: np.ndarray,
    t: np.ndarray,
    b_filter: np.ndarray,
    numtaps: int,
):
    bands = None
    gain = None
    generate_noise_and_evaluate(
        bands=bands,
        gain=gain,
        name="w",
        y_signal=y_signal,
        Syy_signal=Syy_signal,
        power_signal=power_signal,
        snr_desired=snr_desired,
        fs=fs,
        T=T,
        nfft=nfft,
        file_prefix=file_prefix,
        file_affix=file_affix,
        save_png=save_png,
        save_tikz=save_tikz,
        generate_figures=generate_figures,
        f=f,
        t=t,
        b_filter=b_filter,
        numtaps=numtaps,
    )


def eval_lp_noise(
    fs: float,
    T: float,
    y_signal: np.ndarray,
    Syy_signal: np.ndarray,
    power_signal: float,
    snr_desired: float,
    nfft: int,
    file_prefix: str,
    file_affix: str,
    save_png: bool,
    save_tikz: bool,
    generate_figures: bool,
    f: np.ndarray,
    t: np.ndarray,
    b_filter: np.ndarray,
    numtaps: int,
):
    bands = [0, fs * 0.23, fs * 0.27, fs / 2]
    gain = [2 ** 0.5, 0.001]
    generate_noise_and_evaluate(
        bands=bands,
        gain=gain,
        name="lp",
        y_signal=y_signal,
        Syy_signal=Syy_signal,
        power_signal=power_signal,
        snr_desired=snr_desired,
        fs=fs,
        T=T,
        nfft=nfft,
        file_prefix=file_prefix,
        file_affix=file_affix,
        save_png=save_png,
        save_tikz=save_tikz,
        generate_figures=generate_figures,
        f=f,
        t=t,
        b_filter=b_filter,
        numtaps=numtaps,
    )


def eval_hp_noise(
    fs: float,
    T: float,
    y_signal: np.ndarray,
    Syy_signal: np.ndarray,
    power_signal: float,
    snr_desired: float,
    nfft: int,
    file_prefix: str,
    file_affix: str,
    save_png: bool,
    save_tikz: bool,
    generate_figures: bool,
    f: np.ndarray,
    t: np.ndarray,
    b_filter: np.ndarray,
    numtaps: int,
):
    bands = [0, fs * 0.23, fs * 0.27, fs / 2]
    gain = [0.001, 2 ** 0.5]
    generate_noise_and_evaluate(
        bands=bands,
        gain=gain,
        name="hp",
        y_signal=y_signal,
        Syy_signal=Syy_signal,
        power_signal=power_signal,
        snr_desired=snr_desired,
        fs=fs,
        T=T,
        nfft=nfft,
        file_prefix=file_prefix,
        file_affix=file_affix,
        save_png=save_png,
        save_tikz=save_tikz,
        generate_figures=generate_figures,
        f=f,
        t=t,
        b_filter=b_filter,
        numtaps=numtaps,
    )


def generate_noise_and_evaluate(
    bands: np.ndarray,
    gain: np.ndarray,
    name: str,
    y_signal: np.ndarray,
    Syy_signal: np.ndarray,
    power_signal: float,
    snr_desired: float,
    fs: float,
    T: float,
    nfft: int,
    file_prefix: str,
    file_affix: str,
    save_png: bool,
    save_tikz: bool,
    generate_figures: bool,
    f: np.ndarray,
    t: np.ndarray,
    b_filter: np.ndarray,
    numtaps: int,
):
    y_noise, Syy_noise, power_noise = generate_noise(
        bands=bands,
        gain=gain,
        name=name,
        power_signal=power_signal,
        snr_desired=snr_desired,
        fs=fs,
        T=T,
        nfft=nfft,
        file_prefix=file_prefix,
        file_affix=file_affix,
        save_png=save_png,
        save_tikz=save_tikz,
        generate_figures=generate_figures,
        f=f,
        t=t,
        numtaps=numtaps,
    )
    evaluate(
        fs=fs,
        nfft=nfft,
        b_filter=b_filter,
        t=t,
        f=f,
        y_signal=y_signal,
        Syy_signal=Syy_signal,
        power_signal=power_signal,
        y_noise=y_noise,
        Syy_noise=Syy_noise,
        power_noise=power_noise,
        name=name,
        generate_figures=generate_figures,
        save_png=save_png,
        save_tikz=save_tikz,
        file_prefix=file_prefix,
        file_affix=file_affix,
    )


def generate_noise(
    bands: np.ndarray,
    gain: np.ndarray,
    name: str,
    power_signal: float,
    snr_desired: float,
    fs: float,
    T: float,
    nfft: int,
    file_prefix: str,
    file_affix: str,
    save_png: bool,
    save_tikz: bool,
    generate_figures: bool,
    f: np.ndarray,
    t: np.ndarray,
    numtaps: int,
    xlim_time: tuple = None,
    ylim_time: tuple = None,
    xlim_psd: tuple = None,
    ylim_psd: tuple = None,
):

    power_noise_desired = power_signal / snr_desired
    y_noise = np.random.normal(0, 1, size=T * fs)

    if bands is not None and gain is not None:
        if len(bands) == 2 * len(gain):
            b = sp.signal.remez(numtaps, bands, gain, fs=fs)
        elif len(bands) == len(gain):
            b = sp.signal.firls(numtaps, bands, gain, fs=fs)
        else:
            raise (ArgumentError(f"Incompatible bands and gain. {bands=}, {gain=}"))
        y_noise = sp.signal.lfilter(b, 1, np.pad(y_noise, [0, numtaps // 2]))[numtaps // 2 :]
    else:
        b = None

    _, Syy_noise = metrics.psd_from_time(y=y_noise, fs=fs, nfft=nfft)
    power_noise = metrics.power_from_psd(Syy=Syy_noise, fs=fs, nfft=nfft)

    scaling_factor = (power_noise_desired / power_noise) ** 0.5
    y_noise = y_noise * scaling_factor
    _, Syy_noise = metrics.psd_from_time(y=y_noise, fs=fs, nfft=nfft)
    power_noise = metrics.power_from_psd(Syy=Syy_noise, fs=fs, nfft=nfft)

    if generate_figures:
        if b is not None:
            plot_filter(
                b=b,
                fs=fs,
                title="noise_" + name,
                symbol=name,
                save_png=save_png,
                save_tikz=save_tikz,
                file_prefix=file_prefix,
                file_affix=file_affix,
            )
        plot_time(
            y=y_noise,
            t=t,
            title="noise_" + name,
            symbol=f"n_{{{name}}}",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
            xlim=xlim_time,
            ylim=ylim_time,
        )
        plot_power(
            Syy=Syy_noise,
            f=f,
            power=power_noise,
            title="noise_" + name,
            symbol=f"N_{{{name}}}",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
            ylim=ylim_psd,
            xlim=xlim_psd,
        )

    return y_noise, Syy_noise, power_noise


def evaluate(
    fs: float,
    nfft: int,
    b_filter: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    y_signal: np.ndarray,
    Syy_signal: np.ndarray,
    power_signal: float,
    y_noise: np.ndarray,
    Syy_noise: np.ndarray,
    power_noise: np.ndarray,
    name: str,
    generate_figures: bool,
    save_png: bool,
    save_tikz: bool,
    file_prefix: str,
    file_affix: str,
    xlim_time: tuple = None,
    ylim_time: tuple = None,
    xlim_psd: tuple = None,
    ylim_psd: tuple = None,
):
    y_meas = y_signal + y_noise
    _, Syy_meas = metrics.psd_from_time(y=y_meas, fs=fs, nfft=nfft)
    power_input = metrics.power_from_psd(Syy=Syy_meas, fs=fs, nfft=nfft)

    y_meas_filtered = sp.signal.lfilter(b_filter, 1, y_meas)
    _, Syy_meas_filtered = metrics.psd_from_time(y=y_meas_filtered, fs=fs, nfft=nfft)
    power_input_filtered = metrics.power_from_psd(Syy=Syy_meas_filtered, fs=fs, nfft=nfft)

    Syy_noise_scaled = (Syy_signal * Syy_noise) ** 0.5
    power_noise_scaled = metrics.power_from_psd(Syy=Syy_noise_scaled, fs=fs, nfft=nfft)

    Syy_noise_db = 10 * np.log10(Syy_noise)
    Syy_meas_db = 10 * np.log10(Syy_meas)

    y_noise_filtered = sp.signal.lfilter(b_filter, 1, y_noise)
    _, Syy_noise_filtered = metrics.psd_from_time(y=y_noise_filtered, fs=fs, nfft=nfft)
    power_noise_filtered = metrics.power_from_psd(Syy=Syy_noise_filtered, fs=fs, nfft=nfft)

    y_signal_filtered = sp.signal.lfilter(b_filter, 1, y_signal)
    _, Syy_signal_filtered = metrics.psd_from_time(y=y_signal_filtered, fs=fs, nfft=nfft)
    power_signal_filtered = metrics.power_from_psd(Syy=Syy_signal_filtered, fs=fs, nfft=nfft)

    snr = power_signal / power_noise
    snr_filtered = power_signal_filtered / power_noise_filtered
    asc = metrics.asc_from_psd(Syy_signal=Syy_signal, Syy_noise=Syy_noise, fs=fs, nfft=nfft)

    if generate_figures:
        plot_time(
            y=y_meas,
            t=t,
            title="meas_" + name + f", SNR = {10*np.log(snr):.2f} dB",
            symbol=f"s(t) + n(t)",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
            xlim=xlim_time,
            ylim=ylim_time,
        )
        plot_time(
            y=y_meas_filtered,
            t=t,
            title="meas_filtered" + name + f", SNR = {10*np.log(snr_filtered):.2f} dB",
            symbol=f"\\tilde{{s}}(t) + \\tilde{{n}}(t)",
            save_png=save_png,
            save_tikz=save_tikz,
            file_prefix=file_prefix,
            file_affix=file_affix,
            xlim=xlim_time,
            ylim=ylim_time,
        )

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("$f$ $[Hz]$")
        ax.set_ylabel("$S_{dB}(f)$ $[dB]$")
        ax.plot(f, Syy_noise_db, label=f"$S_{{N,\,dB}}(f)$")
        ax.plot(
            f,
            Syy_meas_db,
            label=f"$S_{{M,\,dB}}(f)$",
        )
        ax.fill_between(f, Syy_meas_db, Syy_noise_db, color="green", alpha=0.4, label="ASC")
        ax.set_title(f"$ASC={asc:.2f}\, dB\, Hz$")
        ax.legend()
        if xlim_psd is not None:
            ax.set_xlim(xlim_psd)
        if ylim_psd is not None:
            ax.set_ylim(ylim_psd)
        save_fig(
            fig=fig,
            filename=file_prefix + "asc_" + name + file_affix,
            save_png=save_png,
            save_tikz=save_tikz,
        )
