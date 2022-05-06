from damnedlib.utils.checks import check_type
from damnedlib.utils.figures import save_fig
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def plot_filter_response(
    b, a=1, fs=2, save=False, title=None, file_name=None, dir="../Attachments/img/"
):
    check_type(b, int, float, list, np.ndarray)
    check_type(a, int, float, list, np.ndarray)
    check_type(fs, int, float)
    check_type(save, bool)
    check_type(title, type(None), str)
    check_type(file_name, type(None), str)
    check_type(dir, str)

    freq, response = signal.freqz(b, a, fs=fs)
    fig, ax = plt.subplots(2, 1)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    ax[0].plot(freq, 20 * np.log10(np.abs(response)))
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("$20 \cdot \log(A(jw))\,dB$")
    ax[0].set_xlabel("Frequency (Hz)")

    ax[1].plot(freq, np.unwrap(np.angle(response)))
    ax[1].set_title("Phase Response")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("$B(jw)$ [rad]")

    if save is True:
        save_fig(fig, dir + file_name, save_png=True, save_tikz=False)

    return fig, ax
