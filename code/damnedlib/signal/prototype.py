import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.signal
from damnedlib.utils.checks import check_type
from damnedlib.signal.metrics import psd_from_time
from damnedlib.utils.figures import save_fig

import damnedlib.utils.jupyter_settings


class Prototype:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str = "",
        x_unit: str = "s",
        y_unit: str = "",
        dydx: np.ndarray = None,
        extrapolate: str = "periodic",
        x2: np.ndarray = None,
        y2: np.ndarray = None,
        dydx2: np.ndarray = None,
    ):
        check_type(x, np.ndarray)
        check_type(y, np.ndarray)
        check_type(name, str)
        check_type(x_unit, str)
        check_type(y_unit, str)
        check_type(dydx, np.ndarray, type(None))
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have the same shape but got shapes: {x.shape} and {y.shape}."
            )
        self.x = x
        self.y = y
        self.name = name
        self.x_unit = x_unit
        self.y_unit = y_unit
        if dydx is None:
            self.dydx = np.zeros_like(y)
        else:
            if x.shape != dydx.shape:
                raise ValueError(
                    f"x and dydx must have the same shape but got shapes: {x.shape} and {dydx.shape}."
                )
            self.dydx = dydx
        self.extrapolate = extrapolate

        self.x2 = x2
        self.y2 = y2
        if dydx2 is None:
            self.dydx2 = np.zeros_like(y2)
        else:
            if x2.shape != dydx2.shape:
                raise ValueError(
                    f"x2 and dydx2 must have the same shape but got shapes: {x2.shape} and {dydx2.shape}."
                )
            self.dydx2 = dydx2

        self.gen = interpolate.CubicHermiteSpline(
            self.x, self.y, self.dydx, extrapolate=self.extrapolate
        )
        if self.x2 is not None:
            self.gen2 = interpolate.CubicHermiteSpline(
                self.x2, self.y2, self.dydx2, extrapolate=self.extrapolate
            )
        else:
            self.gen2 = None

    def __str__(self) -> str:
        return f'Prototype signal "{self.name}"'

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __call__(self, x) -> np.ndarray:
        if self.gen2 is not None:
            return np.asarray(self.gen(x)) + np.asarray(self.gen2(x))
        else:
            return np.asarray(self.gen(x))

    def plot(
        self,
        ts,
        T=None,
        nfft=None,
        nperseg=None,
        save_png=False,
        save_tikz=False,
        dir=None,
        ylim_time=None,
        ylim_psd=None,
        xlim_psd=None,
    ):
        fs = 1 / ts
        if T is None:
            T = self.x[-1]

        x = np.arange(0, T, ts)
        y = self(x)

        f, Pyy_den = psd_from_time(y, fs=fs, nfft=nfft, nperseg=nperseg)
        Syy_den = Pyy_den ** 0.5

        fig, ax = plt.subplots()
        fig.suptitle(f"{self.name}-Prototype-Signal")
        ax.set_title(f"Time-Domain")
        ax.set_xlabel(f"Time [${self.x_unit}$]")
        ax.set_ylabel(f"Amplitude [${self.y_unit}$]")
        ax.plot(x, y)
        if ylim_time is not None:
            ax.set_ylim(ylim_time)
        save_fig(fig, f"{dir}{self.name}_time", save_png=save_png, save_tikz=save_tikz)

        fig, ax = plt.subplots()
        fig.suptitle(f"{self.name}-Prototype-Signal")
        ax.set_title(f"Frequency-Domain")
        ax.set_xlabel(f"Frequency [$Hz$]")
        ax.set_ylabel(f"PSD [${self.y_unit}^2/Hz$]")
        ax.semilogy(f, Pyy_den)
        if ylim_psd is not None:
            ax.set_ylim(ylim_psd)
        if xlim_psd is not None:
            ax.set_xlim(xlim_psd)
        save_fig(fig, f"{dir}{self.name}_psd", save_png=save_png, save_tikz=save_tikz)

    @classmethod
    def MCG(cls):
        """
        Standard MCG signal based on Eric's squid measurements at the ptb.
        """
        x = np.array(
            [
                0,
                0.25,
                0.3,
                0.35,  # P-Wave
                0.44,
                0.47,
                0.5,
                0.52,
                0.56,  # QRS-Complex
                0.65,
                0.8,
                0.9,
                1,  # T-Wave
            ]
        )

        y = np.array(
            [0, 0, 0.05, 0, 0, -0.15, 1, -0.1, 0, 0, 0.18, 0, 0]  # P-Wave  # QRS-Complex  # T-Wave
        )
        y = y * 70  # assume 70 picoTesla R-peak

        return cls(x, y, "MCG (I)", y_unit="pT")

    @classmethod
    def PWAVE(cls):
        """
        P-Wave of MCG signal based on Eric's SQUID measurements at the ptb
        """
        x = np.array(
            [
                0,
                0.25,
                0.3,
                0.35,  # P-Wave
                1,
            ]
        )

        y = np.array([0, 0, 0.03, 0, 0])  # P-Wave
        y = y * 70  # assume 70 picoTesla R-peak

        return cls(x, y, "P-Wave", y_unit="pT")

    @classmethod
    def QRS(cls):
        """
        QRS-Complex of MCG signal based on Eric's SQUID measurements at the ptb
        """
        x = np.array(
            [
                0,
                0.44,
                0.47,
                0.5,
                0.52,
                0.56,  # QRS-Complex
                1,
            ]
        )

        y = np.array([0, 0, -0.15, 1, -0.1, 0, 0])  # QRS-Complex
        y = y * 70  # assume 70 picoTesla R-peak

        return cls(x, y, "QRS-Complex", y_unit="pT")

    @classmethod
    def TWAVE(cls):
        """
        T-Wave of MCG signal based on Eric's SQUID measurements at the ptb
        """
        x = np.array(
            [
                0,
                0.6,
                0.75,
                0.85,
                1,  # T-Wave
            ]
        )

        y = np.array([0, 0, 0.18, 0, 0])  # T-Wave
        y = y * 70  # assume 70 picoTesla R-peak

        return cls(x, y, "T-Wave", y_unit="pT")

    @classmethod
    def MCG_AWI(cls):
        """
        MCG signal of patient with anterior wall infarction based on
        example 26, lead V3 in Book "EKG-Kurs für Isabel"
        """

        x = np.array(
            [
                0,
                0.25,
                0.3,
                0.35,  # P-Wave
                0.44,
                0.47,
                0.5,
                0.52,
                0.56,  # QRS-Complex
                0.75,
                0.90,
                1,  # T-Wave
            ]
        )

        y = np.array([0, 0, 0.05, 0, 0, 0.05, -1, 0.1, 0.1, 0.18, 0, 0])

        dydx = np.array([0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 0])

        y = y * 70  # assume 70 picoTesla R-peak

        return cls(
            x=x,
            y=y,
            dydx=dydx,
            name="Anterior Wall Infarction (V3)",
            x_unit="s",
            y_unit="pT",
        )

    @classmethod
    def MCG_LBBB(cls):
        """
        MCG signal of patient with left bundle branch block
        example 13, lead V6 in Book "EKG-Kurs für Isabel"
        """
        x = np.array(
            [
                0,
                0.25,
                0.3,
                0.35,  # P-End
                0.44,  # QRS-Start
                0.46,  # First Peak
                0.50,  # Down again
                0.53,  # Up again
                0.6,  # QRS-Complex
                0.72,  # t-start
                0.83,  # t-peak
                0.9,  # t-end
                1,  # end
            ]
        )

        y = np.array(
            [
                0,
                0,
                0.1,
                0,  # P-Wave
                0,
                1,
                0.8,
                0.96,
                0,  # QRS-Complex
                -0.075,  # t-start
                -0.2,  # t-peak
                0,  # t-end
                0,  # end
            ]
        )

        dydx = np.array(
            [
                0,
                0,
                0,
                0,  # P-Wave
                0,
                0,
                0,
                0,
                -100,  # QRS-Complex
                -50,  # t-start
                0,  # t-peak
                0,  # t-end
                0,  # end
            ]
        )

        y = y * 70  # assume 70 picoTesla R-peak

        return cls(
            x=x,
            y=y,
            dydx=dydx,
            name="Left Bundle Branch Block (V6)",
            x_unit="s",
            y_unit="pT",
        )

    @classmethod
    def MCG_AF(cls):
        """
        MCG signal of patient with artrial fibrillation
        example 38, lead III in Book "EKG-Kurs für Isabel"
        """
        d1 = 0.15 * 5
        d2 = d1 + 0.15 * 7
        d3 = d2 + 0.15 * 10
        d4 = d3 + 0.15 * 6
        x = np.array(
            [
                0,  # start
                0.24,
                0.27,
                0.3,
                0.32,
                0.36,  # QRS-Complex
                0.4,
                0.55,
                0.65,
                d1 + 0,  # start
                d1 + 0.24,
                d1 + 0.27,
                d1 + 0.3,
                d1 + 0.32,
                d1 + 0.36,  # QRS-Complex
                d1 + 0.4,
                d1 + 0.55,
                d1 + 0.65,
                d2 + 0,  # start
                d2 + 0.24,
                d2 + 0.27,
                d2 + 0.3,
                d2 + 0.32,
                d2 + 0.36,  # QRS-Complex
                d2 + 0.4,
                d2 + 0.55,
                d2 + 0.65,
                d3 + 0,  # start
                d3 + 0.24,
                d3 + 0.27,
                d3 + 0.3,
                d3 + 0.32,
                d3 + 0.36,  # QRS-Complex
                d3 + 0.4,
                d3 + 0.55,
                d3 + 0.65,
                d4 + 0,  # start
                d4 + 0.24,
                d4 + 0.27,
                d4 + 0.3,
                d4 + 0.32,
                d4 + 0.36,  # QRS-Complex
                d4 + 0.4,
                d4 + 0.55,
                d4 + 0.65,
                d4 + 1,  # end
            ]
        )

        y = np.array(
            [0, 0, -0.075, 0.8, -0.3, 0, 0, 0.1, 0] * 5 + [0]  # P-Wave  # QRS-Complex  # T-Wave
        )

        y = y * 70  # assume 70 picoTesla R-peak

        x2 = np.array(
            [
                0,
                0.02,
                0.04,
                0.12,
                0.15,
            ]
        )

        y2 = np.array([0, 0, 0.04, 0, 0])

        y2 = y2 * 70

        return cls(
            x=x, y=y, name="Atrial Fibrillation (III)", x_unit="s", y_unit="pT", x2=x2, y2=y2
        )

    @classmethod
    def MCG_LQT(cls):
        """
        MCG signal of patient with long qt time
        lead I
        """

        x = np.array(
            [
                0,
                0.05,
                0.1,
                0.15,  # P-Wave
                0.24,
                0.27,
                0.3,
                0.32,
                0.36,  # QRS-Complex
                0.4,
                0.75,
                0.9,
                1,  # T-Wave
            ]
        )

        y = np.array(
            [0, 0, 0.05, 0, 0, -0.15, 1, -0.1, 0, 0, 0.10, 0, 0]  # P-Wave  # QRS-Complex  # T-Wave
        )
        y = y * 70  # assume 70 picoTesla R-peak

        return cls(
            x=x,
            y=y,
            name="Long-QT syndrome (I)",
            x_unit="s",
            y_unit="pT",
        )

    @classmethod
    def MCG_HK(cls):
        """
        MCG signal of patient with hyperkalaemia
        Lead V3
        Based on the examples found on the following website (2020.03.14):

        https://litfl.com/hypokalaemia-ecg-library/

        https://www.pragueicu.com/ecg-academy/hyperkalemia

        https://uploads-ssl.webflow.com/5ff99f5f7aec6fe4770f6b92/6025339b854c71617cc7915e_liyeCLgWQWsrSnIWb59VFMxvNfrERpwUI70Us9-KcvgUkTQf_HlFMFcekSUCkp4WIyM45SqgfmU2P53TAcvhRAXq_3TOv3e-TF7bd8VlAQ0mFwMQ0TePEI4CbGJ0fSngy26YzlY.png
        """

        x = np.array(
            [
                0,
                0.2,
                0.25,
                0.35,  # P-Wave
                0.45,
                0.47,
                0.49,
                0.54,
                0.61,  # QRS-Complex
                0.71,
                0.78,
                0.90,
                1.2,  # T-Wave
                1.22,
            ]
        )

        y = np.array(
            [
                0,
                0,
                0.02,
                0,
                0,
                -0.05,
                0.5,
                -1,
                0.1,
                1,
                0.2,
                0,
                0,
                0,
            ]
        )
        dydx = np.array([0, 0, 0, 0, 0, 0, 0, 50, 100, 0, -250, 0, 0, 0])
        y = y * 70  # assume 70 picoTesla R-peak

        return cls(
            x=x,
            y=y,
            name="Hyperkalaemia (V3)",
            dydx=dydx,
            x_unit="s",
            y_unit="pT",
        )
