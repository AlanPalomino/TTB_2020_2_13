# -*- coding: utf-8 -*-
# %%

# ===================== Librerias Utilizadas ====================== #
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pyhrv
import time
import wfdb
import re

# ================= Funciones y Definiciones ====================== #
def timeit(func, *args, **kwargs):
    s_time = time.time()
    func(*args, **kwargs)
    e_time = time.time()
    print(f"Function {func.__name__} execution time: {e_time - s_time}")


# ================= Importando Bases de Datos
class Case():
    """
    Generador del compendio de registros y señales para un caso particular.

    Object looks for files in the directory provided, to build a list
    of records from the same case.
    """

    RECORDS = []

    def __init__(self, case_dir: Path, sig_thresh: int=1000):
        self._case_dir = case_dir
        self._case_name = case_dir.stem
        self._sig_thresh = sig_thresh
        self.pathology = re.search(
            f"([a-z_]*)(_{self._case_name})",
            str(case_dir)
        ).groups()[0]
        self._get_records()

    def __str__(self):
        """Prints data of self and internal records"""
        print(f"Case: {self._case_name} - Records above {self._sig_thresh} samples ->")
        for record in self.RECORDS:
            print(record)
        return f"{5*' * '}End of case {self._case_name}{5*' * '}"

    def __getitem__(self, index):
        """Extract record as a list"""
        return self.RECORDS[index]

    def _get_records(self):
        for hea_path in self._case_dir.glob(f"{self._case_name}*[0-9].hea"):
            h = wfdb.rdheader(str(hea_path.parent.joinpath(hea_path.stem)))
            self._get_names(h.seg_name, h.seg_len)

    def _get_names(self, seg_names: list, seg_lens: list):
        for name, slen in zip(seg_names, seg_lens):
            if slen < self._sig_thresh or "~" in name:
                continue
            self._get_data(self._case_dir.joinpath(name))

    def _get_data(self, path: Path):
        self.RECORDS.append(
            Record(path, self._case_name)
        )


class Record():
    def __init__(self, record_dir: Path, case: str):
        reco = wfdb.rdrecord(str(record_dir))
        head = wfdb.rdheader(str(record_dir))
        self.record_dir = record_dir
        self.case = case
        self.name = head.record_name
        self.time = head.base_time
        self.date = head.base_date
        self.fs = reco.fs
        self.len = reco.sig_len
        self.n_sig = reco.n_sig
        self.sig_names = reco.sig_name
        self.units = reco.units

    def __str__(self):
        return f"\t Record: {self.name}, Length:{self.len}, \t# of signals: {self.n_sig} -> {self.sig_names}"

    def __getitem__(self, item):
        try:
            sig_idx = self.sig_names.index(item)
            signals = self._get_signals()
            return signals[:, sig_idx]
        except ValueError:
            raise KeyError(f"'{item}' isn't a valid key. Signals in record:{self.sig_names}")

    def _get_signals(self):
        reco = wfdb.rdrecord(str(self.record_dir))
        return reco.p_signal

    def plot(self):
        fig, axs = plt.subplots(self.n_sig, 1)
        signals = self._get_signals()

        fig.suptitle(f"Record {self.name} of case {self.case}")
        for a, n, u, s in zip(axs, self.sig_names, self.units, list(zip(*signals))):
            a.set_ylabel(f"{n}/{u}")
            a.plot(s)
        axs[-1].set_xlabel("Samples")
        plt.show()
        
        


# ================= Ventaneo de señales
class Windowing():
    """
    Funciones de ventaneo de las señales 
    """

    def RR_Windowing(rr_signal, w_len, over, mode="sample"):
        """ 
        rr_signal :: RR vector of time in seconds
        w_time    :: Defines window time in seconds
        over      :: Defines overlapping between windows
        l_thresh  :: Gets lower threshold of window
        mode      :: Sets mode of windowing;
                        "sample" - Same sized windows, iterates by sample count.
                        "time" - Variable sized windows, iterates over time window.

        """
        means, var, skew, kurt = list(), list(), list(), list()
        step = int(w_len*(1-over))
        
        if mode == "time":
            time_vec = np.cumsum(rr_signal)
            l_thresh = time_vec[0]
            while l_thresh < max(time_vec)-w_len:
                window = np.where(np.bitwise_and((l_thresh < time_vec), (time_vec < (l_thresh+w_len))))
                rr_window = RR[window]
                
                ds = stats.describe(rr_window)
                means.append(ds[2])
                var.append(ds[3])
                skew.append(ds[4])
                kurt.append(ds[5])
        
                l_thresh += step

        elif mode == "sample":
            for rr_window in [rr_signal[i:i+w_len] for i in range(0, len(rr_signal)-w_len, step)]:
                ds = stats.describe(rr_window)
                means.append(ds[2])
                var.append(ds[3])
                skew.append(ds[4])
                kurt.append(ds[5])
        
        return means, var, skew, kurt

    def RR_nonLinear_Windowing(rr_signal, w_len, over, mode="sample"):
        """
        rr_signal :: RR vector of time in seconds
        w_time    :: Defines window time in seconds
        over      :: Defines overlapping between windows
        l_thresh  :: Gets lower threshold of window
        mode      :: Sets mode of windowing;
                        "sample" - Same sized windows, iterates by sample count.
                        "time" - Variable sized windows, iterates over time window.

        """
        app_ent, samp_ent, hfd, dfa = list(), list(), list(), list()
        step = int(w_len*(1-over))
        
        if mode == "time":
            time_vec = np.cumsum(rr_signal)
            l_thresh = time_vec[0]
            while l_thresh < max(time_vec)-w_len:
                window = np.where(np.bitwise_and((l_thresh < time_vec), (time_vec < (l_thresh+w_len))))
                rr_window = RR[window]
                
                app_ent.append(entropy.app_entropy(rr_window, order=2, metric='chebyshev'))
                samp_ent.append(entropy.sample_entropy(rr_window, order=2))
                hfd.append(fractal.higuchi_fd(rr_window, kmax=10))
                dfa.append(fractal.detrended_fluctuation(rr_window))
        
                l_thresh += step

        elif mode == "sample":
            for rr_window in [rr_signal[i:i+w_len] for i in range(0, len(rr_signal)-w_len, step)]:
                app_ent.append(entropy.app_entropy(rr_window, order=2, metric='chebyshev'))
                samp_ent.append(entropy.sample_entropy(rr_window, order=2, metric='chebyshev'))
                hfd.append(fractal.higuchi_fd(rr_window, kmax=10))
                dfa.append(fractal.detrended_fluctuation(rr_window))
            
        return app_ent, samp_ent, dfa


# %%

class CustomPlots:
    #================ Custom Poincaré plot
    def poincarePlot(nni=None,
                rpeaks=None,
                show=True,
                figsize=None,
                ellipse=True,
                vectors=True,
                legend=True,
                marker='o'):
        """Creates Poincaré plot from a series of NN intervals or R-peak locations and derives the Poincaré related
        parameters SD1, SD2, SD1/SD2 ratio, and area of the Poincaré ellipse.
        References: [Tayel2015][Brennan2001]
        Parameters
        ----------
        nni : array
            NN intervals in [ms] or [s]
        rpeaks : array
            R-peak times in [ms] or [s]
        show : bool, optional
            If true, shows Poincaré plot (default: True)
        show : bool, optional
            If true, shows generated plot
        figsize : array, optional
            Matplotlib figure size (width, height) (default: (6, 6))
        ellipse : bool, optional
            If true, shows fitted ellipse in plot (default: True)
        vectors : bool, optional
            If true, shows SD1 and SD2 vectors in plot (default: True)
        legend : bool, optional
            If True, adds legend to the Poincaré plot (default: True)
        marker : character, optional
            NNI marker in plot (default: 'o')
        Returns (biosppy.utils.ReturnTuple Object)
        ------------------------------------------
        [key : format]
            Description.
        poincare_plot : matplotlib figure object
            Poincaré plot figure
        sd1 : float
            Standard deviation (SD1) of the major axis
        sd2 : float, key: 'sd2'
            Standard deviation (SD2) of the minor axis
        sd_ratio: float
            Ratio between SD2 and SD1 (SD2/SD1)
        ellipse_area : float
            Area of the fitted ellipse
        """
        # Check input values
        nn = pyhrv.utils.check_input(nni, rpeaks)

        # Prepare Poincaré data
        x1 = np.asarray(nn[:-1])
        x2 = np.asarray(nn[1:])

        # SD1 & SD2 Computation
        sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
        sd2 = np.std(np.add(x1, x2) / np.sqrt(2))

        # Area of ellipse
        area = np.pi * sd1 * sd2

        # Prepare figure
        if figsize is None:
            figsize = (6, 6)
        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        ax = fig.add_subplot(111)

        ax.set_title(r'$Poincar\acute{e}$')
        ax.set_ylabel('$NNI_{i+1}$ [ms]')
        ax.set_xlabel('$NNI_i$ [ms]')
        ax.set_xlim([np.min(nn) - 50, np.max(nn) + 50])
        ax.set_ylim([np.min(nn) - 50, np.max(nn) + 50])
        ax.grid()
        ax.plot(x1, x2, 'r%s' % marker, markersize=2, alpha=0.5, zorder=3)

        # Compute mean NNI (center of the Poincaré plot)
        nn_mean = np.mean(nn)

        # Draw poincaré ellipse
        if ellipse:
            ellipse_ = mpl.patches.Ellipse((nn_mean, nn_mean), sd1 * 2, sd2 * 2, angle=-45, fc='k', zorder=1)
            ax.add_artist(ellipse_)
            ellipse_ = mpl.patches.Ellipse((nn_mean, nn_mean), sd1 * 2 - 1, sd2 * 2 - 1, angle=-45, fc='lightyellow', zorder=1)
            ax.add_artist(ellipse_)

        # Add poincaré vectors (SD1 & SD2)
        if vectors:
            arrow_head_size = 3
            na = 4
            a1 = ax.arrow(
                nn_mean, nn_mean, (-sd1 + na) * np.cos(np.deg2rad(45)), (sd1 - na) * np.sin(np.deg2rad(45)),
                head_width=arrow_head_size, head_length=arrow_head_size, fc='g', ec='g', zorder=4, linewidth=1.5)
            a2 = ax.arrow(
                nn_mean, nn_mean, (sd2 - na) * np.cos(np.deg2rad(45)), (sd2 - na) * np.sin(np.deg2rad(45)),
                head_width=arrow_head_size, head_length=arrow_head_size, fc='b', ec='b', zorder=4, linewidth=1.5)
            a3 = mpl.patches.Patch(facecolor='white', alpha=0.0)
            a4 = mpl.patches.Patch(facecolor='white', alpha=0.0)
            ax.add_line(mpl.lines.Line2D(
                (min(nn), max(nn)),
                (min(nn), max(nn)),
                c='b', ls=':', alpha=0.6))
            ax.add_line(mpl.lines.Line2D(
                (nn_mean - sd1 * np.cos(np.deg2rad(45)) * na, nn_mean + sd1 * np.cos(np.deg2rad(45)) * na),
                (nn_mean + sd1 * np.sin(np.deg2rad(45)) * na, nn_mean - sd1 * np.sin(np.deg2rad(45)) * na),
                c='g', ls=':', alpha=0.6))

            # Add legend
            if legend:
                ax.legend(
                    [a1, a2, a3, a4],
                    ['SD1: %.3f$ms$' % sd1, 'SD2: %.3f$ms$' % sd2, 'S: %.3f$ms^2$' % area, 'SD1/SD2: %.3f' % (sd1/sd2)],
                    framealpha=1)

        # Show plot
        if show:
            plt.show()

        # Output
        args = (fig, sd1, sd2, sd2/sd1, area)
        names = ('poincare_plot', 'sd1', 'sd2', 'sd_ratio', 'ellipse_area')
        return biosppy.utils.ReturnTuple(args, names)
