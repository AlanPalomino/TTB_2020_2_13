# -*- coding: utf-8 -*-
# %%

# ===================== Librerias Utilizadas ====================== #
from concurrent.futures import ThreadPoolExecutor
from biosppy.utils import ReturnTuple
from matplotlib import pyplot as plt
from itertools import combinations 
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.stats import stats
from collections import Counter
from functools import wraps
import pyhrv.nonlinear as nl
from wfdb import processing
from itertools import chain
from pathlib import Path
from hurst import compute_Hc
import seaborn as sns
import pandas as pd
import numpy as np
import entropy
import biosppy
import pyhrv
import time
import wfdb
import re

from hrvanalysis import (
    get_frequency_domain_features,
    get_poincare_plot_features,
    get_sampen,
    get_time_domain_features
)

# ================= Funciones y Definiciones ====================== # 


def timeit(func):
    def timed_func(*args, **kwargs):
        s_time = time.time()
        r = func(*args, **kwargs)
        e_time = time.time()
        print(f"Function {func.__name__} execution time: {e_time - s_time:.2f}'s")
        return
    return timed_func


def get_hurst(rr):
    H, _, _ = compute_Hc(rr)
    return H


def get_poincare_ratio(rr):
    return get_poincare_plot_features(rr)['ratio_sd2_sd1']


def get_sample_entropy(rr):
    return get_sampen(rr)["sampen"]


# ================= Importando Bases de Datos
class Case():
    """
    Generador del compendio de registros y señales para un caso particular.

    Object looks for files in the directory provided, to build a list
    of records from the same case.
    """

    def __init__(self, case_dir: Path, sig_thresh: int=1000):
        print(case_dir)
        self.RECORDS = []
        self.sig = []
        self.nl_sig = []
        self._case_dir = case_dir
        self._case_name = case_dir.stem
        self._sig_thresh = sig_thresh
        self._processed = False
        try:
            self.pathology = re.search(
                f"([a-z_]*)(_p)",
                str(case_dir.parts[-2])
            ).groups()[0]
        except AttributeError:
            print(case_dir.parts[-1])
            print(case_dir.parts[-2])
            raise AttributeError
        self._get_records()

    def __str__(self):
        """Prints data of self and internal records"""
        print(f"Case: {self._case_name} - Records above {self._sig_thresh} samples ->")
        for record in self.RECORDS:
            print(record)
        return f"{5*' * '}End of case {self._case_name}{5*' * '}"

    def __len__(self):
        """Returns the number of records contained in the case"""
        return len(self.RECORDS)

    def __iter__(self):
        return CaseIterator(self)

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
        try:
            r = Record(path, self._case_name)
            self.RECORDS.append(r)
            return
        except ValueError:
            return

    @timeit
    def _linear_analysis_c(self):
        for record in self.RECORDS:
            if record not in self.l_sig and record._linear_analysis_r(self._main_signal):
                self.l_sig.append(record)

    @timeit
    def _non_linear_analysis_c(self):
        orglen = len(self)
        temp = list()
        for record in self.RECORDS:
            if record.slen >= RR_WINDOW_THRESHOLD and self._main_signal in record.sig_names:
                valid = record._non_linear_analysis_r(self._main_signal)
                if valid:
                    temp.append(record)
        self.RECORDS = temp
        print(f' < PROCESSED CASE {self._case_name} - {len(self)} valid records kept, {orglen-len(self)} records dumped.')

    def process(self, mode: str="nonlinear"):
        """
        Runs analysis on records
        
        If Case hasnt been processed, based on the input mode, it runs linear or
        non linear analysis, this process also erases records which could not be
        used for analysis, effectively reducing the Case data, hence why it's
        recommended to run only once.
        """

        def run_all(d: dict):
            [v() for k, v in d.items() if k != "full"]
            return

        analysis_selector = {
            "linear": self._linear_analysis_c,
            "nonlinear": self._non_linear_analysis_c
        }

        if not self._processed:
            signals = [
                r.sig_names for r in self.RECORDS       # List of lists
                if r.slen >= RR_WINDOW_THRESHOLD        # as long as it's avobe threshold
            ]
            top_signals = Counter(chain.from_iterable(signals)).most_common()
            for sig, count in top_signals:
                if sig in ['APB', 'PLETH R', 'RESP']:
                    continue
                self._main_signal = sig
                print(f' > PROCESSING CASE {self._case_name} optimal signal is "{sig}" present in {len(signals)} of {len(self)} records.')
                break
            else:
                print(f"WARNING - Case {self._case_name} record's have no valid signal for processing.")
                return
            if mode == 'full':
                run_all(analysis_selector)
                self._processed = True
                return
            analysis_selector.get(mode)()
            self._processed = True
            return

        print("Case already processed, cant do it again.")
        return

    @timeit
    def _plot_nonlinear(self):
        titles = ["Entropía Aproximada", "Entropía Muestral", "HFD", "DFA"]
        keys = ["app_ent", "samp_ent", "hfd", "dfa"]
        fig, axs = plt.subplots(nrows=len(keys), ncols=1, figsize=(12, 15))
        fig.suptitle(f"Non Linear Analysis of case {self._case_name}")
        for k, t, a in zip(keys, titles, axs):
            local_max = 0
            for seg in [r.N_LINEAR[k] for r in self.nl_sig]:
                x = np.arange(len(seg)) + local_max
                a.plot(x, seg)
                local_max = np.max(x) + 1
            a.set_title(t)
            a.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.show()
    
    @timeit
    def _plot_linear(self):
        titles = ["Media", "Varianza", "Asimetría", "Curtosis"]
        keys = ["means", "var", "skewness", "kurtosis"]
        fig, axs = plt.subplots(nrows=len(keys), ncols=1, figsize=(12, 15))
        fig.suptitle(f"Linear Analysis of case {self._case_name}")
        for k, t, a in zip(keys, titles, axs):
            local_max = 0
            for seg in [r.N_LINEAR[k] for r in self.l_sig]:
                x = np.arange(len(seg)) + local_max
                a.plot(x, seg)
                local_max = np.max(x) + 1
            a.set_title(t)
            a.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.show()
        
    @timeit
    def plotProcess(self, mode="full"):
        if mode == "full":
            self._plot_nonlinear()
            self._plot_linear()
        elif mode == "nonlinear":
            self._plot_nonlinear()
        elif mode == "linear":
            self._plot_linear()
        return


class CaseIterator:
    """Iterator class for Case object"""
    def __init__(self, case):
        self._case = case
        self._index = 0

    def __next__(self):
        """Returns the next record from the Case object's list of records"""
        if self._index < len(self._case):
            self._index += 1
            return self._case.RECORDS[self._index-1]
        raise StopIteration


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
        self.slen = reco.sig_len
        self.n_sig = reco.n_sig
        self.sig_names = reco.sig_name
        self.units = reco.units
        self.rr = None

    def __str__(self):
        return f"\t Record: {self.name}, Length:{self.slen}, \t# of signals: {self.n_sig} -> {self.sig_names}"

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

    def _linear_analysis_r(self, signal: str):
        if self.rr is None:
            # get RR
            raw_signal = self[signal]
            self.rr = np.diff(get_peaks(raw_signal, self.fs))
            if len(self.rr) < 2048*3:
                return False
            self.hurst = get_hurst(self.rr)
        m, v, s, k = linearWindowing(self.rr, w_len=1024, over=0.95)
        self.LINEAR = {
            "means": m,
            "var": v,
            "skewness": s,
            "kurtosis": k
        }
        return True

    def _non_linear_analysis_r(self, signal: str):
        if self.rr is  None:
            # get RR
            raw_signal = self[signal]
            self.rr_int = np.diff(get_peaks(raw_signal, self.fs))
            self.rr = self.rr_int * (1/self.fs)
            if len(self.rr) < RR_WINDOW_THRESHOLD:
                print(f' > X Record {self.name} - Analysis not possible, rr too short.')
                return False
            # self.hurst = get_hurst(self.rr_int)

        # HRVANALYSIS SECTION
        # self.time_domain = get_time_domain_features(self.rr_temp)
        # self.freq_domain = get_frequency_domain_features(self.rr_temp)
        # self.samp_entropy = get_sampen(self.rr_temp)
        # END OF SECTION

        self.N_LINEAR = {
            m["tag"]: t for m, t in zip(NL_METHODS,
                                        nonLinearWindowing(self.rr))
        }
        print(f' > < Record {self.name} - Non linear analysis done.')
        return True

    def plot(self):
        fig, axs = plt.subplots(self.n_sig, 1)
        signals = self._get_signals()

        fig.suptitle(f"Record {self.name} of case {self.case}")
        for a, n, u, s in zip(axs, self.sig_names, self.units, list(zip(*signals))):
            a.set_ylabel(f"{n}/{u}")
            a.plot(s)
        axs[-1].set_xlabel("Samples")
        plt.show()


def get_peaks(raw_signal: np.ndarray, fs: int) -> np.ndarray:
    MAX_BPM = 220
    raw_peaks, _ = find_peaks(raw_signal, distance=int((60/MAX_BPM)/(1/fs)))
    med_peaks = processing.correct_peaks(raw_signal, raw_peaks, 30, 35, peak_dir='up')
    # print("med_peaks: ", med_peaks[:10])
    # print("med_peaks: ", med_peaks[:10])
    # print("med_peaks: ", med_peaks[:10])
    try:
        wel_peaks = processing.correct_peaks(raw_signal, med_peaks, 30, 35, peak_dir='up') if len(med_peaks) > 0 else raw_peaks
    except ValueError:
        return med_peaks[~np.isnan(med_peaks)]
    return wel_peaks[~np.isnan(wel_peaks)]


# ================= Ventaneo de señales


def linearWindowing(rr_signal: np.ndarray):
    """
    Evaluates rr with linear functions based on a rolling window.

    rr_signal   :: RR vector of time in seconds
    """
    means, var, skew, kurt = list(), list(), list(), list()

    for idx in range(0, len(rr_signal)-RR_WLEN, RR_STEP):
        window_slice = slice(idx, idx+RR_WLEN)
        ds = stats.describe(rr_signal[window_slice])
        means.append(ds[2])
        var.append(ds[3])
        skew.append(ds[4])
        kurt.append(ds[5])

    return means, var, skew, kurt


def nonLinearWindowing(rr_signal: np.ndarray):
    """
    Evaluates rr with non-linear functions based on a rolling window.

    rr_signal   :: RR vector of time in seconds
    """
    DATA_TABLES = [list() for M in NL_METHODS]

    for idx in range(0, len(rr_signal)-RR_WLEN, RR_STEP):
        window_slice = slice(idx, idx+RR_WLEN)
        rr_window = rr_signal[window_slice]
        with ThreadPoolExecutor() as exec:
            for t, m in zip(DATA_TABLES, NL_METHODS):
                t.append(exec.submit(
                    m['func'],
                    rr_window, **m['args']
                ).result())

    return DATA_TABLES


def poincare_ratio(rr_window=None, rpeaks=None):
    """
    This function just returns the SD ratio for data collection.
    """
    # Check input values
    nn = pyhrv.utils.check_input(rr_window, rpeaks)

    # Prepare Poincaré data
    x1 = np.asarray(nn[:-1])
    x2 = np.asarray(nn[1:])

    # SD1 & SD2 Computation
    sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
    sd2 = np.std(np.add(x1, x2) / np.sqrt(2))

    # Returns sd ratio
    return sd2/sd1


def poincarePlot(nni=None, rpeaks=None, show=True, figsize=None, ellipse=True, vectors=True, legend=True, marker='o'):
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

    # Show plot
    if show == True:

        # Area of ellipse
        area = np.pi * sd1 * sd2

        # Prepare figure
        if figsize is None:
            figsize = (6, 6)
            fig = plt.figure(figsize=figsize)
            fig.tight_layout()
            ax = fig.add_subplot(111)

            ax.set_title(r'Diagrama de $Poincar\acute{e}$')
            ax.set_ylabel('$RR_{i+1}$ [ms]')
            ax.set_xlabel('$RR_i$ [ms]')
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
            a3 = plt.patches.Patch(facecolor='white', alpha=0.0)
            a4 = plt.patches.Patch(facecolor='white', alpha=0.0)
            ax.add_line(plt.lines.Line2D(
                (min(nn), max(nn)),
                (min(nn), max(nn)),
                c='b', ls=':', alpha=0.6))
            ax.add_line(plt.lines.Line2D(
                (nn_mean - sd1 * np.cos(np.deg2rad(45)) * na, nn_mean + sd1 * np.cos(np.deg2rad(45)) * na),
                (nn_mean + sd1 * np.sin(np.deg2rad(45)) * na, nn_mean - sd1 * np.sin(np.deg2rad(45)) * na),
                c='g', ls=':', alpha=0.6))

            # Add legend
            if legend:
                ax.legend(
                    [a1, a2, a3, a4],
                    ['SD1: %.3f$ms$' % sd1, 'SD2: %.3f$ms$' % sd2, 'S: %.3f$ms^2$' % area, 'SD1/SD2: %.3f' % (sd1/sd2)],
                    framealpha=1)

        plt.show()
            # Output
        args = (fig, sd1, sd2, sd2/sd1, area)
        names = ('poincare_plot', 'sd1', 'sd2', 'sd_ratio', 'ellipse_area')

    elif show == False:
        # Output
        args = (sd1, sd2, sd2/sd1, area)
        names = ('sd1', 'sd2', 'sd_ratio', 'ellipse_area')
        #result = biosppy.utils.ReturnTuple(args, names)

        
    return biosppy.utils.ReturnTuple(args, names)


def Poincare_Windowing(rr_signal, plotter=False):
    """
    rr_signal :: RR vector of time in seconds
    plotter   :: Boolean to plot the Poincare output or not.
    """

    poin_r = list()

    for idx in range(0, len(rr_signal)-RR_WLEN, RR_STEP):
        window_slice = slice(idx, idx+RR_WLEN)
        rr_window = rr_signal[window_slice]
        if plotter == True:
            poincare_results = nl.poincare(rr_window,show=True,figsize=None,ellipse=True,vectors=True,legend=True)
            poin_r.append(poincare_results["sd_ratio"])
        elif plotter == False:
            poincare_results = poincarePlot(rr_window,show=False)
            poin_r.append(poincare_results["sd_ratio"])

    return poin_r


def add_moments(row: pd.Series):
    """Applies five moments to Series object"""
    means, var, skew, kurt = linearWindowing(row.rr)

    row["M1"] = means
    row["M2"] = var
    row["M3"] = skew
    row["M4"] = kurt
    row["CV"] = np.divide(var, means)
    return row


def add_nonlinear(row: pd.Series):
    """Applies four non-linear equations to Series object"""
    app_ent, samp_ent, hfd, dfa, poin = nonLinearWindowing(row.rr)
    # poin = Poincare_Windowing(row.rr, plotter=False)

    row["AppEn"] = app_ent
    row["SampEn"] = samp_ent
    row["HFD"] = hfd
    row["DFA"] = dfa
    row["SD_ratio"] = poin
    return row


def distribution_cases(db, caso):
    caso = str(caso)
    moment =['M1','M2','M3','M4','CV']
    m_label =['Media','Varianza','Skewsness','Curtosis','Coeficiente de Variación ']
    for idx in range(len(moment)):
            
        title = 'Distribución de ' + m_label[idx] +' en Casos de ' + caso
        xlab = 'Valor de '+ m_label[idx]
        plt.figure(figsize=(10,7), dpi= 100)
        plt.gca().set(title=title, ylabel='Frecuencia',xlabel=xlab)
        for i in range(len(db.index)):

            ms = db.iloc[i][moment[idx]]
                
            lab = db.iloc[i]['record']
            # Plot Settings
            kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
            sns.distplot(ms, label= lab ,rug=False, hist=False,**kwargs)
                
            #X_axis limits
            #x_min = int(np.min(ms)) + 10
            #x_max = int(np.max(ms))+10
            #plt.xlim(x_min,x_max)
            #lims = plt.gca().get_xlim()
            #i = np.where( (ms > lims[0]) &  (ms < lims[1]) )[0]
            #plt.gca().set_xlim( ms[i].min(), ms[i].max() )
            plt.autoscale(enable=True, axis='y', tight=True)
        #show()
        plt.autoscale()
        plt.legend()


def get_all_stats(data, measure):
    """
    DESCRIPCIÓN ESTADISTICA DE TODOS LOS DATOS EN measure
    """
    SERIES = list()
    for condition in data["conditon"].unique():
        CASES = data[(data["conditon"] == condition) & (data["length"] > 5000)]
        if len(CASES) == 0:
            continue
        SERIES.append(CASES[measure].apply(pd.Series).stack().describe().to_frame(name=condition))
    return pd.concat(SERIES, axis=1).round(5)


def distribution_NL(db, caso, area=False):
    caso = str(caso)
    moment =['AppEn','SampEn','HFD','DFA','SD_ratio']
    m_label =['Ent_Aprox','Ent_Muestra','Higuchi','DFA','R= SD1/SD2']
    path = '/imagenes/'
    for idx in range(len(moment)):
            
        title = 'Distribución de ' + m_label[idx] +' en Casos de ' + caso
        #figname = m_label[idx]+"_"+caso+'.png'
        xlab = 'Valor de '+ m_label[idx]
        plt.figure(figsize=(10,7), dpi= 100)
        plt.gca().set(title=title, ylabel='Coeficiente',xlabel=xlab)
        for i in range(len(db.index)):

            ms = db.iloc[i][moment[idx]]
            #x_min =np.min(ms,axis=0)
            #x_max =np.max(ms,axis=0)
                
            lab = db.iloc[i]['record']
            # Plot Settings
            kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
            if area == True:
                sns.distplot(ms, label= lab ,rug=False, hist=True,**kwargs)
                
                #X_axis limits
                #x_min = int(np.min(ms)) + 10
                #x_max = int(np.max(ms))+10
                #plt.xlim(x_min,x_max)
                #lims = plt.gca().get_xlim()
                #i = np.where( (ms > lims[0]) &  (ms < lims[1]) )[0]
                #plt.gca().set_xlim( ms[i].min(), ms[i].max() )
                plt.autoscale(enable=True, axis='y', tight=True)
            else:
                sns.distplot(ms, label= lab ,rug=False, hist=False,**kwargs)
                    
                #X_axis limits
                #x_min = int(np.min(ms)) + 10
                #x_max = int(np.max(ms))+10
                #plt.xlim(x_min,x_max)
                #lims = plt.gca().get_xlim()
                #i = np.where( (ms > lims[0]) &  (ms < lims[1]) )[0]
                #plt.gca().set_xlim( ms[i].min(), ms[i].max() )
                plt.autoscale(enable=True, axis='y', tight=True)
            
        #show()
        plt.autoscale()
        plt.legend()
        #plt.savefig(path + figname )


def get_allNL_stats(data, measure):
    """
    DESCRIPCIÓN ESTADISTICA DE TODOS LOS DATOS EN measure
    """
    SERIES = list()
    for condition in data["conditon"].unique():
        CASES = data[(data["conditon"] == condition) & (data["length"] > 5000)]
        if len(CASES) == 0:
            continue
        SERIES.append(CASES[measure].apply(pd.Series).stack().describe().to_frame(name=condition))
    return pd.concat(SERIES, axis=1).round(5)


def get_max(DF, col):
    return np.max([np.max(DF[col][i]) for i in DF.index if len(DF[col][i]) > 0])


def get_min(DF, col):
    return np.min([np.min(DF[col][i]) for i in DF.index if len(DF[col][i]) > 0])


def plot_NL_metrics(DataBases, techniques, conditions, columns):
    """
    docstring
    """
    for idx, title, col in zip([1, 2, 3, 4, 5], techniques, columns):
        figure, axs = plt.subplots(3, 1, figsize=(8, 10))
        figure.suptitle(title, y=1.01)
        top = np.max([get_max(item, col) for item in DataBases])
        bot = np.min([get_min(item, col) for item in DataBases])
        
        axs[0].set_title(conditions[0])
        for i in range(len(DataBases[0])):
            axs[0].plot(DataBases[0].iloc[i][col])
        axs[0].autoscale(enable=True, axis='x', tight=True)
        axs[0].set_ylim(bottom=bot, top=top)

        axs[1].set_title(conditions[1])
        for i in range(len(DataBases[1])):
            axs[1].plot(DataBases[1].iloc[i][col])
        axs[1].autoscale(enable=True, axis='x', tight=True)
        axs[1].set_ylim(bottom=bot, top=top)

        axs[2].set_title(conditions[2])
        for i in range(len(DataBases[2])):
            axs[2].plot(DataBases[2].iloc[i][col])
        axs[2].autoscale(enable=True, axis='x', tight=True)
        axs[2].set_ylim(bottom=bot, top=top)

        axs[-1].set_xlabel(f"Figura {idx}")
        plt.tight_layout()
        plt.show()



def KS_Testing(Databases, conditions):
    """
    docstring
    """
    
    columns = ["AppEn", "SampEn", "DFA", "HFD","SD_ratio"]
    ks_test=list()
        

    for Data in Databases:
        for cond in conditions:
            #print(Data)
            print("Base de datos: ", cond)
            for col in columns:
                metric = np.array(Data[[col]])
                print("Métrica: ",col)
                #print(type(metric))
                comb = list(combinations(metric, 2))
                #print("Combinaciones posibles: ",len(comb))   
                
                for i in range(len(comb)-1):
                    pair = comb[i]

                    X = np.histogram(np.array(pair[0]).all(), bins='auto')
                    Y = np.histogram(np.array(pair[1]).all(), bins='auto')
                    ks_r = stats.ks_2samp(X[0], Y[0], alternative='two-sided')
                    p_val = ks_r[1]
                    #print(p_val)
                    if p_val < 0.05:
                        ks_test.append(0)
                    elif p_val > 0.05:
                        ks_test.append(1)
                    prob = np.sum(ks_test)/len(ks_test)*100
                print("Porcentaje de Similitud {} %" .format(prob)) 
            print("\n")


def RunAnalysis():
    #ks_test = stats.kstest()
    pass



# ====================== Global Values =========================== #

RR_WLEN = 250
RR_OVER = 0.5
RR_STEP = int(RR_WLEN * (1 - RR_OVER))
RR_WINDOW_THRESHOLD = RR_WLEN * 6   # Mínimo número de datos que requiere un registro rr para ser válido.


NL_METHODS = [
    {
        "name": "Approxiamte Entropy",
        "tag": "ae",
        "func": entropy.app_entropy,
        "args": dict(order=2, metric='chebyshev')
    },{
        "name": "Sample Entropy",
        "tag": "se",
        "func": get_sample_entropy,
        "args": dict()
    },{
        "name": "Higuchi Fractal Dimension",
        "tag": "hfd",
        "func": entropy.fractal.higuchi_fd,
        "args": dict(kmax=10)
    },{
        "name": "Detrended Fluctuation Analysis",
        "tag": 'dfa',
        "func": entropy.fractal.detrended_fluctuation,
        "args": dict()
    },{
        "name": "Poincaré SD Ratio",
        "tag": "psd",
        "func": get_poincare_ratio,
        "args": dict()
    },{
        "name": "Hurst",
        "tag": "hst",
        "func": get_hurst,
        "args": dict()
    }
]

