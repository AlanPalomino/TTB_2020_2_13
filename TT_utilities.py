# -*- coding: utf-8 -*-
# ===================== Librerias Utilizadas ====================== #
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
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
        self._read_records()

    def __str__(self):
        "Prints "
        print(f"Case: {self._case_name} - Records above {self._sig_thresh} samples ->")
        for record in self.RECORDS:
            print(record)
        return f"{5*' * '}End of case {self._case_name}{5*' * '}"

    def __getitem__(self, index):
        return self.RECORDS[index]

    def _read_records(self):
        for hea_path in self._case_dir.glob(f"{self._case_name}*[0-9].hea"):
            h = wfdb.rdheader(str(hea_path.parent.joinpath(hea_path.stem)))
            self._get_names(h.seg_name, h.seg_len)

    def _get_names(self, seg_names: list, seg_lens: list):
        for name, slen in zip(seg_names, seg_lens):
            if slen < self._sig_thresh or "~" in name:
                continue
            self._get_data(self._case_dir.joinpath(name))

    def _get_data(self, path: Path):
        record = wfdb.rdrecord(str(path))
        header = wfdb.rdheader(str(path))
        self.RECORDS.append(
            Record(header, record, self._case_name)
        )


class Record():

    def __init__(self, head, record, case):
        self.name = head.record_name
        self.case = case
        self.time = head.base_time
        self.date = head.base_date
        self.fs = record.fs
        self.len = record.sig_len
        self.n_sig = record.n_sig
        self.signals = record.p_signal
        self.sig_names = record.sig_name
        self.units = record.units

    def __str__(self):
        return f"\t Record: {self.name}, Length:{self.len}, \t# of signals: {self.n_sig} -> {self.sig_names}"

    def __getitem__(self, item):
        try:
            sig_idx = self.sig_names.index(item)
            return self.signals[:, sig_idx]
        except ValueError:
            raise KeyError(f"'{item}' isn't a valid key. Signals in record:{self.sig_names}")

    def plot(self):
        fig, axs = plt.subplots(self.n_sig, 1)
        fig.suptitle(f"Record {self.name} of case {self.case}")
        for a, n, u, s in zip(axs, self.sig_names, self.units, list(zip(*self.signals))):
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
