from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import time
import wfdb
import re

def timeit(func, *args, **kwargs):
    s_time = time.time()
    func(*args, **kwargs)
    e_time = time.time()
    print(f"Function {func.__name__} execution time: {e_time - s_time}")


class Case():
    """
        Object looks for files in the directory provided, to build a list
        of records from the same case 
    """
    RECORDS = []
    def __init__(self, case_dir: Path, sig_thresh: int=1000):
        self._case_dir = case_dir
        self._case_name = case_dir.stem
        self._sig_thresh = sig_thresh
        self._read_records()
    
    def __str__(self):
        print(f"Case: {self._case_name} - Records above {self._sig_thresh} samples ->")
        for record in self.RECORDS:
            print(record)
        return f"{5*' * '}End of case {self._case_name}{5*' * '}"
    
    def __getitem__(self, index):
        return self.RECORDS[index]
    
    def _read_records(self):
        for hea_path in self._case_dir.glob(f"{record_name}*[0-9].hea"):
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
