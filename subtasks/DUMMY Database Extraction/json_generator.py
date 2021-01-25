#!usr/bin/env python3
# _*_ coding: utf-8 _*_ #
#
#___________________________________________________________________________
#|                                                                         |
#|      Generación de archivos JSON de base de datos DUMMY                 |
#|                                                                         |
#|                                                                         |
#|_________________________________________________________________________|


# ===================== Librerias Utilizadas ====================== #

from scipy.signal import find_peaks, iirnotch, filtfilt, sosfiltfilt
from IPython.display import clear_output
from scipy.misc import electrocardiogram
from matplotlib import pyplot as plt
from pprint import pprint
import pandas as pd
import numpy as np
import wfdb
import json
import sys
import os


class Unapproved(Exception):
    pass

class Approved(Exception):
    pass


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def smoother(array, window=5, pad=True, repeat=1):
    """
    Regresa un vector de la misma longitud que el introducido, dependiendo
    del valor de la ventana ajusta los extremos al promedio de los valores
    que puede ocupar.
    
    ó, si pad = False
    Regresa sólo el vector promediado con los (window-1) datos en los ext-
    tremos acortados.
    
    VALOR MÍNIMO DE VENTANA ES 3
    """
    for i in range(repeat):
        serie = pd.Series(array)
        data = list(serie.rolling(window=window).mean())
        numb = sum(np.isnan(data)*1)
        if not pad:
            return data[numb:]
        lower, upper = int(np.ceil(numb/2)), int(np.floor(numb/2))
        ini = [np.mean(array[:i+upper]) for i in range(lower)]
        end = [np.mean(array[(len(array)-(lower+i)):]) for i in range(upper)]
        array = np.array(ini+data[numb:]+end)
    return array


def get_records(database):
    with open(f"{database}/RECORDS", "r") as file:
        records = [line.strip() for line in file.readlines()]
        try:
            with open(f"{database}.json", "r") as js:
                register = json.load(js)
                registered = [obj["record"] for obj in register]
                return [record for record in records if record not in registered], register
        except FileNotFoundError:
            return records, list()


def peak_detection(signal, fs, mode="pos"):
    if mode == "pos":
        "Registro de picos superiores"
        peaks = find_peaks(signal, distance=int((60/MAX_BPM)/(1/fs)), prominence=0.8)[0]
        if (len(signal)*(1/fs)/60)*MIN_BPM > len(peaks):
            return find_peaks(signal, distance=int((60/MAX_BPM)/(1/fs)), height=max(signal)*0.4)[0]
        return peaks
    elif mode == "neg":
        "Registro de picos inferiores"
        n_peaks = find_peaks(signal*(-1), distance=int((60/MAX_BPM)/(1/fs)), prominence=0.8)[0]
        if (len(signal)*(1/fs)/60)*MIN_BPM > len(n_peaks):
            return find_peaks(signal*(-1), distance=int((60/MAX_BPM)/(1/fs)), height=max(signal)*0.4)[0]
        return n_peaks


def display_signals(signals, fs, sig_name):
    """
        Despliega las señales con sus picos encontrados para seleccionar la óptima
    """
    fig, axs = plt.subplots(signals.shape[1], 1, figsize=(30, 10), num=1)
    b, a = iirnotch(w0=60, Q=1, fs=fs)
    for idx in range(signals.shape[1]):
        # Selección de 60 segundos de la señal
        signal = signals[:int(60/(1/fs)), idx]
        # Filtro Notch de 60Hz
        signal = filtfilt(b, a, signal)
        # Restado de la media de la señal
        sig_prom = smoother(signal, window=int(((60*fs)/MAX_BPM)))
        sig_prom = smoother(sig_prom, window=int(((60*fs)/MAX_BPM)*2), repeat=5)
        signal = signal - sig_prom
        # Obtención de picos inferiores y superiores
        peaks = np.array(peak_detection(signal, fs))
        n_peaks = np.array(peak_detection(signal, fs, "neg"))
        time_axis = np.array(range(len(signal)))*(1/fs)
        
        # Ploteo de todo en la figura
        axs[idx].set_title(f"Index: {idx} - {sig_name[idx]}")
        axs[idx].plot(time_axis, signal, "g")
        axs[idx].plot(time_axis, sig_prom, "b")
        axs[idx].plot(peaks*(1/fs), signal[peaks], "*r")
        axs[idx].plot(n_peaks*(1/fs), signal[n_peaks], "*m")
    plt.show()


def wfdb_process(record):
    record = wfdb.rdrecord(record)
    annot = wfdb.rdheader(record)


def database_cycler(database, fixed_cond=None):
    records, register = get_records(database)
    for record in records:
        print(f"\t{database}/{record}")
        try:
            # Obtención del registro
            try:
                signals, fields = wfdb.rdsamp(f"{database}/{record}")
            except ValueError:
                raise Unapproved
            # WFDB ANNOTATIONS SECTION
            try:
                annot = wfdb.rdann(database+"/"+record, extension="atr")
                plt.figure(1)
                plt.plot(signals[:, 0])
                plt.plot(annot.sample, signals[:, 0][annot.sample], '*r')
                plt.show()
                
                if input("Keep Original Annotations? [Y/N]:").lower() == "y":
                    RR = np.diff(annot.sample)
                    if not fixed_cond:
                        condition = input("Enter patient conditon: ")
                    raise Approved
                plt.clf()
            except FileNotFoundError:
                print("Original Annotations Unnavailable")

            display_signals(signals, fields["fs"], fields["sig_name"])
            plt.show()
            print("fs: ", fields["fs"], "\ncomments: ", fields["comments"])
            
            selection = input("Save Data? [Y/N] (E to interrupt): ").lower()
            if selection == "n":
                raise Unapproved
            elif selection == "e":
                raise KeyboardInterrupt
            
            sig_idx = int(input("Select signal index to extract peaks from: "))
            peak_mode = input("Select Upper(U) or Lower(L) peaks to extract: ").lower()
            if not fixed_cond:
                condition = input("Enter patient conditon: ")

            b, a = iirnotch(w0=60, Q=1, fs=fields["fs"])
            signal = filtfilt(b, a, signals[:, sig_idx])
            signal = signal - smoother(smoother(signal, window=int(((60*fields["fs"])/MAX_BPM))), window=int(((60*fields["fs"])/MAX_BPM)*2), repeat=5)
            if peak_mode == "u":
                RR = np.diff(peak_detection(signal, fields["fs"], "pos"))
            elif peak_mode == "l":
                RR = np.diff(peak_detection(signal, fields["fs"], "neg"))
            raise Approved

        except Approved:
            data = dict(
                database=database,
                record=record,
                rr = RR,
                comments=fields["comments"],
                fs=fields["fs"],
                approved=True,
            )
            if fixed_cond:
                data["conditon"] = fixed_cond
            else:
                data["conditon"] = condition
            register.append(data)
        except (Unapproved) as r:
            print(record, r)
            register.append(dict(
                database=database,
                record=record,
                approved=False,
            ))
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            with open(f"{database}.json", "w+") as file:
                json.dump(register, file, cls=NpEncoder)
            print("TASK INTERRUPTED")
            sys.exit()
        finally:
            plt.clf()
            clear_output(wait=True)
            with open(f"{database}.json", "w+") as file:
                json.dump(register, file, cls=NpEncoder)

    with open(f"{database}.json", "w+") as file:
        json.dump(register, file, cls=NpEncoder)
    print("TASK DONE")

# In[]
if __name__ == "__main__":
    plt.ion()
    MAX_BPM = 220
    MIN_BPM = 40
    databases = [
        'afdb-1.0.0.physionet.org',
        'chfdb-1.0.0.physionet.org',
        'ltafdb-1.0.0.physionet.org',
        'mitdb-1.0.0.physionet.org',
        'ptbdb-1.0.0.physionet.org',
    ]
    database_cycler('afdb-1.0.0.physionet.org', fixed_cond="AF")
    database_cycler('chfdb-1.0.0.physionet.org', fixed_cond="CHF")
    database_cycler("mitdb-1.0.0.physionet.org", fixed_cond="AR")
    # database_cycler("ltafdb-1.0.0.physionet.org", fixed_cond="AF")
    database_cycler("ptbdb-1.0.0.physionet.org")
