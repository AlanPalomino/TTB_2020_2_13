# %%
#_
#___________________________________________________________________________
#|                                                                         |
#| Playground para pruebas:                                                |
#|      De ser necesario probar código, este script puede ser usaado       |
#|      para ese propósito. El archivo se eliminará del repositorio        |
#|      una vez concluido el proyecto.                                     |
#|_________________________________________________________________________|

from TT_utilities import Case
from pathlib import Path

from wfdb.processing.qrs import gqrs_detect
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
#from memory_profiler import profile
from matplotlib import gridspec
from scipy.stats import stats
from pprint import pprint 
import seaborn as sns
import entropy as tpy
import pandas as pd
import numpy as np
import biosppy
import decimal
import json
import wfdb
import ast
import os
import re


# %%
@profile
def Load():
    RECORD_DIRS = list(Path("./Data").glob("*p00*"))
    for record_dir in RECORD_DIRS:
        record_name = re.search("p[0-9]{6}", str(record_dir))[0]
        case = Case(record_dir.joinpath(record_name))
        case2 = Case2(record_dir.joinpath(record_name))
        break

Load()

# %%
#================= pyHRV Testing ================================


# %%
RECORD_DIRS = list(Path("./Data").glob("*p00*"))
CASES = list()
for record_dir in RECORD_DIRS:
    record_name = re.search("p[0-9]{6}", str(record_dir))[0]
    c = Case(record_dir.joinpath(record_name))
    CASES.append(c)
    print(c)

data = list()
for data_file in os.listdir("./Data_Jsons"):
    with open("./Data_Jsons/"+data_file) as file:
        mixed = json.load(file)
        appr = [reg for reg in mixed if reg["approved"]]
        data.extend(appr)
        print(f"{data_file} has {len(appr)}/{len(mixed)} approved cases")
data = pd.DataFrame(data)
data["rr"] = data.apply(lambda case: np.array(case["rr"])/case["fs"], axis=1)
data["rr"] = data["rr"].apply(lambda signal: signal[np.where(signal < 2)])
data["length"] = data["rr"].apply(lambda signal: len(signal))

print("Seleccion de casos aprobados...")
num_cases = 15
# AF - Atrial Fibrilation
AF_CASES = data[(data["conditon"] == "AF") & (data["length"] > 1000)][:num_cases]
# CHF - Congestive Heart Failure
CHF_CASES = data[(data["conditon"] == "CHF") & (data["length"] > 1000)][:num_cases]
# HC - Healthy Controls
HC_CASES = data[(data["conditon"] == "HC") & (data["length"] > 1000)][:num_cases]
# AR - Arrhythmia Cases
AR_CASES = data[(data["conditon"] == "AR") & (data["length"] > 1000)][:num_cases]   # NO HAY CASOS QUE CUMPLAN 
# MI - Myocardial Infarction
MI_CASES = data[(data["conditon"] == "MI") & (data["length"] > 1000)][:num_cases]   # NO HAY CASOS QUE CUMPLAN

print(f"""
AF CASES: {len(AF_CASES)}
CHF CASES: {len(CHF_CASES)}
HC CASES: {len(HC_CASES)}
AR CASES: {len(AR_CASES)}
MI CASES: {len(MI_CASES)}
""")
# %%
# Get R-peaks series using biosppy
rr = np.array(HC_CASES.iloc[0,2])

# Compute Poincaré using R-peak series
results = nl.poincare(rpeaks=rr)

# Show the scatter plot without the fitted ellipse, the SD1 & SD2 vectors and the legend
results = nl.poincare(rr, ellipse=False, vectors=False, legend=False)
# %%
# Import packages
import pyhrv
import pyhrv.nonlinear as nl

# Load sample data
#nni = pyhrv.utils.load_sample_nni()
#rr = np.array(AF_CASES.iloc[0,2])
rr = np.array(HC_CASES.iloc[0,2])

# Compute Poincaré using NNI series
results = nl.poincare(rr,show=True,ellipse=True,vectors=True,legend=True)
#results = Windowing.poincarePlot(rr,show=False,ellipse=False,vectors=False,legend=False)
# Print SD1
print(results)
# %%
# Import packages
import biosppy
import pyhrv.time_domain as td
from opensignalsreader import OpenSignalsReader

# Load sample ECG signal stored in an OpenSignals file
#acq = OpenSignalsReader('SampleECG.txt')
# signal = OpenSignalsReader('SampleECG.txt').signal('ECG')

# Get R-peaks series using biosppy
#rpeaks = biosppy.signals.ecg.ecg(signal)[2]

# Compute Poincaré using R-peak series
results = nl.poincare(rpeaks=rr)
# %%
# PRUEBAS DE MÉTRICAS NO LINEALES PARA UNA SEÑAL.

import pyhrv.nonlinear as nl
from entropy import *
import pyhrv

rr = np.array(AF_CASES.iloc[0,2])
# %%
# Combinaciones

from itertools import combinations 

dists = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O"]

comb = list(combinations(dists, 2))
for i in range(len(comb)):
    pair = comb[i]
    print("Combina {}  con {}." .format(pair[0],pair[1]))
# %%
from main import MainDF, MainDummy
# %%
import theano
from theano import tensor
a = tensor.dscalar()
b = tensor.dscalar()
c = a + b
f = theano.function([a,b], c)
d = f(1.5, 2.5)
print (d)
# %%
MainDF = MainDF.sample(frac=1.0)
MainDF.shape

Datmat = MainDF.copy()
del Datmat['case']
Datmat.dropna()

# TRansformations
Datmat['cond'].astype(np.int32)
Datmat['ae_m'].astype('float32')
Datmat.info()
# %%
import umap
import umap.plot
from sklearn.datasets import load_digits

digits = load_digits()

mapper = umap.UMAP().fit(digits.data)
umap.plot.points(mapper, labels=digits.target)
# %%
digits_df = pd.DataFrame(digits.data[:,1:11])
digits_df['digit'] = pd.Series(digits.target).map(lambda x: 'Digit {}'.format(x))
sns.pairplot(digits_df, hue='digit', palette='Spectral')
# %%
import tensorflow as tf
from tensorflow import keras
#convert the pandas object to a tensor

MainDummy.info()
MainDummy.values()
#data=tf.convert_to_tensor(MainDummy)
#type(data)
# %%
mapper = umap.UMAP().fit(Datmat)
#umap.plot.points(mapper, labels=digits.target)
# %%
