# %%
#_
#___________________________________________________________________________
#|                                                                         |
#| Playground para pruebas:                                                |
#|      De ser necesario probar código, este script puede ser usaado       |
#|      para ese propósito. El archivo se eliminará del repositorio        |
#|      una vez concluido el proyecto.                                     |
#|_________________________________________________________________________|

from TT_utilities import Case, CustomPlots
from pathlib import Path

from wfdb.processing.qrs import gqrs_detect
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
#from memory_profiler import profile
from matplotlib import gridspec
from scipy.stats import stats
from pprint import pprint 
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

import biosppy
import numpy as np
import pyhrv.tools as tools
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl

# %%

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
rr = np.array(AF_CASES.iloc[0,2])

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
rr = np.array(AF_CASES.iloc[0,2])


# Compute Poincaré using NNI series
#results = nl.poincare(rr,show=False,ellipse=False,vectors=False,legend=False)
results = CustomPlots.poincarePlot(rr,show=True,ellipse=False,vectors=False,legend=False)
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
