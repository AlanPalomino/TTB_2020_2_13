# %%
#_
#___________________________________________________________________________
#|                                                                         |
#| Playground para pruebas:                                                |
#|      De ser necesario probar código, este script puede ser usaado       |
#|      para ese propósito. El archivo se eliminará del repositorio        |
#|      una vez concluido el proyecto.                                     |
#|_________________________________________________________________________|

import re
from TT_utilities import Case
from pathlib import Path

from wfdb.processing.qrs import gqrs_detect
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
#from memory_profiler import profile
from matplotlib import gridspec
from scipy.stats import stats

from pprint import pprint 
import entropy as tpy
import pandas as pd
import numpy as np
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
results = nl.poincare(rr)

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

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

figsize = (6, 6)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
fig = plt.figure(figsize=figsize)
fig.tight_layout()
ax = fig.add_subplot(111)

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    
    ax.set_title(r'$Poincar\acute{e}$')
    ax.set_ylabel('$NNI_{i+1}$ [ms]')
    ax.set_xlabel('$NNI_i$ [ms]')
    ax.scatter(xs, ys, zs, marker=m)

plt.show()

#from images2gif import writeGif
# %%
# Simulación del diagrama de bifurcación de un modelo logístico


import pynamical
from pynamical import simulate, phase_diagram_3d
import pandas as pd, numpy as np, matplotlib.pyplot as plt, random, glob, os, IPython.display as IPdisplay
from PIL import Image
%matplotlib inline
from pynamical import logistic_map, simulate, bifurcation_plot

pops = simulate(model=logistic_map, num_gens=100, rate_min=0, rate_max=4, num_rates=1000, num_discard=100)
bifurcation_plot(pops)
# %%


import pynamical
from pynamical import simulate, save_fig, phase_diagram, phase_diagram_3d
import pandas as pd, numpy as np, matplotlib.pyplot as plt, IPython.display as IPdisplay
%matplotlib inline



title_font = pynamical.get_title_font()
label_font = pynamical.get_label_font()


# sometimes it is hard to tell if a time series is chaotic or random
# generate two time series of 1,000 steps, one chaotic and one random
# generate 30,000 time steps for the chaotic series but only keep the final 1,000 (when system is fully evolved)

gens = 1000
np.random.seed(1)



#chaos_pops = chaos_pops.iloc[total_gens-gens:].reset_index().drop(labels='index', axis=1)

AF_CASES.head()
rr  = pd.Series(AF_CASES.iloc[0,2])

total_gens = len(rr)
logi = simulate(num_gens=total_gens, rate_min=3.99, num_rates=1)

#rr_inter = AF_CASES.iloc[0,2]
#random_pops = pd.DataFrame(np.random.random(gens), columns=['value'])
time_series = pd.concat([logi, rr], axis=1)
time_series.columns = ['logistic', 'rr']
#time_series.head()


# %%


# plot the chaotic and random time series to show how they are sometimes tough to differentiate
ax = time_series.plot(kind='line', figsize=[10, 6], linewidth=3, alpha=0.6, style=['#003399','#cc0000'])
ax.grid(True)
ax.set_xlim(40, 100)
#ax.set_ylim(0, 1)
ax.set_title('Deterministic Chaos vs HRV', fontproperties=title_font)
ax.set_xlabel('Time', fontproperties=label_font)
ax.set_ylabel('Frec', fontproperties=label_font)
ax.legend(loc=3)

#save_fig('chaos-vs-random-line')
plt.show()

# %%


# plot same data as 2D phase diagram instead
pops = pd.concat([logi, rr], axis=1)
pops.columns = ['logistic', 'rr']
phase_diagram(pops, size=20, color=['#003399','#cc0000'], ymax=1.005, legend=True)

# %%


# plot same data as 3D phase diagram instead
phase_diagram_3d(pops, color=['#003399','#cc0000'],
                 legend=True, legend_bbox_to_anchor=(0.94, 0.9))


# %%
