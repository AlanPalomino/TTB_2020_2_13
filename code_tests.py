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
from TT import Case, Case2
from pathlib import Path
from memory_profiler import profile


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

# %%

# Get R-peaks series using biosppy
rpeaks = biosppy.signals.ecg.ecg(signal)[2]

# Compute Poincaré using R-peak series
results = nl.poincare(rpeaks=rpeaks)

# Show the scatter plot without the fitted ellipse, the SD1 & SD2 vectors and the legend
results = nl.poincare(nni, ellipse=False, vectors=False, legend=False)
