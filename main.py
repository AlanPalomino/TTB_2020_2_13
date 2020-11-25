# %%
#_
#___________________________________________________________________________
#|                                                                         |
#|    TTB__2020_1_13 Main code:                                            |
#|      Código principal para el trabajo                                   |
#|                                                                         |
#|                                                                         |
#|_________________________________________________________________________|

# %%
# ===================== Librerias Utilizadas ====================== #

from wfdb.processing.qrs import gqrs_detect
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
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

# ===================== Funciones y Métodos ======================= #
from TT_utilities import Case


# ================================================================= #
# %%
# Importando BD de prueba (Data_Jsons)
""" 
Se genera un dataframe con todos los datos válidos de todas las bases de datos.
"""

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

#   MIMIC 3 DATA LOAD
RECORD_DIRS = list(Path("./Data").glob("*p00*"))
for record_dir in RECORD_DIRS:
    record_name = re.search("p[0-9]{6}", str(record_dir))[0]
    case = Case(record_dir.joinpath(record_name))
    break
# %%
