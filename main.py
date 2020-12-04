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
from TT_utilities import *
from TT_utilities import add_moments,add_nonlinear

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
#======== Agregando las métricas obtenidas a las Bases 
print("ACTUALIZANDO DATABASES...")
#AF_CASES = AF_CASES.apply(add_moments, axis=1)
#CHF_CASES = CHF_CASES.apply(add_moments, axis=1)
#HC_CASES = HC_CASES.apply(add_moments, axis=1)

AF_CASES_NL = AF_CASES.copy()
CHF_CASES_NL = CHF_CASES.copy()
HC_CASES_NL = HC_CASES.copy()
print("Métricas agregadas:  ")

print(" - ".join(AF_CASES.columns))
print(" - ".join(CHF_CASES.columns))
print(" - ".join(HC_CASES.columns))

print("ACTUALIZANDO DATABASES...")
AF_CASES_NL = AF_CASES_NL.apply(add_nonlinear, axis=1)
CHF_CASES_NL = CHF_CASES_NL.apply(add_nonlinear, axis=1)
HC_CASES_NL = HC_CASES_NL.apply(add_nonlinear, axis=1)

print("Métricas agregadas:  ")
print(" - ".join(AF_CASES_NL.columns))
print(" - ".join(CHF_CASES_NL.columns))
print(" - ".join(HC_CASES_NL.columns))

# %%
AF_CASES_NL.head()
# %%

def get_max(DF, col):
    return np.max([np.max(DF[col][i]) for i in DF.index if len(DF[col][i]) > 0])

def get_min(DF, col):
    return np.min([np.min(DF[col][i]) for i in DF.index if len(DF[col][i]) > 0])

conditions = ["Fibrilación Atrial", "Insuficiencia Cardíaca Congestiva", "Casos Saludables"]
techniques = ["Entropía aproximada", "Entropía muestral", "Analisis de Fluctuación sin Tendencia (DFA)", "Coeficiente de Higuchi (HFD)","Radio = SD1/SD2"]
columns = ["AppEn", "SampEn", "DFA", "HFD","SD_ratio"]
cases = [AF_CASES_NL, CHF_CASES_NL, HC_CASES_NL]

def plot_NL_metrics(DataBases, techniques, conditions, columns):
    """
    docstring
    """
    for idx, title, col in zip([1, 2, 3, 4, 5], techniques, columns):
        figure, axs = plt.subplots(3, 1, figsize=(8, 10))
        figure.suptitle(title, y=1.01)
        
        top = np.max([get_max(c, col) for c in cases])
        bot = np.min([get_min(c, col) for c in cases])
        
        axs[0].set_title(conditions[0])
        for i in range(len(cases[0])):
            axs[0].plot(cases[0].iloc[i][col])
        axs[0].autoscale(enable=True, axis='x', tight=True)
        axs[0].set_ylim(bottom=bot, top=top)

        axs[1].set_title(conditions[1])
        for i in range(len(cases[1])):
            axs[1].plot(cases[1].iloc[i][col])
        axs[1].autoscale(enable=True, axis='x', tight=True)
        axs[1].set_ylim(bottom=bot, top=top)

        axs[2].set_title(conditions[2])
        for i in range(len(cases[2])):
            axs[2].plot(cases[2].iloc[i][col])
        axs[2].autoscale(enable=True, axis='x', tight=True)
        axs[2].set_ylim(bottom=bot, top=top)

        axs[-1].set_xlabel(f"Figura {idx}")
        plt.tight_layout()
        plt.show()
    

# %%
from TT_utilities import add_moments,add_nonlinear