#!usr/bin/env python3
# _*_ coding: utf-8 _*_ #
#
#___________________________________________________________________________
#|                                                                         |
#|    TTB__2020_1_13 Main code:                                            |
#|      Código principal para el trabajo                                   |
#|                                                                         |
#|                                                                         |
#|_________________________________________________________________________|


# ===================== Librerias Utilizadas ====================== #

from wfdb.processing.qrs import gqrs_detect
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from itertools import combinations 
from matplotlib import gridspec
from pprint import pprint 
from scipy import stats
#import entropy as tpy
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
from TT_utilities import add_moments,add_nonlinear, plot_NL_metrics, distribution_NL

# ================================================================= #

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
AF_CASES = data[(data["conditon"] == "AF") & (data["length"] > RR_WINDOW_THRESHOLD)][:num_cases]
# CHF - Congestive Heart Failure
CHF_CASES = data[(data["conditon"] == "CHF") & (data["length"] > RR_WINDOW_THRESHOLD)][:num_cases]
# HC - Healthy Controls
HC_CASES = data[(data["conditon"] == "HC") & (data["length"] > RR_WINDOW_THRESHOLD)][:num_cases]
# AR - Arrhythmia Cases
AR_CASES = data[(data["conditon"] == "AR") & (data["length"] > RR_WINDOW_THRESHOLD)][:num_cases]   # NO HAY CASOS QUE CUMPLAN 
# MI - Myocardial Infarction
MI_CASES = data[(data["conditon"] == "MI") & (data["length"] > RR_WINDOW_THRESHOLD)][:num_cases]   # NO HAY CASOS QUE CUMPLAN
print(f"""
AF CASES: {len(AF_CASES)}
CHF CASES: {len(CHF_CASES)}
HC CASES: {len(HC_CASES)}
AR CASES: {len(AR_CASES)}
MI CASES: {len(MI_CASES)}
""")


#   MIMIC 3 DATA LOAD
RECORD_DIRS = list(Path("./Data").glob("*p00*"))
for record_dir in RECORD_DIRS:
    record_name = re.search("p[0-9]{6}", str(record_dir))[0]
    case = Case(record_dir.joinpath(record_name))
    break


#======== Agregando las métricas obtenidas a las Bases 
print("ACTUALIZANDO DATABASES...")
AF_CASES = AF_CASES.apply(add_moments, axis=1)
CHF_CASES = CHF_CASES.apply(add_moments, axis=1)
HC_CASES = HC_CASES.apply(add_moments, axis=1)

AF_CASES_NL = AF_CASES.copy()
CHF_CASES_NL = CHF_CASES.copy()
HC_CASES_NL = HC_CASES.copy()
print("Métricas Lineales agregadas:  ")

print(" - ".join(AF_CASES.columns))
print(" - ".join(CHF_CASES.columns))
print(" - ".join(HC_CASES.columns))

print("ACTUALIZANDO DATABASES...")

AF_CASES = AF_CASES.apply(add_nonlinear, axis=1)
CHF_CASES = CHF_CASES.apply(add_nonlinear, axis=1)
HC_CASES = HC_CASES.apply(add_nonlinear, axis=1)
#AF_CASES_NL = AF_CASES_NL.apply(add_nonlinear, axis=1)
#CHF_CASES_NL = CHF_CASES_NL.apply(add_nonlinear, axis=1)
#HC_CASES_NL = HC_CASES_NL.apply(add_nonlinear, axis=1)

print("Métricas No-lineales agregadas:  ")
print(" - ".join(AF_CASES.columns))
print(" - ".join(CHF_CASES.columns))
print(" - ".join(HC_CASES.columns))

#print(" - ".join(AF_CASES_NL.columns))
#print(" - ".join(CHF_CASES_NL.columns))
#print(" - ".join(HC_CASES_NL.columns))


# Ploteo de Evolución temporal  NL
conditions = ["Fibrilación Atrial", "Insuficiencia Cardíaca Congestiva", "Casos Saludables"]
techniques = ["Entropía aproximada", "Entropía muestral", "Analisis de Fluctuación sin Tendencia (DFA)", "Coeficiente de Higuchi (HFD)","Radio = SD1/SD2"]
columns = ["AppEn", "SampEn", "DFA", "HFD","SD_ratio"]
cases = [AF_CASES, CHF_CASES, HC_CASES]
    
plot_NL_metrics(cases, techniques, conditions, columns)

# Ploteo de Distribuciones NL


conditions = ["Fibrilación Atrial", "Insuficiencia Cardíaca Congestiva", "de Control"]
techniques = ["Entropía aproximada", "Entropía muestral", "Analisis de Fluctuación sin Tendencia (DFA)", "Coeficiente de Higuchi (HFD)","Radio = SD1/SD2"]
columns = ["AppEn", "SampEn", "DFA", "HFD","SD_ratio"]
cases = [AF_CASES, CHF_CASES, HC_CASES]
"""
for idx in range(len(cases)):
    distribution_NL(cases[idx], conditions[idx])
"""
distribution_NL(HC_CASES_NL, 'Grupo Sano')


# KS TEST (CONVERTIR EN FUNCIÓN GENERAL Y BORRAR DE MAIN)
conditions = ["FA", "ICC", "Control"]
Databases = [AF_CASES, CHF_CASES, HC_CASES]

columns = ["AppEn", "SampEn", "DFA", "HFD","SD_ratio"]
ks_test=list()
        

for Data,cond in zip(Databases, conditions):
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

#KS_Testing(Databases, conditions)

comp_data = pd.read_csv('complete_data.csv')
MainDF  = pd.DataFrame(comp_data)



MainDummy = pd.concat([HC_CASES,AF_CASES,CHF_CASES])
MainDummy

