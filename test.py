#!usr/bin/env python3
# _*_ coding: utf-8 _*_ #
#
#___________________________________________________________________________
#|                                                                         |
#| Pruebas en servidor:                                                    |
#|      Script para generación de pruebas en el servidor.                  |
#|                                                                         |
#|_________________________________________________________________________|

from TT_utilities import NL_METHODS, nonLinearWindowing
from entropy import spectral_entropy
from scipy.stats import stats
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os


punctual_names = [
    '_mean',
    '_variance',
    '_skewness',
    '_spectral_entropy'
]


COND_ID = dict(
    AF=['atrial_fibrillation', 0],
    AR=['atrial_fibrillation', 0],
    CHF=['congestive_heartfailure', 1],
    MI=['myocardial_infarction', 2],
    HC=['control', 3]
)


def vector2csv(df):
    def process_row(row: pd.Series) -> pd.Series:
        data = dict(
            row[[m["tag"] for m in NL_METHODS]]
        )
        for tag, vec in data.items():
            s = stats.describe(vec)
            values = [
                s[2],
                s[3],
                s[4],
                spectral_entropy(vec, sf=row['fs'], method='fft')
            ]
            for n, v in zip(punctual_names, values):
                row[tag + n] = v
        return row

    df_extended = df.apply(process_row, axis=1)
    for m in NL_METHODS:
        del df_extended[m["tag"]]
    del df_extended["fs"]
    del df_extended['rr']

    return df_extendedo


def dummy_process(jsonfiles: list, filename: str) -> pd.DataFrame:
    def get_ids(row: pd.Series) -> pd.Series:
        try:
            row['condition'] = COND_ID.get(row['conditon'])[0]
            row['cond_id'] = COND_ID.get(row['conditon'])[1]
        except KeyError as e:
            print(row)
            raise KeyError(e)
        return row

    # Data es read from json paths
    data = list()
    for jf in jsonfiles:
        with jf.open() as file:
            mixed = json.load(file)
            data.extend([reg for reg in mixed if reg['approved']])
    data = pd.DataFrame(data)
    data = data.apply(get_ids, axis=1)
    # Dataframe adjustment and ordering
    data["rr"] = data.apply(lambda row: np.array(row["rr"])/row["fs"], axis=1)
    data["rr"] = data["rr"].apply(lambda signal: signal[np.where(signal < 2)])
    data["length"] = data["rr"].apply(lambda signal: len(signal))
    data["case"] = data["record"]
    # We filter for only relevant columns
    data = data[['case', 'record', 'condition', 'cond_id', 'length', 'fs', 'rr']]
    print(f" < Vectorizing for {filename}...", end="")
    pickleData = vectorize_df(data)
    print("\tDONE >")

    print(f" < Generate Pickle for {filename}...", end="")
    with open(filename, "wb") as pf:
        pickle.dump(pickleData, pf)
    print("\tDONE >")
    
    return pickleData


def check_ids():
    jsonfiles = [
            Path('Data_Jsons/afdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/chfdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/ltafdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/mitdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/ptbdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/normal-sinus-rhythm-rr-interval-database-1.0.0.json'),
            Path('Data_Jsons/nn-cases-healthy-control.json')
        ]
    ls = list()
    for j in jsonfiles:
        with j.open() as f:
            d = [r for r in json.load(f) if r["approved"]]
            print(j)
            print(d[0].keys())
            try:
                ls.extend([r["conditon"] for r in d])
            except KeyError:
                print(" > has no condition data")
    print(f"\nColumns: {set(ls)}\n")


def load_dummy():
    try:
        with open('Test/linear_healthy.pkl', 'rb') as pf:
            pickleData = pickle.load(pf)
        with open('Test/linear_sick.pkl', 'rb') as pf:
            pickleData = pickleData.append(pickle.load(pf))

    except FileNotFoundError:

        print('No previous Pickled data has been found, generating from zero.')

        
        s_jsonfiles = [
            Path('Data_Jsons/afdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/chfdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/ltafdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/mitdb-1.0.0.physionet.org.json'),
            Path('Data_Jsons/ptbdb-1.0.0.physionet.org.json'),
            ]

        h_pickleData = dummy_process(h_jsonfiles, 'Test/linear_healthy.pkl')
        s_pickleData = dummy_process(s_jsonfiles, 'Test/linear_sick.pkl')

    return

    print(" < Generate .csv...", end="")
    csvData = vector2csv(hpickleData)
    csvData.to_csv("Test/healthy.csv")
    print("\tDONE >")





def linear2csv():
    def edit_row(row):
        rr = row['rr']
        s = stats.describe(rr)
        new_row = row[['record', 'condition']]
        new_row['mean'] = s[2]
        new_row['variance'] = s[3]
        new_row['skewness'] = s[4]
        new_row['kurtosis'] = s[5]
        return new_row


    def process_files(jsonfiles, csvfilename):
        json_data = list()
        for jf in jsonfiles:
            with jf.open() as file:
                mixed = json.load(file)
                appr = [reg for reg in mixed if reg['approved']]
                json_data.extend(appr)
        DATA = pd.DataFrame(json_data)
        DATA['condition'] = DATA['conditon']
        DATA['rr'] = DATA.apply(lambda row: np.array(row['rr'])/row['fs'], axis=1)
        DATA['rr'] = DATA['rr'].apply(lambda signal: signal[np.where(signal < 2)])
        
        DATA = DATA.apply(edit_row, axis=1)
        DATA.to_csv(csvfilename, index=True)

    h_jsonfiles = [
            Path('Data_Jsons/normal-sinus-rhythm-rr-interval-database-1.0.0.json'),
            Path('Data_Jsons/nn-cases-healthy-control.json')
            ]
    s_jsonfiles = [
        Path('Data_Jsons/afdb-1.0.0.physionet.org.json'),
        Path('Data_Jsons/chfdb-1.0.0.physionet.org.json'),
        Path('Data_Jsons/ltafdb-1.0.0.physionet.org.json'),
        Path('Data_Jsons/mitdb-1.0.0.physionet.org.json'),
        Path('Data_Jsons/ptbdb-1.0.0.physionet.org.json'),
        ]

    process_files(h_jsonfiles, 'linear_healthy.csv')
    process_files(s_jsonfiles, 'linear_sick.csv')


    # Get RR stats from each row
    # csvLinearData = pickleData.apply(edit_row, axis=1)
    # csvLinearData.to_csv('Test/linear_healthy.csv', index=False)


if __name__ == "__main__":
        load_dummy()
