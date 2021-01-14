from TT_utilities import NL_METHODS, nonLinearWindowing
from scipy.stats import stats
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os


punctual_names = [
    '_mean',
    '_variance',
    '_skewness',
    '_spectral_entropy'
]


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
                spectral_entropy(v, sf=row['fs'], method='fft')
            ]
            for n, v in zip(punctual_names, values):
                row[tag + n] = v
        return row

    df_extended = df.apply(process_row, axis=1)
    for m in NL_METHODS:
        del df_extended[m["tag"]]

    return df_extended


def load_healthy():
    jsonfiles = [
        Path('Data_Jsons/normal-sinus-rhythm-rr-interval-database-1.0.0.json'),
        Path('Data_Jsons/nn-cases-healthy-control.json')
    ]
    data = list()
    for jf in jsonfiles:
        with jf.open() as file:
            mixed = json.load(file)
            data.extend([reg for reg in mixed if reg['approved']])
    print(" < READING FINISHED >")
    data = pd.DataFrame(data)

    # Dataframe adjustment and ordering
    data["rr"] = data.apply(lambda row: np.array(row["rr"])/row["fs"], axis=1)
    data["rr"] = data["rr"].apply(lambda signal: signal[np.where(signal < 2)])
    data["length"] = data["rr"].apply(lambda signal: len(signal))
    data["case"] = data["database"]
    data['condition'] = 'control'
    data['cond_id'] = 3

    data = data[['case', 'record', 'condition', 'cond_id', 'length', 'fs', 'rr']]
    csvData = data

    print(" < Vectorizing...", end="")
    pickleData = vectorize_df(data)
    print("\tDONE >")

    print(" < Generate Pickle...", end="")
    with open("Test/healthy.pkl", "wb") as pf:
        pickle.dump(pickleData, pf)
    print("\tDONE >")

    print(" < Generate .csv...", end="")
    csvData = vector2csv(pickleData)
    csvData.to_csv("Test/healthy.csv")
    print("\tDONE >")

    return data


def vectorize_df(data: pd.DataFrame):
    def gen_vectors(row: pd.Series) -> pd.Series:
        rr = row['rr']
        for m, t in zip(NL_METHODS, nonLinearWindowing(rr)):
            row[m['tag']] = t
        return row
    return data.apply(gen_vectors, axis=1)


if __name__ == "__main__":
    df = load_healthy()
    print(df.columns)
    print(df.loc[0])
