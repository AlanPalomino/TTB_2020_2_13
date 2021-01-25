#!usr/bin/env python3
# _*_ coding: utf-8 _*_ #
#
#___________________________________________________________________________
#|                                                                         |
#|    TTB__2020_1_13 server code:                                          |
#|      Código para distribución de carga en el servidor                   |
#|                                                                         |
#|                                                                         |
#|_________________________________________________________________________|


# ===================== Librerias Utilizadas ====================== #
from TT_utilities import Case, NL_METHODS
from TT_utilities import Record
from multiprocessing import Pool
from scipy.stats import stats
from pathlib import Path
from entropy import spectral_entropy
from hurst import compute_Hc
import TT_utilities
import pandas as pd
import numpy as np
import pickle
import sys
import os
import re


CSV_COLS = [
    'case',
    'record',
    'condition',
    'cond_id',
    'length',
]
for m in NL_METHODS:
    CSV_COLS.extend([
        m['tag'] + '_mean',
        m['tag'] + '_variance',
        m['tag'] + '_skewness',
        m['tag'] + '_spectral_entropy',
    ])


COND_ID = dict(
    atrial_fibrillation=0,
    congestive_heartfailure=1,
    myocardial_infarction=2,
    control=3
)

# ===================== Funciones y Métodos ======================= #
def hurst_eval(rr):
    H, _, _ = compute_Hc(rr)
    return H


def generate_csv():
    cases_list = unpickle_data()
    csv_name = 'complete_data.csv'
    FULL_CSV = pd.DataFrame(columns=CSV_COLS)
    for c in cases_list:
        print(f"    > Case {c._case_name}")
        for r in c:
            print(f"\t\t + RECORD {r.name}", end="")
            values = list()
            for k, v in r.N_LINEAR.items():
                s = stats.describe(v)
                values.extend([
                    s[2],                                       # Mean
                    s[3],                                       # Variance
                    s[4],                                       # Skewness
                    spectral_entropy(v, sf=r.fs, method='fft')  # Spectral Entropy
                ])
            row_data = [
                c._case_name,               # Case
                r.name,                     # Record
                c.pathology,                # Condition
                COND_ID[c.pathology],       # Condition ID
                len(r.rr),                  # RR Length
            ] + values
            FULL_CSV = FULL_CSV.append(
                pd.Series(
                    data=row_data,
                    index=CSV_COLS
                ), ignore_index=True
            )
            print("[v]")
    FULL_CSV.to_csv(csv_name, index=False)


def unpickle_data():
    p_paths = list(Path('./Pickled').glob('*.pkl')) 
    UNPICKLED = list()
    for pkl in p_paths:
        with pkl.open('rb') as pf:
            UNPICKLED.append(
                pickle.load(pf)
            )
    return UNPICKLED


def process_case(case_path: Path):
    c = Case(case_path)
    c.process()
    print(f"\n\n\t\tCASE {c._case_name} has {len(c)} RECORDS\n\n")
    if len(c) > 0:
        with open(f'Pickled/case_{c._case_name}.pkl', 'wb') as pf:
            pickle.dump(c, pf)


def gen_name(path):
    c_name = re.search('p[0-9]{6}', str(path))[0]
    return path.joinpath(c_name)


def pickle_data():
    RECORD_DIRS = list(Path("./Data").glob("*_p0*")) 
    RECORD_DIRS = [gen_name(p) for p in RECORD_DIRS]

    p = Pool()
    p.map(process_case, RECORD_DIRS)
    p.close()


def help():
    global OPTS
    print("""
    SERVER SCRIPT OPTIONS
        Exclusive options for use in server!!
          """)
    for opt in OPTS:
        print(f"{', '.join(opt['opts'])} :")
        print(f"\t{opt['desc']}\n")


def test_unpickle(parent):
    unpickled_cases = list()
    for d in parent.glob('*.pkl'):
        with d.open('rb') as pf:
            unpickled_cases.append(
                pickle.load(pf)
            )
    return unpickled_cases


def save_test():
    TEST_DIRS = list(Path('.').glob('Test_*ws/'))
    for td in TEST_DIRS:
        t_cases = test_unpickle(td)

        pdir = "Test/"

        csv_name = pdir + td.stem + '.csv'
        pkl_name = pdir + td.stem + '.pkl'

        csv_data = pd.DataFrame(columns=CSV_COLS)
        pkl_data = pd.DataFrame(columns=CSV_COLS[:5])

        for c in t_cases:
            for r in c:
                # Process for CSV
                values = list()
                row_data = [
                    c._case_name,
                    r.name,
                    c.pathology,
                    COND_ID[c.pathology],
                    len(r.rr_int),
                ]
                for k, v in r.N_LINEAR.items():
                    s = stats.describe(v)
                    row_data.extend([
                        s[2],
                        s[3],
                        s[4],
                        spectral_entropy(v, sf=r.fs, method='fft')
                    ])
                csv_data = csv_data.append(
                    pd.Series(
                        data=row_data,
                        index=CSV_COLS,
                    ), ignore_index=True
                )
                # Process for pickle
                pkl_row = {
                    'case': c._case_name,
                    'record': r.name,
                    'condition': c.pathology,
                    'cond_id': COND_ID[c.pathology],
                    'length': len(r.rr_int)
                }
                pkl_row.update(r.N_LINEAR)
                pkl_data = pkl_data.append(pd.DataFrame(pkl_row))

        # DATA IS SAVED IN BOTH FORMATS
        csv_data.to_csv(csv_name, index=False)
        with open(pkl_name, 'wb') as pf:
            pickle.dump(pkl_data, pf)


def test_case(ddir: Path):
    c = Case(ddir)
    c.process()
    if len(c) != 0:
        with open(f'Test_{TT_utilities.RR_WLEN}ws/case_{c._case_name}.pkl', 'wb') as pf:
            pickle.dump(c, pf)
        print(f'\n\n\tTEST CASE with {len(c)} records processed and saved to: case_{c._case_name}.pkl\n\n')


def run_test():
    n = int(sys.argv[2])

    af_dirs = list(Path('Data/').glob('atrial_fibrillation_p*'))[:n]
    mi_dirs = list(Path('Data/').glob('myocardial_infarction_p*'))[:n]
    ch_dirs = list(Path('Data/').glob('congestive_heartfailure_p*'))[:n]

    try:
        os.mkdir(f"Test_{TT_utilities.RR_WLEN}ws/")
    except FileExistsError:
        print("REWRITING TEST VALUES")

    data_dirs = [ gen_name(d) for d in af_dirs + mi_dirs + ch_dirs]

    p = Pool()
    p.map(test_case, data_dirs)
    p.close()


def main(argv):
    global OPTS
    for opt in OPTS:
        print(f'is {argv} in {opt["opts"]}')
        if argv in opt['opts']:
            opt['func']()
            break
    else:
        print("""
        No valid parameter detected
        Check bellow for valid options:
              """)
        help()


OPTS = [
    {
        'opts': ['-h', '--help'],
        'desc': 'Prints valid options to use the script.',
        'func': help
    },{
        'opts': ['-pd', '--pickle_data'],
        'desc': 'Processes and pickles downloaded data',
        'func': pickle_data
    },{
        'opts': ['-gc', '--generate_csv'],
        'desc': 'Unpickles data and generates the corresponding csv.',
        'func': generate_csv
    },{
        'opts': ['-rt', '--run_test'],
        'desc': 'Run selected test with [n] number of cases per pathology.',
        'func': run_test
    },{
        'opts': ['-st', '--save_test'],
        'desc': 'Recounts each data compendium and generates a csv file.',
        'func': save_test
    }
]

# ===================== Ejecución principal ======================= #
if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1])

