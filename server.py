from TT_utilities import Case, NL_METHODS
from multiprocessing import Pool
from scipy.stats import stats
from pathlib import Path
from entropy import spectral_entropy
import pandas as pd
import numpy as np
import pickle
import sys
import re


def hurst_eval(rr):
    return 1


def generate_csv():
    condition_ids = dict(
        atrial_fibrillation=0,
        congestive_heartfailure=1,
        myocardial_infarction=2
    )
    cases_list = unpickle_data()
    csv_name = 'complete_data.csv'
    columns = [
        'case',
        'record',
        'condition',
        'cond_id',
        'hurst',
    ]
    for m in NL_METHODS:
        columns.extend([
            m['tag'] + '_mean',
            m['tag'] + '_variance',
            m['tag'] + '_skewness',
            m['tag'] + '_spectral_entropy'
        ])
    FULL_CSV = pd.DataFrame(columns=columns)
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
                condition_ids[c.pathology],  # Condition ID
                hurst_eval(r.rr)            # RR Hurst value
            ] + values
            FULL_CSV = FULL_CSV.append(
                pd.Series(
                    data=row_data,
                    index=columns
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
           
    
def pickle_data(num: int=0):
    def gen_name(path):
        c_name = re.search('p[0-9]{6}', str(path))[0]
        return path.joinpath(c_name)
    
    def process_case(case_path: Path):
        c = Case(case_path)
        c.process()
        if len(c) > 0:
            with open(f'case_{c._case_name}.pkl', 'wb') as pf:
                pickle.dump(c, pf)

    RECORD_DIRS = list(Path("./Data").glob("*_p0*")) 
    RECORD_DIRS = [gen_name(p) for p in RECORD_DIRS]
    if num != 0:
        RECORD_DIRS = RECORD_DIRS[:num]

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
        print(f"\n{opt['desc']}")


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
    }
]


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1])

