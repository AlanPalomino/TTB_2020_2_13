from TT_utilities import Case, NL_METHODS
from scipy.stats import stats
from pathlib import Path
from multiprocessing import Pool
import pickle
import pandas as pd
import re


def create_df() -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def read_df():
    return pd.read_csv(filename)


def gen_name(path):
    return re.search("p[0-9]{6}", str(path))[0]


def process_case(case_path):
    global CSV_DATA
    case_DF = create_df()
    c = Case(case_path)
    c.process()
    with open(f"case_{c._case_name}.pkl", "wb") as pf:
        pickle.dump(c, pf)


RECORD_DIRS = list(Path("./Data").glob("*_p0*"))
RECORD_DIRS = [RECORD_DIRS[0]]

p = Pool()
p.map(process_case, RECORD_DIRS)
p.close()

