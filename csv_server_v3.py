from TT_utilities import Case, NL_METHODS
from scipy.stats import stats
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
import re


filename = 'complete_data.csv'
columns = ["case", "record", "cond"]
for m in NL_METHODS:
    columns.extend([m["tag"] + "_m", m["tag"] + "_v"])


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
    for r in c:
        vals = list()
        for k, v in r.N_LINEAR.items():
            s = stats.describe(v)
            vals.extend([s[2], s[3]])
        row = [c._case_name, r.name, c.pathology] + vals
        case_DF = case_DF.append(
            pd.Series(
                data=row,
                index=columns
            ),
            ignore_index=True
        )
    case_DF.to_csv(f"csv_dump/{c._case_name}_data")


try:
    CSV_DATA = read_df()
except FileNotFoundError:
    CSV_DATA = create_df()

RECORDED_CASES = list(CSV_DATA["case"].unique())

RECORD_DIRS = list(Path("./Data").glob("*_p0*"))

CASE_DIRS = list()
for c in RECORD_DIRS:
    case_name = gen_name(c)
    if case_name not in RECORDED_CASES:
        CASE_DIRS.append(c.joinpath(case_name))

p = Pool()
p.map(process_case, CASE_DIRS)
p.close()

CSV_DATA.to_csv(filename, index=False)
