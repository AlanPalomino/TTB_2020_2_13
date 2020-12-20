from TT_utilities import Case, NL_METHODS
from scipy.stats import stats
from pathlib import Path
import pandas as pd
import re


filename = 'complete_data.csv'


def create_df() -> pd.DataFrame:
    columns = ["case", "record", "cond"]
    for m in NL_METHODS:
        columns.extend([m["tag"] + "_m", m["tag"] + "_v"])
    return pd.DataFrame(columns=columns)


def read_df():
    return pd.read_csv(filename)


try:
    CSV_DATA = read_df()
except FileNotFoundError:
    CSV_DATA = create_df()

RECORDED_CASES = list(CSV_DATA["case"].unique())

RECORD_DIRS = list(Path("./Data").glob("*_p0*"))
for record_dir in RECORD_DIRS:
    record_name = re.search("p[0-9]{6}", str(record_dir))[0]
    if record_name in RECORDED_CASES:
        continue
    c = Case(record_dir.joinpath(record_name))
    c.process()
    for r in c:
        vals = list()
        for k, v in r.N_LINEAR.items():
            s = stats.describe(v)
            vals.extend([s[2], s[3]])
        row = [c._case_name, r.name, c.pathology] + vals
        CSV_DATA = CSV_DATA.append(
            pd.Series(
                data=row,
                index=columns),
            ignore_index=True)
        CSV_DATA.to_csv("complete_data.csv", index=False)
