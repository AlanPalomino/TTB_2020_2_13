from pathlib import Path
import pandas as pd


data_dirs = list(Path("Data/").glob("*_p0*"))
csv_dirs = list(Path("csv_dump/").glob("*_data"))

MAIN_CSV = pd.read_csv("complete_data.csv")
print(MAIN_CSV.columns)
print(f"Current data has: {len(MAIN_CSV['case'].unique())} cases.")

for csv_file in csv_dirs:
    temp_df = pd.read_csv(csv_file)
    MAIN_CSV = MAIN_CSV.append(temp_df, ignore_index=True)
    csv_file.unlink()

MAIN_CSV.to_csv("complete_data.csv", index=False)
