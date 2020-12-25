from pathlib import Path
import re
import pandas as pd


DOWNLOADED = list(Path("Data/").glob("*_p0*"))
csv_data = pd.read_csv("complete_data.csv")
PROCESSED = csv_data["case"].unique()

print(f"\nDe los presentes {len(DOWNLOADED)} casos se han procesado {len(PROCESSED)}\n")
print(f"Myocardial infarction rows: {len(csv_data[csv_data['cond'] == 'myocardial_infarction'])}")
print(f"Congestive heart failure rows: {len(csv_data[csv_data['cond'] == 'congestive_heartfailure'])}")
print(f"Atrial fibrillation rows: {len(csv_data[csv_data['cond'] == 'atrial_fibrillation'])}\n")

