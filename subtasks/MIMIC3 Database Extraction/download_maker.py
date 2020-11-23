from pathlib import Path
import os
import csv


def main():
    patient_files = [ file for file in os.listdir() if file.endswith(".txt") and file != "RECORDS.txt" ]
    data = list()
    with open("RECORDS.txt", "r") as r:
        RECORDS = r.read()

    for condition_file in patient_files:
        condition = Path(condition_file).stem
        with open(condition_file, "r") as f:
            condition_data = [(condition, RECORDS[idx-4:idx+7]) for idx in 
                              [RECORDS.find(f"p{int(line):06d}") for line in 
                               f.read().splitlines() if RECORDS.find(f"p{int(line):06d}") != -1]]
            data.extend(condition_data)
        print(f"- {condition} has {len(condition_data)} records.")

    with open("download_commands.sh", "w+", newline="\n") as f:
        i = 0
        for condition, record_route in data:
            f.write(f"""
mkdir {condition}_{record_route[4:]}
cd {condition}_{record_route[4:]}
if [[ ! -d $PWD/{record_route[4:]} ]]
then
    wget -r -nc -np -c https://physionet.org/static/published-projects/mimic3wdb-matched/1.0/{record_route}/
    mv physionet.org/static/published-projects/mimic3wdb-matched/1.0/{record_route} .
fi
rm -r physionet.org/
cd ..
                """)
    with open("download_commands.sh", "r") as f:
        print(f"\n{len(f.readlines())//5} of {len(data)} records in the bash downloader.")


if __name__ == "__main__":
    main()
