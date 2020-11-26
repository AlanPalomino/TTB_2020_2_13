from pathlib import Path
import os
import csv


def complete_db():
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

    with open("download_complete.sh", "w+", newline="\n") as f:
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
    with open("download_complete.sh", "r") as f:
        print(f"\n{len(f.readlines())//10} of {len(data)} records in the bash downloader.")

def worksample_db():
    patient_files = [ file for file in os.listdir() if file.endswith(".txt") and file != "RECORDS.txt" ]
    data = list()
    with open("RECORDS.txt", "r") as r:
        RECORDS = r.read()

    for condition_file in patient_files:
        condition = Path(condition_file).stem
        i = 0
        with open(condition_file, "r") as f:
            condition_data = [(condition, RECORDS[idx-4:idx+7]) for idx in 
                              [RECORDS.find(f"p{int(line):06d}") for line in 
                               f.read().splitlines() if RECORDS.find(f"p{int(line):06d}") != -1]]
            data.extend(condition_data[:3])
        print(f"- {condition} has {len(condition_data)} records.")

    with open("download_worksample.sh", "w+", newline="\n") as f:
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

def main():
    complete_db()
    worksample_db()


if __name__ == "__main__":
    main()
