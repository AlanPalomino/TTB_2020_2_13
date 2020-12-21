from pathlib import Path

DOWNLOADED = list(Path("Data/").glob("*_p0*"))
print(len(DOWNLOADED))
