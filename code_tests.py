# %%
#_
#___________________________________________________________________________
#|                                                                         |
#| Playground para pruebas:                                                |
#|      De ser necesario probar código, este script puede ser usaado       |
#|      para ese propósito. El archivo se eliminará del repositorio        |
#|      una vez concluido el proyecto.                                     |
#|_________________________________________________________________________|

from TT_utilities import Case
from pathlib import Path
import re


# %%
RECORD_DIRS = list(Path("./Data").glob("*p00*"))
CASES = list()
for record_dir in RECORD_DIRS:
    record_name = re.search("p[0-9]{6}", str(record_dir))[0]
    c = Case(record_dir.joinpath(record_name))
    CASES.append(c)
    print(c)

