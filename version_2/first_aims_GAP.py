
import os
import sys
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from colored import fg, bg, attr
import AIMS

arg = sys.argv
vib = arg[1]
step = int(arg[2])
rank_from = int(arg[3])
rank_to = int(arg[4])
cutoff = float(arg[5])
sparse = int(arg[6])
dup_filter = arg[7]
short_filter = arg[8]
energy_filter = arg[9]
if len(arg) == 11:
    DEBUG = arg[10]
else:
    DEBUG = "n"

# filter by energy
HIGH = 50.0
LOW = 0.5

vib = vib.split()
_vib = "-".join(vib)

wd_name = f"GAP_{_vib}_{step}_{rank_from}-{rank_to}_{cutoff}_{sparse}"

if wd_name not in os.listdir('./'):
    os.mkdir(wd_name)

AIMS_1 = AIMS.AIMS(step, vib, DEBUG)

loc_top_structures = os.path.join(os.getcwd(), "top_structures")
files = AIMS_1.GET_FILE_LIST(loc_top_structures, ".xyz", DEBUG)

columns = shutil.get_terminal_size().columns
print()
print(f"{fg(15)} {bg(21)} Preparing Data {attr(0)}".center(columns))
print()

os.chdir(wd_name)
cwd = os.getcwd()
for f in files:
    lambda_energy = pd.DataFrame()
    rank = int(f.split('/')[-1].split('.')[0])
    if rank_from <= rank <= rank_to:
        '''
        final_path_full, last_path, no_of_atoms = AIMS_1.CONVERT_XYZ_TO_GEOMETRY(f)
        AIMS_1.PREP_CON_SUBMIT_FILES(final_path_full, last_path, "n")
        AIMS_1.SUBMIT_AIMS_OPT_JOB(final_path_full, "n")
        '''

        final_path_full = os.path.join(os.getcwd(), f.split('/')[-1].split('.')[0])
        last_path = f.split('/')[-1].split('.')[0]
        aims_energy, aims_final_energy, force, coord, atom, no_of_atoms = AIMS_1.GREP_AIMS_OPT(final_path_full, 12) #no_of_atoms)
        #AIMS_1.SUBMIT_AIMS_OPT_JOB(final_path_full, "y")
        AIMS_1.GREP_AIMS_VIB(final_path_full, no_of_atoms)

os.chdir('..')
#shutil.rmtree(wd_name)












