""" Generating training data using GULP for GAP potential """
import os
import sys
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from colored import fg, bg, attr
import GULP

'''
NOTE:

'''

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

GULP_1 = GULP.GULP(step, vib, SP='set')

try:
    GULP_1.CHANGE_LABEL_TOP_STR("top_structures", ".xyz", DEBUG)
except IndexError:
    pass

loc_top_structures = os.path.join(os.getcwd(), "top_structures")
files = GULP_1.GET_FILE_LIST(loc_top_structures, ".xyz", DEBUG)

columns = shutil.get_terminal_size().columns
print()
print(f"{fg(15)} {bg(21)} Preparing Data {attr(0)}".center(columns))
print()

os.chdir(wd_name)
cwd = os.getcwd()
All_dirs = []
for f in files:
    lambda_energy = pd.DataFrame()
    rank = int(f.split('/')[-1].split('.')[0])
    EIGVALS = {}
    if rank_from <= rank <= rank_to:
        core_write, shel_write, dest, no_of_atoms = GULP_1.CONVERT_XYZ_TO_GULP(f, 0, short_filter, DEBUG)
        try:
            os.mkdir(dest)
        except FileExistsError:
            sys.exit()
        output_xyz_path = os.path.join(dest, dest)
        GULP_1.WRITE_GULP(dest, output_xyz_path, core_write, "n", DEBUG)
        loc_gulp_placed = GULP_1.RUN_GULP(dest)
        raw = GULP_1.OPEN_GULP_OUTPUT(dest, DEBUG)
        IP_energy = GULP_1.GREP_IP_ENERGY(raw, DEBUG)
        total_energy = GULP_1.GREP_TOTAL_ENERGY(raw, DEBUG)
        eigvec_array, freq_line_no, freq_eigval = GULP_1.GREP_FREQ(raw, no_of_atoms, DEBUG)
        force_gulp = GULP_1.GREP_ATOMIC_FORCE(no_of_atoms, dest, DEBUG)
        if dup_filter == "y" and len(vib) != 1:
            eigval_new, eigvec_new = GULP_1.DUP_FILTER(None, dest, eigvec_array, freq_eigval, no_of_atoms, None, 1, 0, DEBUG)
            GULP_2 = GULP.GULP(step, eigval_new, SP="set")
            first_gulp_xyz = os.path.join(dest, f"{dest}_eig.xyz")
            mod_xyz = GULP_2.MODIFY_XYZ(dest, first_gulp_xyz, eigval_new, eigvec_array, no_of_atoms, IP_energy, DEBUG)
        else:
            GULP_2 = GULP.GULP(step, vib, SP='set')
            first_gulp_xyz = os.path.join(dest, f"{dest}_eig.xyz")
            mod_xyz = GULP_2.MODIFY_XYZ(dest, first_gulp_xyz, vib, eigvec_array, no_of_atoms, IP_energy, DEBUG)

        core_write, shel_write, dest, no_of_atoms = GULP_2.CONVERT_XYZ_TO_GULP(0, mod_xyz, short_filter, DEBUG)

        hashtable_e_ = []
        for i in tqdm(range(len(dest)), desc="Preparing vibrational mode strcture data:"):
            All_dirs.append(dest[i])
            core = ' '.join(core_write[i])
            output_xyz_path = os.path.join(dest[i], dest[i].split('/')[-1])
            GULP_2.WRITE_GULP(dest[i], output_xyz_path, core, "y", DEBUG)
            loc_gulp_placed = GULP_2.RUN_GULP(dest[i], DEBUG)

            raw = GULP_2.OPEN_GULP_OUTPUT(dest[i], DEBUG)
            IP_energy = GULP_2.GREP_IP_ENERGY(raw, DEBUG)

            if energy_filter == "y":
                if LOW < float(IP_energy) < HIGH:
                    pass
                else:
                    with open("filtered_by_ENERGY.txt", 'a') as f:
                        f.write(f"{dest[i]} {IP_energy}\n")
                    continue
            else: pass

            total_energy = GULP_2.GREP_TOTAL_ENERGY(raw, DEBUG)
            eigvec_array, freq_line_no, freq_eigval = GULP_2.GREP_FREQ(raw, no_of_atoms, DEBUG)
            force_gulp = GULP_2.GREP_ATOMIC_FORCE(no_of_atoms, dest[i], DEBUG)

            if dup_filter == "n" or len(vib) == 1:
                GULP_2.PREP_EXTENDED_XYZ(dest[i], no_of_atoms, eigvec_array, force_gulp, IP_energy) 
            else: pass

            marker = dest[i].split('_')
            marker = [x.replace('/mod', '') for x in marker]
            rank = marker[0].split('/')[0]
            mode = marker[0].split('/')[1]
            Lambda = marker[1]
            hashtable_e_.append([rank, mode, Lambda, IP_energy])

if dup_filter == "y" and len(vib) != 1:
    target_dirs = GULP_2.DUP_FILTER(All_dirs, dest[i], eigvec_new, freq_eigval, no_of_atoms, hashtable_e_, 0, 1, DEBUG)

    GULP_2.FINAL_PREP()
else:
    GULP_2.FINAL_PREP()

print()
print(f"{fg(15)} {bg(21)} Preparing data for trainig GAP - DONE {attr(0)}".center(columns))




