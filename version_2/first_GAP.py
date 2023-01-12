""" Generating training data using GULP for GAP potential """
import os
import sys
import shutil
import pandas as pd
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
if len(arg) == 9:
    DEBUG = arg[8]
else:
    DEBUG = "n"

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

#if "ext_movie.xyz" in os.listdir('./'):
#    os.remove('ext_movie.xyz')

columns = shutil.get_terminal_size().columns
print()
print(f"{fg(15)} {bg(21)} Preparing Data {attr(0)}".center(columns))
print()

os.chdir(wd_name)
cwd = os.getcwd()
for f in files:
    lambda_energy = pd.DataFrame()
    rank = int(f.split('/')[-1].split('.')[0])
    EIGVALS = {}
    if rank_from <= rank <= rank_to:
        (core_write, shel_write, coord_only, dest, no_of_atoms) = GULP_1.CONVERT_XYZ_TO_GULP(f, 0, DEBUG)
        os.mkdir(dest)
        output_xyz_path = os.path.join(dest, dest)
        GULP_1.WRITE_GULP(dest, output_xyz_path, core_write, "n", DEBUG)
        loc_gulp_placed = GULP_1.RUN_GULP(dest)
        raw = GULP_1.OPEN_GULP_OUTPUT(dest, DEBUG)
        IP_energy = GULP_1.GREP_IP_ENERGY(raw, DEBUG)
        total_energy = GULP_1.GREP_TOTAL_ENERGY(raw, DEBUG)
        eigvec_array, freq_line_no, freq_eigval = GULP_1.GREP_FREQ(raw, no_of_atoms, DEBUG)
        force_gulp = GULP_1.GREP_ATOMIC_FORCE(no_of_atoms, dest, "y", DEBUG)
        if dup_filter == "y":
            eigval_new, eigvec_new = GULP_1.DUP_FILTER(dest, eigvec_array, freq_eigval, no_of_atoms, None, 1, 0, DEBUG)
            print(eigval_new, eigvec_new)
            print(vib, eigvec_array) 
            GULP_2 = GULP.GULP(step, eigval_new, SP="set")

        #GULP_2 = GULP.GULP(step, vib, SP='set')
        first_gulp_xyz = os.path.join(dest, f"{dest}_eig.xyz")
        mod_xyz = GULP_2.MODIFY_XYZ(dest, first_gulp_xyz, eigval_new, eigvec_new, no_of_atoms, IP_energy, DEBUG="n")

        core_write, shel_write, coord_only, dest, no_of_atoms = GULP_2.CONVERT_XYZ_TO_GULP(0, mod_xyz, DEBUG)

        hashtable_e_ = []
        for i in tqdm(range(len(core_write)), desc="Preparing vibrational mode strcture data:"):
            core = ' '.join(core_write[i])
            output_xyz_path = os.path.join(dest[i], dest[i].split('/')[-1])
            GULP_2.WRITE_GULP(dest[i], output_xyz_path, core, "y", DEBUG)

            loc_gulp_placed = GULP_2.RUN_GULP(dest[i], DEBUG)
            if dup_filter == "y":
                condi = GULP_2.SHORT_DIST_FILTER(dest[i], no_of_atoms)
                if condi == True:
                    print(dest[i])
                    continue

            raw = GULP_2.OPEN_GULP_OUTPUT(dest[i], DEBUG)

            IP_energy = GULP_2.GREP_IP_ENERGY(raw, DEBUG)

            total_energy = GULP_2.GREP_TOTAL_ENERGY(raw, DEBUG)

            eigvec_array, freq_line_no, freq_eigval = GULP_2.GREP_FREQ(raw, no_of_atoms, DEBUG)

            force_gulp = GULP_2.GREP_ATOMIC_FORCE(no_of_atoms, dest[i], "y", DEBUG)

            GULP_2.PREP_EXTENDED_XYZ(dest[i], no_of_atoms, eigvec_array, force_gulp, IP_energy) 

            Lambda = dest[i].split('_')
            #hashtable_e[Lambda] = float(IP_energy)
            hashtable_e_.append([Lambda[0].split('/')[0], Lambda[1], IP_energy])
GULP_2.DUP_FILTER(dest[i], eigvec_new, freq_eigval, no_of_atoms, hashtable_e_, 0, 1, DEBUG)
GULP_2.FINAL_PREP()

columns = shutil.get_terminal_size().columns
print()
print(f"{fg(15)} {bg(21)} Preparing data for trainig GAP - DONE {attr(0)}".center(columns))

