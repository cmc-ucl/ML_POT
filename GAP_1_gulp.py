import gulp
import os, sys, time, gc
import random
from colored import fg, bg, attr
import pandas as pd
import numpy as np



FROM = int(sys.argv[1])
TO = int(sys.argv[2])
STEP = int(sys.argv[3])
FROM_rank = int(sys.argv[4])
TO_rank = int(sys.argv[5])
Breath = sys.argv[6]
cutoff = float(sys.argv[7])
sparse = int(sys.argv[8])
wd_name = f'GAP_{FROM}-{TO}_{STEP}_{FROM_rank}-{TO_rank}_{Breath}_{cutoff}_{sparse}'
if wd_name  not in os.listdir('./'):
    os.mkdir(wd_name)

GULP = gulp.GULP(STEP, FROM, TO, SP='set')

# Re-naming the xyz files in the top_structures
try:
    GULP.Re_top_str()
except IndexError:
    pass

files = GULP.Get_file_list(os.getcwd() + '/top_structures')

if 'ext_movie.xyz' in os.listdir('./'):
    os.remove('ext_movie.xyz')

print(f'{fg(15)} {bg(5)} Preparing Data {attr(0)}')
print()
print()


os.chdir(wd_name)
for f in files:
    rank = int(f.split('/')[-1].split('.')[0])
    if FROM_rank <= rank <= TO_rank:
        cation, anion_core, anion_shel, DIR_IP_RANK, no_of_atoms = GULP.Convert_xyz_Gulp(f)
        cwd = os.getcwd()
        os.mkdir(DIR_IP_RANK)
        GULP_gout_PATH = os.path.join(cwd, DIR_IP_RANK)
        GULP_OUT_PATH = GULP_gout_PATH.split('/')[-1] + '/' + GULP_gout_PATH.split('/')[-1]
        GULP.Write_Gulp(DIR_IP_RANK, GULP_OUT_PATH, cation, anion_core, anion_shel, 'n')
        Gulp_output_path = GULP.Run_Gulp(GULP_gout_PATH, DIR_IP_RANK)
        total_energy, eigvec_array, freq = GULP.Grep_Data(Gulp_output_path, no_of_atoms, DIR_IP_RANK, 'n')

        if FROM != 0 and TO != 0:
            GULP.Modifying_xyz(DIR_IP_RANK, GULP_gout_PATH + f'/{DIR_IP_RANK}_eig.xyz', eigvec_array, freq, no_of_atoms, total_energy, Breath)
        else:
            pass


        if Breath == 'y':
            GULP.Breathing_xyz(DIR_IP_RANK, GULP_gout_PATH + f'/{DIR_IP_RANK}_eig.xyz', no_of_atoms, total_energy)
        else:
            pass

        sub_wd = [x for x in os.listdir(f'{GULP_gout_PATH}') if os.path.isdir(f'{GULP_gout_PATH}/{x}')]
        sub_wd = sorted(sub_wd, key=lambda x: int(x))
        for i in sub_wd:     # sub_wd = dir that named with the order of eigenvale (0, 1, 2 ...) (and breathing (100))
            SECOND_LAST_PATH_FULL = os.path.join(GULP_gout_PATH, i)
            mod_list = [x for x in os.listdir(SECOND_LAST_PATH_FULL)
                            if not os.path.isdir(os.path.join(SECOND_LAST_PATH_FULL, x)) and 'movie.xyz' not in x]
            mod_list_PATH = [os.path.join(SECOND_LAST_PATH_FULL, x) for x in mod_list]
            mod_list_PATH = sorted(mod_list_PATH, key=lambda x: x.split('/')[-1].split('_')[1].split('.')[0]) # list of mod_{lambda}.xyz
            mod_dir_list = [x for x in os.listdir(SECOND_LAST_PATH_FULL)
                                    if os.path.isdir(os.path.join(SECOND_LAST_PATH_FULL, x))] # list of sp_mod_{labmda} dir
            for j in mod_list_PATH:
                cat, an_core, an_shel, MOD_XYZ_LABEL, no_of_atoms = GULP.Convert_xyz_Gulp(j)
                FINAL_PATH_FULL = os.path.join(SECOND_LAST_PATH_FULL, 'sp_' + MOD_XYZ_LABEL)
                spliter = FINAL_PATH_FULL.split('/')[-4] + '/'
                GULP_OUT_PATH = FINAL_PATH_FULL.split(spliter)[1] + '/' + FINAL_PATH_FULL.split('/')[-1]

                if len(mod_dir_list) == 0:
                    os.mkdir(FINAL_PATH_FULL)
                    GULP.Write_Gulp(FINAL_PATH_FULL, GULP_OUT_PATH, cat, an_core, an_shel, 'y') # single-point calculation 
                    gulp_output_path = GULP.Run_Gulp(FINAL_PATH_FULL, FINAL_PATH_FULL)
                    gulp_out = FINAL_PATH_FULL + '/gulp.gout'
                    tot_energy, eigv_array, fre, FORCES_GULP = GULP.Grep_Data(gulp_out, no_of_atoms, FINAL_PATH_FULL, SP='y')
                    GULP.Ext_xyz_gulp(FINAL_PATH_FULL, no_of_atoms, eigvec_array, FORCES_GULP, tot_energy)

os.chdir('..')
del files, GULP_gout_PATH, sub_wd, mod_list, mod_list_PATH, spliter, gulp_out


#########################
# Fitting GAP potential #
#########################
cat = 'Al'
an = 'F  '
Al_atom_energy = 0.000 #-13.975817
F_atom_energy =  0.000 #-5.735796

def file_len(fname):
    with open(fname) as f:
        f = f.readlines()
    return len(f)

cwd = os.getcwd()
wd = [x for x in os.listdir('./') if os.path.isdir(x) and 'GAP' in x][0]
lists_dir = [x for x in os.listdir('./') if os.path.isdir(x)]
lists_gap = [x for x in lists_dir if 'GAP' in x]
del lists_dir

lists_gap.sort(key = lambda x: os.path.getmtime(x))
wd = lists_gap[-1]
del lists_gap

full_wd = os.path.join(cwd, wd)
ext_fpath = os.path.join(full_wd, 'ext_movie.xyz')

From = []
To = []
with open(ext_fpath, 'r') as f:
    lines = f.readlines()
for numi, i in enumerate(lines):
    if len(i) <= 10:
        From.append(numi)
        To.append(numi)

To.append(len(lines))
To = To[1:]

block = {From[i]: To[i] for i in range(len(From))}
del From, To

# Preprocessing the training/validation xyz data in 80:20 ratio with random selection
keys_list = list(block.keys())  
random.shuffle(keys_list)  

nkeys_80 = int(1.0 * len(keys_list))  # how many keys does 80% equal
keys_80 = keys_list[:nkeys_80]
keys_20 = keys_list[nkeys_80:]
del nkeys_80

# create new dicts
train_80 = {k: block[k] for k in keys_80}
valid_20 = {k: block[k] for k in keys_20}
del keys_80, keys_20

FIT_dir_path = os.path.join(full_wd, 'FIT')
os.mkdir(FIT_dir_path)
Training_xyz_path = os.path.join(FIT_dir_path, 'Training_set.xyz')
Valid_xyz_path = os.path.join(FIT_dir_path, 'Valid_set.xyz')
del FIT_dir_path

with open(Training_xyz_path, 'a') as f:
    for numi, i in enumerate(lines):
        for j in train_80.keys():
            if j <= numi < block[j]:
                f.write(i)

with open(Valid_xyz_path, 'a') as f:
    for numi, i in enumerate(lines):
        for j in valid_20.keys():
            if j <= numi < block[j]:
                f.write(i)
print(Training_xyz_path)
# Add single atoms
with open(Training_xyz_path, 'a') as f:
    f.write('1\n')
    f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 free_energy={Al_atom_energy} pbc="F F F"\n')
    f.write('Al 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n')
    f.write('1\n')
    f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 free_energy={F_atom_energy} pbc="F F F"\n')
    f.write('F 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n')





