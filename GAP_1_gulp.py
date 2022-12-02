""" Preparing GULP dataset for training GAP """
import os
import sys
import collections
import pandas as pd
import random
from colored import fg, bg, attr
import numpy as np
import gulp

Arg = sys.argv
EIGVEC = Arg[1]
STEP = int(Arg[2])
RANK_from = int(Arg[3])
RANK_to = int(Arg[4])
Breath = Arg[5]
cutoff = float(Arg[6])
sparse = int(Arg[7])
dup_filter = Arg[8]
if len(Arg) == 10:
    DEBUG = Arg[9]
else:
    DEBUG = "n"

EIGVEC = EIGVEC.split()
_EIGVEC = "-".join(EIGVEC)


wd_name = f"GAP_{_EIGVEC}_{STEP}_{RANK_from}-{RANK_to}_{Breath}_{cutoff}_{sparse}"


if wd_name not in os.listdir("./"):
    os.mkdir(wd_name)

# Instantiate the class
GULP = gulp.GULP(STEP, EIGVEC, SP="set")

# Re-naming the xyz files in the top_structures
try:
    GULP.Re_top_str("top_structures", ".xyz", DEBUG)
except IndexError:
    pass

files = GULP.Get_file_list(os.getcwd() + "/top_structures", ".xyz", DEBUG)

if "ext_movie.xyz" in os.listdir("./"):
    os.remove("ext_movie.xyz")

print()
print(f"{fg(15)} {bg(5)} Preparing Data {attr(0)}")
print()


os.chdir(wd_name)

for f in files:
    lambda_energy = pd.DataFrame()
    rank = int(f.split("/")[-1].split(".")[0])
    EIGVALS = {}
    if RANK_from <= rank <= RANK_to:
        (
            cation,
            anion_core,
            anion_shel,
            DIR_IP_RANK,
            no_of_atoms,
            CarteigVal,
            CarteigVec,
            cluster_Dipole,
            amp_cluster_Dipole,
            mu,
        ) = GULP.Convert_xyz_Gulp(f, DEBUG)
        cwd = os.getcwd()
        os.mkdir(DIR_IP_RANK)
        GULP_gout_PATH = os.path.join("./", DIR_IP_RANK)  # cwd, DIR_IP_RANK)
        GULP_OUT_PATH = (
            GULP_gout_PATH.split("/")[-1] + "/" + GULP_gout_PATH.split("/")[-1]
        )
        GULP.Write_Gulp(
            DIR_IP_RANK, GULP_OUT_PATH, cation, anion_core, anion_shel, "n", DEBUG
        )
        Gulp_output_path = GULP.Run_Gulp(GULP_gout_PATH, DIR_IP_RANK)

        # CRUCIAL #
        total_energy, eigvec_array, freq, Freedom, eigval = GULP.Grep_Data(
            Gulp_output_path, no_of_atoms, DIR_IP_RANK, "n", DEBUG
        )
                                
        ## Filtering degenerate frequency (eigenvalue)
        if dup_filter == 'y':
            if len(EIGVEC) != 1:
                eigval = [float(x) for x in eigval]

                print("Eigenvalues")
                print(eigval)
                eigval_2 = [int(x*1000)/1000 for x in eigval]
                dup_eigval_2 = [item for item, count in
                                    collections.Counter(eigval_2).items() if count > 1]
                dup_indicies = []
                for numi, i in enumerate(dup_eigval_2):
                    dup_eigval_2_index = []
                    for numj, j in enumerate(eigval_2):
                        if i == j:
                            dup_eigval_2_index.append(numj+1)
                    dup_indicies.append(dup_eigval_2_index)

                print("Degenerate eiganvalues")
                print(dup_indicies)
                print()

                eigval_inquire = [int(x) for x in EIGVEC]
                dup_indicies_ordered = []
                EIGVEC_NEW = []
                c2 = 0
                for i in dup_indicies:
                    c = 0
                    for j in eigval_inquire:
                        if j in i:
                            c += 1
                            dup_indicies_ordered.append(j)
                            if c >= 2:
                                EIGVEC_NEW.append(str(i[0]))

                flat_dup_indicies = [item for sublist in dup_indicies for item in sublist]

                for i in eigval_inquire:
                    if i not in flat_dup_indicies:
                        EIGVEC_NEW.append(str(i))
                EIGVEC_NEW = sorted(EIGVEC_NEW, key=lambda x: int(x))

                #print(eigvec_array)
                EIGVEC_array = np.zeros((no_of_atoms, 3))
                for i in EIGVEC_NEW:
                    EIGVEC_array = np.append(EIGVEC_array, eigvec_array[int(i)-1, :, :], axis=0)

                EIGVEC_array = np.reshape(EIGVEC_array, (len(EIGVEC_NEW)+1, no_of_atoms, 3))
                EIGVEC_array = EIGVEC_array[1:, :, :]
                print("Picked EIGVALUE (in order) with consideration of degeneracy")
                print(EIGVEC_NEW)
                print("Picked EIGVEC_array")
                print(EIGVEC_array)
                print()

                GULP = gulp.GULP(STEP, EIGVEC_NEW, SP="set")                            ###########
                ## Degenerate filter END
            else:
                pass
                                    
        if "0" not in EIGVEC:
            GULP.Modifying_xyz(
                DIR_IP_RANK,
                GULP_gout_PATH + f"/{DIR_IP_RANK}_eig.xyz",
                eigvec_array,
                freq,
                no_of_atoms,
                total_energy,
                Breath,
                DEBUG,
            )
        else:
            pass

        if Breath == "y":
            GULP.Breathing_xyz(
                DIR_IP_RANK,
                GULP_gout_PATH + f"/{DIR_IP_RANK}_eig.xyz",
                no_of_atoms,
                total_energy,
            )
        else:
            pass

        sub_wd = [
            x for x in os.listdir(f"{GULP_gout_PATH}")
            if os.path.isdir(f"{GULP_gout_PATH}/{x}")
        ]
        sub_wd = sorted(sub_wd, key=lambda x: int(x))

        pack_hashtable_e = dict()
        for i in (sub_wd):
            # sub_wd = dir that named with the order of
            # eigenvale (0, 1, 2 ...) (and breathing (100))
            SECOND_LAST_PATH_FULL = os.path.join(GULP_gout_PATH, i)
            mod_list = [
                x
                for x in os.listdir(SECOND_LAST_PATH_FULL)
                if not os.path.isdir(os.path.join(SECOND_LAST_PATH_FULL, x))
                and "movie.xyz" not in x
            ]
            mod_list_PATH = [os.path.join(SECOND_LAST_PATH_FULL, x) for x in mod_list]
            mod_list_PATH = sorted(
                mod_list_PATH,
                key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]),
            )  # list of mod_{lambda}.xyz

            mod_dir_list = [
                x for x in os.listdir(SECOND_LAST_PATH_FULL)
                if os.path.isdir(os.path.join(SECOND_LAST_PATH_FULL, x))
            ]  # list of sp_mod_{labmda} dir
            all_mu = []
            all_eigval = []
            ###################################################
            #   Calculate dipole moment of the all clusters   #
            ###################################################
            hashtable_e = dict()

            E = []
            for numj, j in enumerate(mod_list_PATH):
                (
                    cat,
                    an_core,
                    an_shel,
                    MOD_XYZ_LABEL,
                    no_of_atoms,
                    CarteigVal,
                    CarteigVec,
                    cluster_Dipole,
                    amp_cluster_Dipole,
                    mu,
                ) = GULP.Convert_xyz_Gulp(j)
                EIGVALS[j] = CarteigVal
                FINAL_PATH_FULL = os.path.join(
                    SECOND_LAST_PATH_FULL, "sp_" + MOD_XYZ_LABEL
                )
                spliter = FINAL_PATH_FULL.split("/")[-4] + "/"
                GULP_OUT_PATH = (
                    FINAL_PATH_FULL.split(spliter)[1]
                    + "/"
                    + FINAL_PATH_FULL.split("/")[-1]
                )
                if len(mod_dir_list) == 0:
                    os.mkdir(FINAL_PATH_FULL)
                    GULP.Write_Gulp(
                        FINAL_PATH_FULL,
                        GULP_OUT_PATH,
                        cat,
                        an_core,
                        an_shel,
                        "y",
                        DEBUG,
                    )  # single-point calculation
                    gulp_output_path = GULP.Run_Gulp(
                        FINAL_PATH_FULL, FINAL_PATH_FULL, DEBUG
                    )
                    gulp_out = FINAL_PATH_FULL + "/gulp.gout"

                    try:
                        tot_energy, eigv_array, fre, FORCES_GULP = GULP.Grep_Data(
                            gulp_out, no_of_atoms, FINAL_PATH_FULL, "y", DEBUG
                        )
                    except:
                        pass

                    lambda_name = int(j.split('/')[-1].split('_')[1].split('.xyz')[0])
                    hashtable_e[lambda_name] = float(tot_energy)

                    
                    #try:
                        #########
                    if dup_filter == "n":
                        GULP.Ext_xyz_gulp(
                            FINAL_PATH_FULL,
                            no_of_atoms,
                            eigvec_array,
                            FORCES_GULP,
                            tot_energy,
                        )
                        ########
                    #except:
                    #    pass
                    

                    E.append(tot_energy)
            df_dict = pd.DataFrame.from_dict(hashtable_e, orient='index', columns=[i])
            lambda_energy = pd.concat([lambda_energy, df_dict], axis=1)

            #pack_hashtable_e[i] = hashtable_e

        ## Symmetric configuration
        # Drop columns if the component is choosen degenerate eigval
        if dup_filter == 'y':
            if len(EIGVEC) != 1:
                for i in dup_indicies_ordered:
                    for j in lambda_energy.columns:
                        if i == int(j):
                            lambda_energy = lambda_energy.drop(columns=j)
            else:
                pass

        lambda_energy = lambda_energy.drop(index=0)
        #lambda_energy.to_csv('energy_lambda.csv')

        # dictionary of dictionary which contains the structure energy
        dict_lambda_energy = lambda_energy.to_dict('index')

        # filter the symmetric configuration from eigvec
        if dup_filter == 'y':
            minus = dict_lambda_energy[list(dict_lambda_energy)[0]]
            plus = dict_lambda_energy[list(dict_lambda_energy)[-1]]
            print("Minium λ")
            print(minus)
            print("Maximum λ")
            print(plus)
            print()

            same = []
            sym = []
            for pos_key, pos in plus.items():
                for min_key, mi in minus.items():
                    if pos_key == min_key:
                        print("sym")
                        print(pos_key, min_key)
                        print(pos, mi, abs(pos-mi))
                        print()
                        if abs(pos - mi) < 1: #10**-7:
                            sym.append(f"{str(pos_key)}")
                print()
                print()
                print()
            print("Symmetric")
            print(sym)
            print()
            print()
            print()
        # symmetric config filter END
os.chdir("..")
del files, GULP_gout_PATH, sub_wd, mod_list, mod_list_PATH, spliter, gulp_out
                    

#################################################
# Preparing for GAP potential trainig data file #
#################################################


pot = os.path.join(wd_name, '001')
eigval_dir = [os.path.join(pot, x) for x in os.listdir(pot)
              if os.path.isdir(os.path.join(pot, x)) == True]
eigval_dir = sorted(eigval_dir, key=lambda x: int(x.split('/')[-1]))

#EIGVEC_NEW
#sym

for i in eigval_dir:
    sp_dir = [os.path.join(i, x) for x in os.listdir(i)
           if os.path.isdir(os.path.join(i, x))]
    sp_dir = sorted(sp_dir, key=lambda x: int(x.split('_')[-1]))
    print(i)
    #for j in EIGVEC_NEW:
   
    if dup_filter == 'y':
        if i.split('/')[-1] in sym:
            for k in sp_dir[int(round(len(sp_dir)/2))+1:]:
                print(k)
                gulp_out = os.path.join(k, "gulp.gout")
                tot_energy, eigv_array, fre, FORCES_GULP = GULP.Grep_Data(
                    gulp_out,
                    no_of_atoms,
                    k,
                    "y",
                    DEBUG
                )
                GULP.Ext_xyz_gulp(
                    k, no_of_atoms, None, FORCES_GULP, tot_energy
                )

        else:
            for l in sp_dir:
                print(l)
                gulp_out = os.path.join(l, "gulp.gout")
                tot_energy, eigv_array, fre, FORCES_GULP = GULP.Grep_Data(
                                gulp_out,
                                no_of_atoms,
                                l,
                                "y",
                                DEBUG
                            )
                GULP.Ext_xyz_gulp(
                    l, no_of_atoms, None, FORCES_GULP, tot_energy
                )
        



cat = "Al"
an = "F  "
Al_atom_energy = 0.000  # -13.975817
F_atom_energy = 0.000  # -5.735796


def file_len(fname):
    with open(fname) as f:
        contents = f.readlines()
    return len(contents)


cwd = os.getcwd()

full_wd = os.path.join(cwd, wd_name)
ext_fpath = os.path.join(full_wd, "ext_movie.xyz")

From = []
To = []
with open(ext_fpath, "r") as f:
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

nkeys_80 = int(1.0 * len(keys_list))        # Split dataset
keys_80 = keys_list[:nkeys_80]
keys_20 = keys_list[nkeys_80:]
del nkeys_80

# create new dicts
train_80 = {k: block[k] for k in keys_80}
valid_20 = {k: block[k] for k in keys_20}
del keys_80, keys_20

FIT_dir_path = os.path.join(full_wd, "FIT")
os.mkdir(FIT_dir_path)
Training_xyz_path = os.path.join(FIT_dir_path, "Training_set.xyz")
Valid_xyz_path = os.path.join(FIT_dir_path, "Valid_set.xyz")
del FIT_dir_path

with open(Training_xyz_path, "a") as f:
    for numi, i in enumerate(lines):
        for j in train_80.keys():
            if j <= numi < block[j]:
                f.write(i)

with open(Valid_xyz_path, "a") as f:
    for numi, i in enumerate(lines):
        for j in valid_20.keys():
            if j <= numi < block[j]:
                f.write(i)

# Add single atoms
with open(Training_xyz_path, "a") as f:
    f.write("1\n")
    f.write(
        f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 \
free_energy={Al_atom_energy} pbc="F F F"\n'
    )
    f.write("Al 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n")
    f.write("1\n")
    f.write(
        f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 \
free_energy={F_atom_energy} pbc="F F F"\n'
    )
    f.write("F 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n")



