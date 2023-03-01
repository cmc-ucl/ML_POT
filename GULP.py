""" Version 2.0.0 """

# essential libraries
import os
import sys
import numpy as np
import random
import shutil
import subprocess
import collections
import pandas as pd

import Structure_Analysis

# for visual effects
from tqdm import tqdm
from colored import fg, bg, attr

from sklearn.metrics import mean_squared_error
# for calculations
from ase import Atoms
from quippy.potential import Potential
# for plots
import plotly.graph_objects as go

class GULP:
    def __init__(self, STEP, EIGVEC, SP, DEBUG='n'):
        self.STEP = STEP
        self.EIGVEC = EIGVEC
        self.SP = SP
        self.DEBUG = DEBUG

    def _trunc(self, value, decimal=0):
        return np.trunc(value*10**decimal)/(10**decimal)

    def GET_FILE_LIST(self, path, ext=".xyz", DEBUG="n"):
        """ """
        files = [os.path.join(path, x) for x in os.listdir(path) if ext in x]
        files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0])) 
        if DEBUG == "debug":
            print("\nDebugging mode on: Get_file_list")
            for i in files:
                print(i)
        else:
            pass
        return files

    def GET_LABEL_TOP_STR(self, xyz, DEBUG="n"):
        """ """
        if DEBUG == "debug":
            print("\nDebugging mode on: Label_top_str")
            print(xyz.split("-")[1].split(".xyz")[0])
        else:
            pass
        return xyz.split("-")[1].split(".xyz")[0]

    def CHANGE_LABEL_TOP_STR(self, top_structures="top_structures", ext=".xyz", DEBUG="n"):
        """ """
        cwd = os.getcwd()
        path = os.path.join(cwd, top_structures)

        try:
            xyz_orig = [x for x in os.listdir(path) if ext in x]
        except FileNotFoundError:
            print()
            print("Cannot find the {top_structures} directory!")
            print()
            sys.exit()
        xyz_orig_ordered = sorted(xyz_orig, key=lambda x: int("".join(filter(str.isdigit, x))))

        Max = xyz_orig_ordered[-1]
        Max_len = len(Max)

        for xyz in tqdm(xyz_orig_ordered, desc="Changing [top_structures] file labels:"):
            label = str(self.GET_LABEL_TOP_STR(xyz, DEBUG))

            if Max_len == 4:
                if len(str(label)) == 1:
                    label = "000" + label
                elif len(str(label)) == 2:
                    label = "00" + label
                elif len(str(label)) == 3:
                    label = "0" + label
            if Max_len == 3:
                if len(str(label)) == 1:
                    label = "00" + label
                elif len(str(label)) == 2:
                    label = "0" + label
            if Max_len == 2:
                if len(str(label)) == 1:
                    label = "0" + label

            #if len(label) < Max_len:
            #    print("0"*(Max_len-len(label) + label)

            rename = f"{label}.xyz"
            old = os.path.join(path, xyz)
            new = os.path.join(path, rename)
            os.rename(old, new)
            if DEBUG == "debug":
                print("\nDebugging mode on: Re_top_str")
                print(f"{old} --> {new}")
        return None

    def STRUC_PROP(self, xyz_file, no_of_atoms, ID, coord_only, DEBUG="n"):
        """ """
        # Calculate Carteisna coordinate eigenvectors, eigenvalues
        STRUC_ANAL = structure_shape(xyz_file)
        no_of_atoms, ID, coord_only = STRUC_ANAL.load_xyz()
        com = STRUC_ANAL.CenterofMAss(no_of_atoms, coord_only)
        transformed = STRUC_ANAL.Transformation(no_of_atoms, coord_only, com)
        Careteigval, Carteigvec, itensor = STRUC_ANAL.InertiaTensor(no_of_atoms, coord_only) #transformed

        # Calculate dipole momen (mass weighted)
        atomic_dipole, cluster_dipole, amp_dipole, mu = STRUC_ANAL.Dipole(no_of_atoms, ID, transformed)

        if DEBUG == "debug":
            print("### Center of mass of the original atomic position ###")
            print(com)
            print()
            print("### Shift atomic positions to the COM ###")
            print(transformed)
            print()
            print("### Inertia tensor ###")
            print(itensor)
            print()
            print(" ### Principal Axes of inertia ###")
            print(carteigVal)
            print()
            print("### eigenvector ###")
            print(carteigVec)


    def CONVERT_XYZ_TO_GULP(self, xyz_file, mod_xyz, short_filter, DEBUG):
        """ convert xyz file to GULP intput file"""
        if mod_xyz == 0:
            with open(xyz_file, "r") as coord:
                lines = coord.readlines()
                core_write, shel_write, coord_only, no_of_atoms = self.SUB_CONVERT_XYZ_TO_GULP(lines, DEBUG="n")
                dest = xyz_file.split("/")[-1]
                dest = dest.split(".")[0]
            return core_write, shel_write, dest, no_of_atoms

        elif xyz_file == 0:
            core = []
            shel = []
            coord_only_ = []
            DEST = []
            for dest, lines in tqdm(mod_xyz.items(), desc="Convert [xyz] to the [gulp] input:"):
                core_write, shel_write, coord_only, no_of_atoms = self.SUB_CONVERT_XYZ_TO_GULP(lines, DEBUG="n")

                if short_filter == "y":
                    condi = self.SHORT_FILTER(coord_only, no_of_atoms)
                    if condi == True:
                        with open("filtered_by_DIST.txt", 'a') as f:
                            f.write(f"{dest}\n{coord_only}\n\n")
                        continue
                    else: pass
                else: pass

                os.mkdir(dest)
                core.append([core_write])
                shel.append([shel_write])
                DEST.append(dest)
            return core, shel, DEST, no_of_atoms


    def SUB_CONVERT_XYZ_TO_GULP(self, lines, DEBUG="n"):
        ''' subprocess of CONVERT_XYZ_TO_GULP
        convert xyz file to {ID} {core/shel} {atomic coordination} '''
        anion_candi = ["N", "O", "F", "S", "Cl", "Se", "Br", "Te", "I", "Po", "At"]
        no_of_atoms = int(lines[0])
        lines = lines[2:]
        coord = [x.split() for x in lines]
        coord = np.asarray(coord)
        coord_only = coord[:, 1:].astype(float)
        ID = coord[:, 0]
        anion_index = [np.where(coord == x) for x in ID if x in anion_candi][0][0]
        cation_index = [np.where(coord == x) for x in ID if x not in anion_candi][0][0]

        cation_coord = np.array([coord[x, :] for x in cation_index])
        anion_coord = np.array([coord[x, :] for x in anion_index])

        cation_core = np.char.replace(cation_coord, "Al", "Al  core")
        anion_core = np.char.replace(anion_coord, "F", "F  core")
        anion_shel = np.char.replace(anion_coord, "F", "F  shel")

        core = np.vstack((cation_core, anion_core))
        shel = np.vstack((core, anion_shel))

        core_write = "".join("\t\t".join(x)+"\n" for x in core)
        shel_write = "".join("\t\t".join(x)+"\n" for x in shel)

        if DEBUG == "debug":
            print("\nDebugging mode on: Convert_xyz_Gulp")
            print("Rigid ion model")
            print(core_write)
            print()
            print("If you are using anion shell...")
            print(shel_write)
            print()
            print("### original atomic position ###")
            print(coord)
            print(coord_only)
            print()
        else:
            pass
        return (core_write, shel_write, coord_only, no_of_atoms)

    def WRITE_GULP(self, path, outXYZ, geometry, SP, DEBUG="n"):
        """ write GULP intput file (single point calculation // optimisation) """
        if SP == "y":
            keywords = "single eigenvectors nodens"
            with open(path + "/gulp.gin", "w") as f:
                f.write(f"{keywords}\n")
                f.write("cartesian\n")
                f.write(geometry)
                f.write("library /home/uccatka/auto/for_GAP/lib/AlF_noC_RM\n")
                f.write("xtol opt 6.000\n")
                f.write("ftol opt 5.000\n")
                f.write("gtol opt 8.000\n")
                f.write("switch_min rfo gnorm 0.0001\n")
                f.write("maxcyc 2000\n")
                f.write(f"output xyz {outXYZ}_eig\n")
                f.write(f"output drv {outXYZ}_F_out")

        elif SP == "n":
            keywords = "opti conp conj prop eigenvectors nodens"
            gulp_input = os.path.join(path, "gulp.gin")
            with open(gulp_input, "w") as f:
                f.write(f"{keywords}\ncartesian\n")
                f.write(geometry)
                f.write("library /home/uccatka/auto/for_GAP/lib/AlF_BM_RM\n")
                f.write("xtol opt 6.000\n")
                f.write("ftol opt 5.000\n")
                f.write("gtol opt 8.000\n")
                f.write("switch_min rfo gnorm 0.0001\n")
                f.write("maxcyc 2000\n")
                f.write(f"output xyz {outXYZ}_eig\n")
                f.write(f"output drv {outXYZ}_F_out")    ####
        else:
            pass

        if DEBUG == "debug":
            print("\nDebugging mode on: Write_Gulp")
            print(f"{path}/gulp.gin")
            with open(f"{path}/gulp.gin", "r") as f:
                lines = f.readlines()
                for i in lines:
                    print(i.strip())
        else: pass
        return None

    def RUN_GULP(self, loc_gulp_placed, DEBUG="n"):
        """ run GULP in head node """
        subprocess.run(["/home/uccatka/software/gulp-5.1/Src/gulp", f"{loc_gulp_placed}/gulp"])
        loc_gulp_placed = os.path.join(loc_gulp_placed, "gulp.gout")
        if DEBUG == "debug":
            print("Debugging mode on: RUN_GULP")
            print(f"/home/uccatka/software/gulp-5.1/Src/gulp {dest}/gulp")
        else:
            pass
        return loc_gulp_placed

    def OPEN_GULP_OUTPUT(self, loc_gulp_placed, DEBUG="n"):
        """ take all GULP output file (unprocessed) """
        gulp_gout = os.path.join(loc_gulp_placed, "gulp.gout")
        with open(gulp_gout, "r") as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        if DEBUG == "y":
            print("\nDebugging mode on: OPEN_GULP_OUTPUT")
            print(f"Open {loc_gulp_placed} to grep data")
        return lines

    def GREP_IP_ENERGY(self, lines, DEBUG="n"):
        ''' get interatomic potential energy '''
        for numi, i in enumerate(lines):
            if "Interatomic potentials     =" in i:
                IP_energy = i.split()[3]
                if DEBUG == "debug":
                    print("\nDebugging mode on: Grep_IP_ENERGY")
                    print(f"Interatomic potentials     = {IP_energy}")
                else: pass
        return IP_energy

    def GREP_TOTAL_ENERGY(self, lines, DEBUG="n"):
        ''' get total energy of the system '''
        for numi, i in enumerate(lines):
            if ("Total lattice energy       =" in i) and ("eV" in i):
                total_energy = i.split()[4]
                if DEBUG == "debug":
                    print("\nDebugging mode on: Grep_TOTAL_ENERGY")
                    print(f"Total lattice energy       = {total_energy}")
                else: pass
        return total_energy

    def GREP_FREQ(self, lines, no_of_atoms, DEBUG="n"):
        ''' get eigvenvalue and eigenvector '''
        freq_from = []
        freq_to = []
        freq_line_no = []
        freq_eigval =[]
        freq = list(range(3 * no_of_atoms))
        #cnt = 0
        for numi, i in enumerate(lines):
            if "Frequency   " in i:
                freq_from.append(numi + 7)
                freq_to.append(numi - 3)
                freq_line_no.append(numi)
                freq_eigval.extend(i.split()[1:])
                #cnt += 1
                if DEBUG == "debug":
                    print()
                else: pass

            elif "Vibrational properties (for cluster)" in i:
                for detect in range(7):
                    if 1 < len(lines[numi-detect]) < 70:
                        if "Vibrational" not in lines[numi-detect]:
                            if len(lines[numi-detect]) > 10:
                                #print(lines[numi-detect])
                                #print(numi-detect)
                                freq_to.append(numi-detect)
                                break
                #freq_to.append(numi - 6)
                if DEBUG == "debug":
                    print("\nDebugging mode on: GREP_FREQ")
                    print("\nVibrational properties (for cluster):")
                    print("i")
                else: pass
            else: pass
        freq_to = freq_to[1:]

        # Retrieve eigenvectors
        arr_1 = []
        arr_temp = []
        for numj, j in enumerate(freq_from):
            arr_temp_2 = []
            for numk, k in enumerate(lines):
                if freq_from[numj] <= numk <= freq_to[numj]:
                    a = np.array([float(x) for x in k.split()[2:]])
                    arr_1.append(a)
                    arr_temp_2.append(a)
            arr_temp.append(arr_temp_2)

        arr_temp_3 = []
        for i in arr_temp:
            arr_temp = np.stack(i, axis=1)
            for numi, i in enumerate(arr_temp):
                row = i.reshape(-1, no_of_atoms, 3)
                arr_temp_3.append(row)
        arr_temp_3 = np.stack(arr_temp_3, axis=1)
        eigvec_array = arr_temp_3.reshape(-1, no_of_atoms, 3)

        if DEBUG == "debug":
            print("\nDebugging mode on: GREP_FREQ:")
            print("\nFrequency eigenvectors:")
            print(eigvec_array)
        else: pass

        return eigvec_array, freq_line_no, freq_eigval

    def GREP_ATOMIC_FORCE(self, no_of_atoms, dest, DEBUG="n"):
        ''' Get atomic forces from the 'drv' file '''
        marker = []
        forces = []
        #if (SP == "y" and os.path.isdir(dest)):
        drv_file = [os.path.join(dest, x) for x in os.listdir(dest) if "drv" in x][0]
        with open(drv_file, "r") as f:
            lines = f.readlines()
        for numi, i in enumerate(lines):
            if "gradients cartesian eV/Ang" in i:
                marker.append(numi + 1)
                marker.append(numi + no_of_atoms + 1)

        for numj, j in enumerate(lines):
            if numj in range(marker[0], marker[1]):
                force = [float(x) for x in j.split()[1:]]
                forces.append(force)

        force_gulp = np.asarray(forces) * -1

        if DEBUG == "debug":
            print("Atomic forces:")
            print(force_gulp)
        else: pass
        return force_gulp


    def DUP_FILTER(self, All_dirs, dest, eigvec_array, freq_eigval, no_of_atoms, hashtable_e, degen, sym, DEBUG):
        if len(self.EIGVEC) != 1:
            if (degen == 1 and sym == 0):
                freq_eigval = [int(float(x)*1000)/1000 for x in freq_eigval]  # up to three decimal places
                dup_freq_eigval = [item for item, count in collections.Counter(freq_eigval).items() if count > 1]
                dup_indicies = []
                for numi, i in enumerate(dup_freq_eigval):
                    dup_freq_eigval_index = []
                    for numj, j in enumerate(freq_eigval):
                        if i == j:
                            dup_freq_eigval_index.append(numj+1)
                        else: pass

                    dup_indicies.append(dup_freq_eigval_index)  # group of degenerate

                for i in dup_indicies:
                    i = [str(x) for x in i]
                    i = ' '.join(i[1:])
                    with open("filtered_by_DEGEN.txt", 'a') as f:
                        f.write(f"{str(i)}\n")
                eigval_inquire = [int(x) for x in self.EIGVEC]
                dup_indicies_ordered = []
                EIGVAL_NEW = []
                for i in dup_indicies:
                    c = 0
                    for j in eigval_inquire:
                        if j in i:
                            c += 1
                            dup_indicies_ordered.append(j)
                            if c >= 2:      # append only one eigval among the set of degenerate
                                EIGVAL_NEW.append(str(i[0]))
                flat_dup_indicies = [item for sublist in dup_indicies for item in sublist]
                for i in eigval_inquire:
                    if i not in flat_dup_indicies:
                        EIGVAL_NEW.append(str(i))
                EIGVAL_NEW = sorted(EIGVAL_NEW, key=lambda x: int(x))

                EIGVEC_NEW = np.zeros((no_of_atoms, 3))
                for i in EIGVAL_NEW:
                    EIGVEC_NEW = np.append(EIGVEC_NEW, eigvec_array[int(i)-1, :, :], axis=0)

                EIGVEC_NEW = np.reshape(EIGVEC_NEW, (len(EIGVAL_NEW)+1, no_of_atoms, 3))
                EIGVEC_NEW = EIGVEC_NEW[1:, :, :]
                return EIGVAL_NEW, EIGVEC_NEW
        else:
            return None

        target_dirs = []
        if (degen == 0 and sym == 1):
            ''' remove duplicated structure based on the first
            and the last step of structure energy'''
            lambda_energy = pd.DataFrame(hashtable_e, columns=["rank", "mode", "lambda", 'E'])
            lambda_energy = lambda_energy.set_index(["mode"])
            index_list = list(set(lambda_energy.index.values.tolist()))
            index_list = sorted(index_list, key=int)
            #lambda_energy.to_csv('test.csv')
            min_lam = {}
            max_lam = {}
            for i in index_list:
                df = lambda_energy.loc[i]
                rank = list(set(df['rank'].tolist()))[0]
                head = df.head(1)['E'].tolist()
                tail = df.tail(1)['E'].tolist()
                max_lam[f"{rank}/{i}"] = head[0]
                min_lam[f"{rank}/{i}"] = tail[0]

            symmetric = []
            for max_key, ma in max_lam.items():
                for min_key, mi in min_lam.items():
                    if max_key == min_key:
                        if abs(float(ma) - float(mi)) < 1:
                            symmetric.append(f"{str(max_key)}")
                        else: pass
                    else: pass

            for i in symmetric:
                with open("filtered_by_SYM.txt", 'a') as f:
                    f.write(f"{i}\n")
                target = os.path.join(os.getcwd(), i)
                target_contents = [x for x in os.listdir(target) if os.path.isdir(os.path.join(target, x))]
                target_contents = sorted(target_contents, key = lambda x: int(x.split('_')[1]))
                target_contents = target_contents[int(len(target_contents)/2):]
                target_contents = [os.path.join(i, x) for x in target_contents]
                target_dirs.append(target_contents)

            target_dirs = [x for sub in target_dirs for x in sub]
            a = [x for x in All_dirs if x not in target_dirs]
            for j in a:
                raw = self.OPEN_GULP_OUTPUT(j, DEBUG)
                IP_energy = self.GREP_IP_ENERGY(raw, DEBUG)
                total_energy = self.GREP_TOTAL_ENERGY(raw, DEBUG)
                eigvec_array, freq_line_no, freq_eigval = self.GREP_FREQ(raw, no_of_atoms, DEBUG)
                force_gulp = self.GREP_ATOMIC_FORCE(no_of_atoms, j, DEBUG)
                self.PREP_EXTENDED_XYZ(j, no_of_atoms, eigvec_array, force_gulp, IP_energy)
            return #target_dirs


    def DIST_CALC_DISCRETE(self, no_of_atoms, coord, ID):
        ''' subprocess of RANDOM_MOVE_XYZ '''
        all_dist, het_dist, homo_dist = [], [], []
        all_dup_filter, het_dup_filter, homo_dup_filter = [], [], []

        for numi, i in enumerate(range(no_of_atoms)):
            for numj, j in enumerate(range(i+1, no_of_atoms)):
                distance = np.linalg.norm(coord[i, :] - coord[j, :])
                all_dist.append(distance)

                # Interatomic distance between hetero species
                if ID[i] != ID[j] and (str(i)+str(j) not in het_dup_filter):
                    distance = np.round(np.linalg.norm(coord[i, :] - coord[j, :]), 9)
                    het_dist.append(distance)
                    het_dup_filter.append(str(i)+str(j))

                # Interatomic distance between homo species
                if ID[i] == ID[j] and i != j and (str(j)+str(i) not in homo_dup_filter):
                    distance = np.round(np.linalg.norm(coord[i,:] - coord[j, :]), 9)
                    homo_dist.append(distance)
                    homo_dup_filter.append(str(i)+str(j))
        return  all_dist, het_dist, homo_dist

    def RANDOM_MOVE_XYZ(self, path, gulp_xyz, no_of_atoms, energy, DEBUG):
        ''' Preparing xyz configs for training dataset:
            random displacement on each atom of optimised xyz coord '''
        het_SHORT = 0.8
        homo_SHORT = 1.2
        STRUC_ANAL = Structure_Analysis.structure_shape(gulp_xyz)
        mod_xyz = {}
        for i in range(1):
            wd = os.path.join(gulp_xyz.split('/')[-2], str(i))
            os.mkdir(wd)
            for j in tqdm(range(-1000, 1000+self.STEP, self.STEP), desc="Randomly displace atoms"):
                if j != 0:
                    '''
                    # randomly select ONE atom and randomly displace x, y, z coordinate #################################### 
                    no_of_atoms, ID, coord_only = STRUC_ANAL.load_xyz()
                    com = STRUC_ANAL.CenterofMass(no_of_atoms, coord_only)
                    transformed = STRUC_ANAL.Transformation(no_of_atoms, coord_only, com)
                    no_of_row = np.shape(transformed)[0]
                    rand_select_atom = np.random.choice(no_of_row, replace=False) # randomly select an atom
                    rand_select_atom_coord = transformed[rand_select_atom, :]
                    deg_displace = np.random.uniform(-1, 1, (1,3)) # generate random cart. coord. in between 0 and 1
                    rand_displace = np.add(rand_select_atom_coord, deg_displace, out=None)
                    new_coord = transformed
                    new_coord[rand_select_atom, :] = rand_displace
                    all_dist, het_dist, homo_dist = self.DIST_CALC_DISCRETE(no_of_atoms, new_coord, ID) 

                    while (any(item < het_SHORT for item in het_dist) or \
                        any(item < homo_SHORT for item in homo_dist)):
                        no_of_atoms, ID, coord_only = STRUC_ANAL.load_xyz()
                        com = STRUC_ANAL.CenterofMass(no_of_atoms, coord_only)
                        transformed = STRUC_ANAL.Transformation(no_of_atoms, coord_only, com)

                        no_of_row = np.shape(transformed)[0]
                        rand_select_atom = np.random.choice(no_of_row, replace=False) # randomly select an atom
                        rand_select_atom_coord = transformed[rand_select_atom, :]
                        deg_displace = np.random.uniform(-1, 1, (1,3)) # generate random cart. coord.
                        rand_displace = np.add(rand_select_atom_coord, deg_displace)
                        new_coord = transformed
                        new_coord[rand_select_atom, :] = rand_displace
                        all_dist, het_dist, homo_dist = self.DIST_CALC_DISCRETE(no_of_atoms, new_coord, ID)
                    stack = np.c_[ID, new_coord]
                    stack = stack.tolist()
                    stack.insert(0, [str(no_of_atoms)])
                    stack.insert(1, [energy])
                    stack = ["\t\t".join(x) + "\n" for x in stack]
                    label = os.path.join(wd, f"mod_{str(j)}")
                    mod_xyz[label] = stack
                    #########################################################################################################
                    '''
                    # random displacement for ALL atoms #####################################################################
                    no_of_atoms, ID, coord_only = STRUC_ANAL.load_xyz()
                    com = STRUC_ANAL.CenterofMass(no_of_atoms, coord_only)
                    transformed = STRUC_ANAL.Transformation(no_of_atoms, coord_only, com)
                    deg_displace = np.random.uniform(-1, 1, (no_of_atoms, 3)) # generate random cart. coord. in between 0 and 1
                    rand_displace = np.add(transformed, deg_displace, out=None)
                    new_coord = rand_displace
                    all_dist, het_dist, homo_dist = self.DIST_CALC_DISCRETE(no_of_atoms, new_coord, ID)

                    het_SHORT = 0.8
                    homo_SHORT = 1.2
                    while (any(item < het_SHORT for item in het_dist) or \
                        any(item < homo_SHORT for item in homo_dist)):
                        no_of_atoms, ID, coord_only = STRUC_ANAL.load_xyz()
                        com = STRUC_ANAL.CenterofMass(no_of_atoms, coord_only)
                        transformed = STRUC_ANAL.Transformation(no_of_atoms, coord_only, com)

                        no_of_row = np.shape(transformed)[0]
                        deg_displace = np.random.uniform(-1, 1, (no_of_atoms,3)) # generate random cart. coord.
                        rand_displace = np.add(transformed, deg_displace)
                        new_coord = rand_displace
                        all_dist, het_dist, homo_dist = self.DIST_CALC_DISCRETE(no_of_atoms, new_coord, ID)

                    stack = np.c_[ID, new_coord]
                    stack = stack.tolist()
                    stack.insert(0, [str(no_of_atoms)])
                    stack.insert(1, [energy])
                    stack = ["\t\t".join(x) + "\n" for x in stack]
                    label = os.path.join(wd, f"mod_{str(j)}")
                    mod_xyz[label] = stack
                    #########################################################################################################
                    
        new_coord = np.around(coord_only, 9)
        stack = np.c_[ID, new_coord]
        stack = stack.tolist()
        stack.insert(0, [str(no_of_atoms)])
        stack.insert(1, [energy])
        stack = ["\t\t".join(x)+"\n" for x in stack]

        label = os.path.join(wd, "mod_0")
        mod_xyz[label] = stack

        return mod_xyz


    def MODIFY_XYZ(self, path, gulp_xyz, eigval, eigvec, no_of_atoms, energy, DEBUG):
        ''' Preparing xyz configs for training dataset: (optimised xyz coord)+(eigvec*lambda) '''
        with open(gulp_xyz, "r") as f:
            lines = f.readlines()[2:]
            coord = [x.split() for x in lines]
            array = np.asarray(coord)
            coord = array[:, 1:].astype(float)
            ID = array[:, 0].astype(str)

        mod_xyz = {}
        for i in eigval:
            if DEBUG == "debug":
                print("***")
                print(i)
                print(eigvec[int(i)-1])
            else: pass

            wd = os.path.join(gulp_xyz.split('/')[-2], str(i))
            try:
                os.mkdir(wd)
            except FileExistsError:
                os.remove(wd)
                os.mkdir(wd)
            for j in range(-1000, 1000+self.STEP, self.STEP):
                if j != 0:
                    mod_eigvec = eigvec[int(i)-1] * (int(j) / 1000)
                    new_coord = np.around(coord + mod_eigvec, 9)
                    stack = np.c_[ID, new_coord]
                    stack = stack.tolist()
                    stack.insert(0, [str(no_of_atoms)])
                    stack.insert(1, [energy])
                    stack = ["\t\t".join(x)+"\n" for x in stack]
                    label = os.path.join(wd, f"mod_{str(j)}")
                    mod_xyz[label] = stack

        if DEBUG == "debug":
            print("\nDebugging mode on: MODIFY_XYZ")
            print(f"step size (λ) for eigenvectors: {self.STEP/1000}")
            print("original coordinates:")
            print(coord)
            print("the modified eigenvectors:")
            print(mod_eigvec)
            print("modified coordinates (coord + mod_eigvec):")
            print(new_coord)
        else: pass

        ########################################################
        # Add the GM data at the last choosen eigvec dir #
        ########################################################
        new_coord = np.around(coord, 9)
        stack = np.c_[ID, new_coord]
        stack = stack.tolist()
        stack.insert(0, [str(no_of_atoms)])
        stack.insert(1, [energy])
        stack = ["\t\t".join(x)+"\n" for x in stack]

        label = os.path.join(wd, "mod_0")
        mod_xyz[label] = stack

        if DEBUG == "debug":
            print("\nDebugging mode on: MODIFY_XYZ")
        else: pass
        return mod_xyz

    def DIST_CALC(self, coord, no_of_atoms):
        """Calculate all pairwise interatomic distance"""
        all_dist = []
        for numi, i in enumerate(range(no_of_atoms)):
            for numj, j in enumerate(range(i+1, no_of_atoms)):
                distance = np.linalg.norm(coord[i, :] - coord[j, :])
                all_dist.append(distance)
        return all_dist

    def SHORT_FILTER(self, coord, no_of_atoms):
        DIST = 0.8
        all_dist = self.DIST_CALC(coord, no_of_atoms)
        if any(item < DIST for item in all_dist):
            return True
        else:
            return False


    def PREP_EXTENDED_XYZ(self, final_full_path, no_of_atoms, eigvec_array, forces, energy):
        """ """
        xyz_name = final_full_path.split("/")[-1] + "_eig.xyz"
        loc_gulp_xyz = os.path.join(final_full_path, xyz_name)
        with open(loc_gulp_xyz, "r") as f:
            lines = f.readlines()
        no_of_atoms = int(lines[0])
        total_energy = lines[1]

        lines = lines[2:]
        array = [x.split() for x in lines]
        array = np.asarray(array)
        coord = array[:, 1:].astype(float)

        atom = array[:, 0].astype(str)
        atom = atom.reshape(-1, 1)
        atom_coord_and_force = np.concatenate([atom, coord, forces], axis=1)
        atom_coord_and_force = atom_coord_and_force.tolist()

        ext_xyz = os.path.join(final_full_path, "ext_gulp.xyz")
        with open(ext_xyz, "a") as f:
            f.write(str(no_of_atoms) + "\n")
            f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" ')
            f.write('Properties=species:S:1:pos:R:3:forces:R:3 ')
            f.write(f'energy={energy} pbc="F F F"\n')
            for i in atom_coord_and_force:
                new = [str(x) for x in i]
                new = "    ".join(new) + "\n"
                f.write(new)

        parent_wd = os.path.join(final_full_path.split('/')[0], final_full_path.split('/')[1])
        all_ext_movie = os.path.join(parent_wd, "ext_movie.xyz")
        with open(all_ext_movie, "a") as f:
            f.write(str(no_of_atoms) + "\n")
            f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" ')
            f.write('Properties=species:S:1:pos:R:3:forces:R:3 ')
            f.write(f'energy={energy} pbc="F F F"\n')
            for i in atom_coord_and_force:
                new = [str(x) for x in i]
                new = "    ".join(new) + "\n"
                f.write(new)
        return None

    def FINAL_PREP(self):
        Al_atom_energy = 0.000  # -13.975817
        F_atom_energy = 0.000  # -5.735796

        cwd = os.getcwd()
        lists = [x for x in os.listdir(cwd) if os.path.isdir(x)]
        lists = [[os.path.join(x, y) for y in os.listdir(x)] for x in lists]
        lists = [x for sub in lists for x in sub if os.path.isdir(x)]
        try:
            lists = sorted(lists, key= lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
        except ValueError:
            pass

        try:
            lists = sorted(lists, key=lambda x: (int(x.split('/')[-1])))
        except ValueError:
            pass
        os.mkdir('FIT')
        training_set = "FIT/Training_set.xyz"
        with open(training_set, 'wb') as outf:
            for i in tqdm(lists, desc="Generating Training_set.xyz:"):
                ext_path = os.path.join(i, "ext_movie.xyz")
                if os.path.isfile(ext_path):
                    with open(ext_path, 'rb') as readf:
                        shutil.copyfileobj(readf, outf)
                else: pass


        with open("FIT/Valid_set.xyz", 'w') as f:
            f.write(' ')

        # Add single atoms
        with open(training_set, "a") as f:
            f.write("1\n")
            f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" ')
            f.write('Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 ')
            f.write(f'free_energy={Al_atom_energy} pbc="F F F"\n')
            f.write("Al 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n")
            f.write("1\n")
            f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" ')
            f.write('Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 ')
            f.write(f'free_energy={F_atom_energy} pbc="F F F"\n')
            f.write("F 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n")
        return None

    def GAP_2b_fit(self, wd_name, cutoff, sparse):
        os.system(
"/scratch/home/uccatka/virtualEnv/bin/gap_fit \
energy_parameter_name=energy \
force_parameter_name=forces \
do_copy_at_file=F \
sparse_separate_file=T \
gp_file=%s/FIT/GAP.xml \
at_file=%s/FIT/Training_set.xyz \
default_sigma={0.008 0.04 0 0} \
sparse_jitter=1.0e-8 \
gap={distance_2b \
cutoff=%s \
covariance_type=ard_se delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
n_sparse=%s}" % (wd_name, wd_name, cutoff, sparse))

        columns = shutil.get_terminal_size().columns
        print("\nCalculate Training data using the trained GAP IP".center(columns))

        # Compute Training_set.xyz with the trained GAP pot
        os.system(
            "/scratch/home/uccatka/virtualEnv/bin/quip E=T F=T \
        atoms_filename=%s/FIT/Training_set.xyz param_filename=%s/FIT/GAP.xml \
        | grep AT | sed 's/AT//' > %s/FIT/quip_train.xyz"
            % (wd_name, wd_name, wd_name)
        )

        # Compute Valid_set.xyz with the trained GAP pot
        print("\nCalculate Validation data using the trained GAP IP".center(columns))
        os.system(
            "/scratch/home/uccatka/virtualEnv/bin/quip E=T F=T \
        atoms_filename=%s/FIT/Valid_set.xyz param_filename=%s/FIT/GAP.xml \
        | grep AT | sed 's/AT//' > %s/FIT/quip_validate.xyz"
            % (wd_name, wd_name, wd_name)
        )
        return None

    def VIS_ESSENTIAL(self, wd_name):
        cwd = os.getcwd()
        wd_path = os.path.join(cwd, wd_name)
        FIT_path = os.path.join(wd_path, "FIT")
        Train_xyz_path = os.path.join(FIT_path, "Training_set.xyz")
        Valid_xyz_path = os.path.join(FIT_path, "Valid_set.xyz")
        return wd_path, FIT_path, Train_xyz_path

    def DIMER_GAP_CALC(self, FIT_path):
        columns = shutil.get_terminal_size().columns
        print()
        print(f"{fg(1)} Al-F GAP pairwise interaction {attr(0)}".center(columns))
        dimers = [Atoms("AlF", positions=[[0, 0, 0], [x, 0, 0]]) for x in np.arange(0.0, 6.01, 0.01)]
        pot = Potential("IP GAP", param_filename="%s/GAP.xml" % (FIT_path))
        dimer_curve = []
        for dim in dimers:
            dim.set_calculator(pot)
            dimer_curve.append(dim.get_potential_energy())
        x_axis = np.array([dim.positions[1, 0] for dim in dimers])
        y_axis = np.array(dimer_curve)
        data = np.c_[x_axis, y_axis]
        output_list = data.tolist()
        df = pd.DataFrame(output_list, columns=["r", "Al-F(GAP)"])
        del (dimers, dimer_curve, x_axis, y_axis, data, output_list)

        # Al-F scaled to match with scaled F-F
        dimers_AlF_scaled = [Atoms("AlF", positions=[[0, 0, 0], [round(x / np.sqrt(3), 3), 0, 0]])
            for x in np.arange(0.0, 6.01, 0.01)]
        dimer_curve_AlF_scaled = []
        for dim_AlF_scaled in dimers_AlF_scaled:
            dim_AlF_scaled.set_calculator(pot)
            dimer_curve_AlF_scaled.append(dim_AlF_scaled.get_potential_energy())
        x_axis_AlF_scaled = np.array([dim_AlF_scaled.positions[1, 0] for dim_AlF_scaled in dimers_AlF_scaled])
        y_axis_AlF_scaled = np.array(dimer_curve_AlF_scaled)
        data_AlF_scaled = np.c_[x_axis_AlF_scaled, y_axis_AlF_scaled]
        output_list_AlF_scaled = data_AlF_scaled.tolist()
        df_AlF_scaled = pd.DataFrame(output_list_AlF_scaled, columns=["r_scaled", "Al-F(GAP)_scaled"])
        del (dim_AlF_scaled, dimers_AlF_scaled, dimer_curve_AlF_scaled, x_axis_AlF_scaled, y_axis_AlF_scaled, data_AlF_scaled, output_list_AlF_scaled)

        print()
        print(f"{fg(3)} Al-Al GAP  pairwise interaction {attr(0)}".center(columns))
        dimers = [Atoms("AlAl", positions=[[0, 0, 0], [x, 0, 0]]) for x in np.arange(0.0, 6.01, 0.01)]
        dimer_curve = []
        for dim in dimers:
            dim.set_calculator(pot)
            dimer_curve.append(dim.get_potential_energy())
        x_axis = np.array([dim.positions[1, 0] for dim in dimers])
        y_axis = np.array(dimer_curve)
        data = np.c_[x_axis, y_axis]
        output_list = data.tolist()
        dff = pd.DataFrame(output_list, columns=["x2", "Al-Al(GAP)"])
        del dimers, dimer_curve, x_axis, y_axis, data, output_list

        print()
        print(f"{fg(2)} F-F GAP pairwise interaction {attr(0)}".center(columns))
        dimers = [Atoms("FF", positions=[[0, 0, 0], [x, 0, 0]]) for x in np.arange(0.0, 6.01, 0.01)]
        dimer_curve = []
        for dim in dimers:
            dim.set_calculator(pot)
            dimer_curve.append(dim.get_potential_energy())
        x_axis = np.array([dim.positions[1, 0] for dim in dimers])
        y_axis = np.array(dimer_curve)
        data = np.c_[x_axis, y_axis]
        output_list = data.tolist()
        dfff = pd.DataFrame(output_list, columns=["x3", "F-F(GAP)"])
        del dimers, dimer_curve, y_axis, data, output_list

        print()
        df = df.join(dff)
        df = df.join(dfff)
        df_dimer = df.join(df_AlF_scaled)
        df_dimer.drop(columns=["x2", "x3"], inplace=True)
        del dff, dfff, df_AlF_scaled
        return df_dimer, x_axis


    def DIST_BIN_CALC(self, wd_path, FIT_path, Train_xyz_path, binwidth, sig2):
        with open(Train_xyz_path, 'r') as f:
            full_lines = f.readlines()
        no_of_atoms = int(full_lines[0])
        #coord_force_lines = [x for x in full_lines if len(x) > 10 and "Properties" not in x][:-2]
        #cluster_counter = len([x for x in full_lines if len(x) < 5])

        check_continue, cluster_set, clusters, ID, ID_set = [], [], [], [], []
        for numi, i in enumerate(full_lines):
            if len(i) > 10 and "Properties" not in i:
                check_continue.append(numi)
                if numi - check_continue[-1] == 0:
                    clusters.append(i.split()[1:4]) # atomic coordination
                    ID.append(i.split()[0])         # atomic species
                else: pass

            else:
                if len(clusters) != 0:
                    clusters = np.array(clusters).astype(float)
                    cluster_set.append(clusters) # make nested list
                    ID_set.append(ID)            # same here
                else: pass
                ID, clusters = [], []

        cluster_set = np.array(cluster_set[:-1])

        cat_cat_dist, an_an_dist, cat_an_dist = [], [], []
        for i in range(len(cluster_set)):
            (nbins, npairs_all, npairs_cat_cat, npairs_an_an, npairs_cat_an, all_dist, \
            c_c_dist, a_a_dist, c_a_dist) = self.RDF(no_of_atoms, cluster_set[i], ID_set[i], binwidth)

            cat_cat_dist += c_c_dist # concatenate the list
            an_an_dist += a_a_dist
            cat_an_dist += c_a_dist
        return cat_an_dist, cat_cat_dist, an_an_dist

    def GET_GM_MEAN_BOND_DIST(self, wd_name):
        LM_rank = [os.path.join(wd_name, x) for x in os.listdir(wd_name) if 'FIT' not in x if os.path.isdir(os.path.join(wd_name, x))] #if 'FIT' not in x if os.path.isdir(x)]
        GM_dir_path = sorted(LM_rank, key=lambda x: int(x.split('/')[1]))[0]
        GM_xyz_path = os.path.join(GM_dir_path, '001_eig.xyz')
        with open(GM_xyz_path, 'r') as f:
            contents = f.readlines()[2:]
        contents = np.array([x.split() for x in contents])
        ID = contents[:, 0]
        coord = contents[:, 1:].astype(float)
        out = self.RDF(np.shape(coord)[0], coord, ID, None)
        c_c_dist = out[-3]
        a_a_dist = out[-2]
        c_a_dist = out[-1]

        c_c_dist = [x for x in c_c_dist if x < 3.2]
        a_a_dist = [x for x in a_a_dist if x < 3.2]
        c_a_dist = [x for x in c_a_dist if x < 2.1]

        mean_c_c = np.average(c_c_dist)
        mean_a_a = np.average(a_a_dist)
        mean_c_a = np.average(c_a_dist)

        return mean_c_c, mean_a_a, mean_c_a



    def PLOT_DIMER(self, df, wd_name, FIT_path, x_axis, cat_an_dist, cat_cat_dist, an_an_dist):
        mean_c_c, mean_a_a, mean_c_a = self.GET_GM_MEAN_BOND_DIST(wd_name)
        # E vs r
        fig = go.FigureWidget()
        # Fitted Born-Mayer potential for AlF3
        def BM(x):
            return 3760 * np.exp(-x / 0.222)

        def deriv_BM(x):
            return -1880000 * np.exp(-500 * x / 111) / 111

        def buck4(x):  # 2.73154 Å F-F distance
            if x.all() < 2.0:
                return 1127.7 * np.exp(-x / 0.2753)
            elif 2.0 <= x.all() < 2.726:
                return (
                    -3.976 * x**5
                    + 49.0486 * x**4
                    - 241.8573 * x**3
                    + 597.2668 * x**2
                    - 741.117 * x
                    + 371.2706
                )
            elif 2.726 <= x.all() < 3.031:
                return -0.361 * x**3 + 3.2362 * x**2 - 9.6271 * x + 9.4816
            elif x.all() >= 3.031:
                return -15.83 / x**6

        def Coulomb(x, cat_q, an_q):
            return 1 / (cat_q*an_q) * 14.3996439067522

        # Born-Mayer Al-F potential
        BM_color = "rgb(10, 120, 24)"
        df["Al-F(BM)"] = BM(x_axis)

        trace1 = fig.add_scatter(
            x=x_axis,
            y=BM(x_axis),
            mode="lines",
            name="Al-F Born-Mayer",
            line=dict(shape="linear", color=BM_color, dash="dot"),
        )
        # Buckingham 4-region F-F potential
        df["F-F(buck4)"] = buck4(x_axis)
        df.to_csv(f"{FIT_path}/GAP_pot_tabulated.csv", index=False)

        trace2 = fig.add_scatter(
            x=x_axis,
            y=buck4(x_axis),
            mode="lines",
            name="F-F Buck4",
            line=dict(shape="linear", color="firebrick", dash="dot"),
        )

        #####################
        ### GAP potential ###
        #####################
        # Al-F
        trace3 = fig.add_scatter(
            x=df["r"],
            y=df[df.columns[1]],
            mode="lines",
            name="Al-F GAP potential",
            line=dict(shape="linear", color=BM_color),
        )  # , secondary_y=False,)

        # Al-Al
        trace4 = fig.add_scatter(
            x=df["r"],
            y=df["Al-Al(GAP)"],
            mode="lines",
            name="Al-Al GAP potential",
            line=dict(shape="linear", color="blue"),
        )  # , secondary_y=False,)

        ## Al-F (same results but the data points' x-axis is match with F-F scaled)
        # trace_ = fig.add_scatter(x=df['r_scaled'], y=df['Al-F(GAP)_scaled'],
        #mode='lines', name='Al-F GAP potential_2',\
        #line=dict(shape='linear', color=BM_color))

        # F-F original
        trace5 = fig.add_scatter(
            x=df["r"],
            y=df["F-F(GAP)"],
            mode="lines",
            name="F-F GAP potential",
            line=dict(shape="linear", color="firebrick"),
        )  # , secondary_y=False,)

        # F-F scaled
        scaling = df["r"].tolist()
        scaling = [round(x / np.sqrt(3), 3) for x in scaling]
        trace6 = fig.add_scatter(
            x=df["r_scaled"],
            y=df["F-F(GAP)"],
            mode="lines",
            name="F-F GAP potential_scaling",
            visible="legendonly",
        )
        #df.drop(columns=["Al-F(BM)", "Al-Al(GAP)"], inplace=True)

        # Al-F + F-F
        df["GAP_sum"] = df["F-F(GAP)"] + df["Al-F(GAP)_scaled"]
        df.to_csv("df_gap_sum.csv")
        trace7 = fig.add_scatter(
            x=df["r_scaled"],
            y=df["GAP_sum"],
            mode="lines",
            name="sum(GAP)",
            visible="legendonly",
        )

        ## MSE (± 0.5 from the equilibrium interatomic distnace)
        from_point = list(df["r"]).index([x for x in df["r"] if mean_c_a - 0.5 < x][0])
        to_point = list(df["r"]).index([x for x in df["r"] if x < mean_c_a + 0.5][-1])
        MSE_catan = round(mean_squared_error(
                df["Al-F(GAP)"][from_point:to_point],
                BM(x_axis)[from_point:to_point],
                squared=False), 4)

        from_point_a = list(df["r"]).index([x for x in df["r"] if mean_a_a - 0.5 < x < mean_a_a + 0.5][0])
        to_point_a = list(df["r"]).index([x for x in df["r"] if x < mean_a_a + 0.5][-1])
        try:
            MSE_an = round(
                mean_squared_error(
                    df["F-F(GAP)"][from_point_a:to_point_a],
                    buck4(x_axis)[from_point_a:to_point_a],
                    squared=False),  4)
        except ValueError:
            MSE_an = "F-F is removed"
            pass
        with open(f"{wd_name}/MSE.txt", "w") as f:
            f.write(str(MSE_catan))
            f.write("\n")
            f.write(str(MSE_an))
        columns = shutil.get_terminal_size().columns
        print("**************************".center(columns))
        print(f"RMSE(cat-an): {MSE_catan}".center(columns))
        print(f"RMSE(an-an): {MSE_an}".center(columns))
        print("**************************".center(columns))
        print("\n")
        # Histogram above the potential figure (uppger panel)

        trace8 = fig.add_histogram(
            x=cat_an_dist, #all_het_dist,
            xbins=dict(start=0, end=6, size=0.005),
            marker_color=BM_color,
            name="cat-an dist",
            yaxis="y2")

        trace9 = fig.add_histogram(
            x=cat_cat_dist, #all_homo_dist,
            xbins=dict(start=0, end=6, size=0.005),
            marker_color="blue",
            name="cat-cat dist",
            yaxis="y2")

        trace10 = fig.add_histogram(
            x=an_an_dist,
            xbins=dict(start=0, end=6, size=0.005),
            marker_color="firebrick",
            name="an-an dist",
            yaxis="y2")

        fig.layout = dict(
            xaxis=dict(
                domain=[0, 0.8],
                range=[0,6.0],
                showgrid=False,
                zeroline=False,
                title="Interatomic distance / Å"),

            yaxis=dict(
                domain=[0, 0.8],
                range=[-100, 200],
                showgrid=False,
                zeroline=True,
                title="Potential energy / eV"),

            legend=dict(
                x=0.85,
                y=1.0,
                ),

            margin=dict(l=80, r=80, t=80, b=80),
            width=1400,
            height=800,
            hovermode="closest",
            bargap=0.8,

            xaxis2=dict(
                domain=[0.85, 1],
                showgrid=False,
                zeroline=False),

            yaxis2=dict(domain=[0.85, 1], showgrid=False, zeroline=False, title="Count"),
            font=dict(size=20))

        # Transparent vertical region where from the shortest to longest
        # interatomic distance covered by the training structure
        if len(cat_an_dist) != 0:
            fig.add_vrect(
                x0=min(cat_an_dist),
                x1=max(cat_an_dist),
                fillcolor=BM_color,
                opacity=0.4,
                layer="below",
                line_width=1)
        else: pass

        if len(cat_cat_dist) != 0:
            fig.add_vrect(
                x0=min(cat_cat_dist),
                x1=max(cat_cat_dist),
                fillcolor="blue",
                opacity=0.4,
                layer="below",
                line_width=1)
        else: pass

        if len(an_an_dist) != 0:
            fig.add_vrect(
                x0=min(an_an_dist),
                x1=max(an_an_dist),
                fillcolor="firebrick",
                opacity=0.4,
                layer="below",
                line_width=1)
        else: pass

        # Vertial-dash line to show equilibrium interatomic distances for Al-F, F-F
        fig.add_vline(x=mean_c_a, line_width=2, line_dash="dash", line_color=BM_color)
        fig.add_vline(x=mean_a_a, line_width=2, line_dash="dash", line_color="firebrick")

        fig.add_annotation(
            # size=15,
            x=0.6,
            y=0.8,
            text=f"RMSE(Al-F) = {MSE_catan} <br> RMSE(F-F) = {MSE_an}",
            xanchor="left",
            showarrow=False,
            xref="paper",
            yref="paper")
        fig.write_html(f"./{wd_name}/plot.html")


    def RDF(self, no_of_atoms, coord, ID, binwidth):
        '''Binning the interatomic distnace'''
        # Calculate the interatomic disntaces
        all_dist = []
        c_a_dist = []
        c_c_dist = []
        a_a_dist = []
        npairs_all = 0
        npairs_cat_an = 0
        npairs_cat_cat = 0
        npairs_an_an = 0

        all_dup_filter = []
        cat_an_dup_filter = []
        cat_cat_dup_filter = []
        an_an_dup_filter = []
        all_dist = []
        for i in range(len(ID)):
            for j in range(i+1, len(ID)):
                npairs_all += 1
                distance = np.linalg.norm(coord[i, :] - coord[j, :])
                all_dist.append(distance)

                # Interatomic distance between hetero species (cat-an)
                if ID[i] != ID[j] and (str(i)+str(j) not in cat_an_dup_filter):
                    npairs_cat_an += 1
                    distance = np.round(np.linalg.norm(coord[i, :] - coord[j, :]), 9)
                    c_a_dist.append(distance)
                    cat_an_dup_filter.append(str(i)+str(j))

                # Interatomic distance between homo species
                with open('/home/uccatka/auto/for_GAP/lib/anions.lib', 'r') as f:
                    anions_list = f.readlines()
                anions_list = [x.strip() for x in anions_list]
                if ID[i] in anions_list:
                    if ID[i] == ID[j] and i != j and (str(j)+str(i) not in an_an_dup_filter):
                        npairs_an_an += 1
                        distance = np.round(np.linalg.norm(coord[i,:] - coord[j, :]), 9)
                        a_a_dist.append(distance)
                        an_an_dup_filter.append(str(i)+str(j))

                else:
                    if ID[i] == ID[j] and i != j and (str(j) + str(i) not in cat_cat_dup_filter):
                        npairs_cat_cat += 1
                        distance = np.round(np.linalg.norm(coord[i,:] - coord[j, :]), 9)
                        c_c_dist.append(distance)
                        cat_cat_dup_filter.append(str(i) + str(j))
        if binwidth != None:
            # Prepare the bin
            tmp = np.ceil(max(all_dist)) + 1
            tmp = tmp / binwidth
            nbins = int(round(tmp, 0) + 1)

            #num = 0.0
            #opdata = [0.0]
            #for i in range(nbins - 1):
            #    num += binwidth
            #    opdata.append(round(num, 2))
            #print(a_a_dist)
            #print()
            #print(c_a_dist)
            #print()
            #print(c_c_dist)
            return (nbins, npairs_all, npairs_cat_cat, npairs_an_an, \
            npairs_cat_an, all_dist, c_c_dist, a_a_dist, c_a_dist)
        else:
            return (npairs_all, npairs_cat_cat, npairs_an_an, \
            npairs_cat_an, all_dist, c_c_dist, a_a_dist, c_a_dist)


