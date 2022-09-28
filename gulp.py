import os, sys, time
import time
import subprocess
import re
import shutil
import numpy as np
from colored import fg, bg, attr

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations


class GULP:
    def __init__(self, STEP, FROM, TO, SP):
        self.FROM = FROM
        self.TO = TO 
        self.STEP = STEP  
        self.SP = SP



    def trunc(values, decs=0):
        return np.trunc(values*10**decs)/(10**decs)    
    
   
    
    def Get_file_list(self, path, ext='.xyz'):
        files = [x for x in os.listdir(path) if ext in x]
        files = [(path + '/' + x) for x in files]
        files.sort()
        return files

   
       
    def Label_top_str(self, xyz):
        return xyz.split('-')[1].split('.xyz')[0]
     


    def Re_top_str(self, TOP_structures='top_structures', Extension='.xyz'):
        cwd = os.getcwd()
        path = f'{cwd}/{TOP_structures}/'
       
        xyz_orig = [x for x in os.listdir(path) if Extension in x]
        xyz_orig_ordered = sorted(xyz_orig, key= lambda x: int(''.join(filter(str.isdigit, x))))
        
        #xyz_orig_ordered = sorted(xyz_orig, key=self.Label_top_str)
        
        Max = xyz_orig_ordered[-1]
        Max_len = len(Max)
    
        change = []
        for xyz in xyz_orig_ordered:
            label = str(self.Label_top_str(xyz)) 
            
            if Max_len == 4:
                if len(str(label)) == 1:
                    label = '000' + label
                elif len(str(label)) == 2:
                    a = '00' + label
                elif len(str(label)) == 3:
                    a = '0' + label
            if Max_len == 3:
                if len(str(label)) == 1:
                    a = '00' + label
                elif len(str(label)) == 2:
                    a = '0' + label
            if Max_len == 2:
                if len(str(label)) == 1:
                    a = '0' + label

            rank_xyz = xyz.split('-')[1]
            rank = rank_xyz.replace('.xyz', '')

            rename = str(label) + '.xyz'
            change.append(rename)                    
        
            os.rename(path + '/' + xyz, path + '/' + rename)



    def Convert_xyz_Gulp(self, f):
        cation = []
        anion_core = []
        anion_shel = []
        with open(f, 'r') as coord:
            for i, line in enumerate(coord):
                if i > 1:
                    if 'Al' in line:
                        c = line.replace('Al', 'Al  core')
                        cation.append(c)
                    if 'F' in line:
                        a_core = line.replace('F', 'F   core')
                        a_shel = line.replace('F', 'F   shel')
                        anion_core.append(a_core)
                        anion_shel.append(a_shel)
        no_of_anion = len(anion_core)
        anion_core = ''.join(anion_core)
        anion_core = anion_core.split('\n')
        anion_core = '\n'.join(anion_core)

        anion_shel = ''.join(anion_shel)
        anion_shel = anion_shel.split('\n')
        anion_shel = '\n'.join(anion_shel)

        no_of_cation = len(cation)
        cation = ''.join(cation)
        cation = cation.split('\n')
        cation = '\n'.join(cation)

        no_of_atoms = no_of_anion + no_of_cation

        dest = f.split('/')[-1]
        dest = dest.split('.')[0]

        return cation, anion_core, anion_shel, dest, no_of_atoms



    def Write_Gulp(self, path, outXYZ, cation, anion_core, anion_shel, SP):
        if SP == 'y':
            keywords = 'single eigenvectors nodens' #shel conp eigenvectors
            with open(path + '/gulp.gin', 'w') as f:
                f.write(f'{keywords}\ncartesian\n')
                f.write(cation)
                f.write(anion_core)
                #f.write(anion_shel)
                f.write(
                f'library /home/uccatka/auto/for_GAP/AlF_BM_RM\n\
xtol opt 6.000\n\
ftol opt 5.000\n\
gtol opt 8.000\n\
switch_min rfo gnorm 0.01\n\
maxcyc 2000\n\n\
output xyz {outXYZ}_eig\n\
output drv {outXYZ}_F_out')
        if SP == 'n': #else:
            keywords = 'opti conp conj prop eigenvectors nodens'
            with open(path + '/gulp.gin', 'w') as f:
                f.write(f'{keywords}\ncartesian\n')
                f.write(cation)
                f.write(anion_core)
                #f.write(anion_shel)
                f.write(
f'library /home/uccatka/auto/for_GAP/AlF_BM_RM\n\
xtol opt 6.000\n\
ftol opt 5.000\n\
gtol opt 8.000\n\
switch_min rfo gnorm 0.01\n\
maxcyc 2000\n\n\
output xyz {outXYZ}_eig\n')



    def Run_Gulp(self, path_of_gulp, dest):
        subprocess.check_output(['/home/uccatka/software/gulp-5.1/Src/gulp', f'{dest}/gulp'])
        gulp_output_path = os.path.join(path_of_gulp, 'gulp.gout')            
        return gulp_output_path
    


    def Grep_Data(self, gulp_output_path, no_of_atoms, dest, SP):
        #                      #
        # From the [gulp.gout] #
        #                      #
        with open(gulp_output_path, 'r') as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]

            From = []
            To = []
            Freedom = []
            freq = []
            freq = list(range(3*no_of_atoms))
            pot = []
            DUM = [] 
            for numi, i in enumerate(lines):
                if 'Interatomic potentials     =' in i:
                    total_energy = i.split()[3]
                    
                #if 'Total lattice energy       = ' in i:
                #    if 'eV' in i: 
                #        total_energy = i.split()[4] 
                    
                if 'Frequency   ' in i:
                    From.append(numi+7)
                    To.append(numi-3)
                    Freedom.append(numi)
                    
                if 'Vibrational properties (for cluster)' in i:
                    To.append(numi-6)
            To = To[1:]

            #                       #
            # Retrieve eigenvectors #
            #                       #
            
            arr_1 = []
            for numj, j in enumerate(From):
                for numk, k in enumerate(lines):
                    if From[numj] <= numk <= To[numj]:
                        a = np.array([float(x) for x in k.split()[2:]])
                        arr_1.append(a)
                        for i in a:
                            DUM.append(i)
            #print(arr_1) 
            arr_1 = np.stack(arr_1, axis=1)
            arr_2 = []
            split_marker = int(arr_1.shape[1]/len(Freedom)) 
            for numi, i in enumerate(arr_1):
                row = np.array(np.split(i, split_marker)).reshape(len(Freedom), -1, 3)
                arr_2.append(row)
            arr_2 = np.stack(arr_2, axis=1) #.shape
            eigvec_array = arr_2.reshape(-1, no_of_atoms, 3)
            #print(eigvec_array)
            
        #               #
        # atomic forces #
        #               #
        force_out = [x for x in os.listdir(dest) if 'drv' in x]
        marker = []
        forces = []
        if SP == 'y':
            if os.path.isdir(dest) == True:
                if force_out[0] in os.listdir(dest): 
                    with open(dest + '/' + force_out[0], 'r') as f:
                        lines = f.readlines()
                        for numi, i in enumerate(lines):
                            if 'gradients cartesian eV/Ang' in i:
                                marker.append(numi + 1)
                                marker.append(numi + no_of_atoms + 1)       
                        
                        for numj, j in enumerate(lines):
                            if numj in range(marker[0], marker[1]):
                                force = [float(x) for x in j.split()[1:]]
                                forces.append(force) 
            
            FORCES_GULP = np.asarray(forces)*-1
            #FORCES_GULP = np.round(FORCES_GULP, decimals=9)
            
            #print(f'# {fg(2)} forces on each individual atoms {attr(0)}')
            #np.set_printoptions(suppress=True)
            #print(gulp_output_path)
            #print(FORCES_GULP)
            return total_energy, eigvec_array, freq, FORCES_GULP
        return total_energy, eigvec_array, freq


       
    def Modifying_xyz(self, path, gulp_new_xyz, eigvec_array, freq, no_of_atoms, total_energy, Breath):
        with open(gulp_new_xyz, 'r') as f:
            lines = f.readlines()[2:]
            coord = [x.split() for x in lines]
            array = np.asarray(coord)
            coord = array[:, 1:].astype(float)
            ID = array[:, 0].astype(str)

            for numi, i in enumerate(range(len(freq))):
                if self.FROM <= numi+1 <= self.TO:
                    os.mkdir(f'{path}/{str(numi+1)}')
                    print(f'{fg(2)} Optimised cartesian coordinates + [{numi+1} eigvec] {attr(0)}')
                    print(eigvec_array[i])
                    print()

                    checker_dist = []
                    checker_ang = []
                    for j in range(-100, 101, self.STEP):                           # Resolution of vibrational mode 
                        if j != 0:
                            mod_eigvec_array = eigvec_array[i] * (int(j)/100)
                            # step=10 -> [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5,...]
                            # step=30 -> [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, ...]
                            new_coord = coord + mod_eigvec_array
                            #new_coord = np.around((new_coord), decimals = 9)

                            ###############################################################
                            # Calculate interatomic distances in between all of the atoms #
                            # and angle between anion-cation-anion                        #
                            ###############################################################
                            stack = np.c_[ID, new_coord]
                            Dist = []
                            Angle = []
                            pre_Angle = []
                            for I in stack:
                                II = I[1:].astype(float)
                                for J in stack:
                                    JJ = J[1:].astype(float)
                                    for K in stack:   
                                        KK = K[1:].astype(float)

                                        if (np.array_equal(II, JJ) == False and 
                                            np.array_equal(JJ, KK) == False and 
                                            np.array_equal(II, KK) == False):
                                            if J[0] != 'F':
                                                pre_Angle.append(np.stack((II, JJ, KK)))
                                                JI = II - JJ
                                                JK = KK - JJ
                                                cosine_angle = np.dot(JI, JK) / (np.linalg.norm(JI) * np.linalg.norm(JK))
                                                angle = np.arccos(cosine_angle)
                                                angle = np.degrees(angle)
                                                Angle.append(angle)
                                    if np.array_equal(I, J) == False:
                                        dist = np.linalg.norm(I[1:].astype(float)-J[1:].astype(float))
                                        Dist.append(dist)
                            
                            checker_dist.append(Dist)
                            checker_ang.append(Angle)
                            stack = stack.tolist()
                            with open(f'{path}/{str(numi+1)}/mod_{str(j)}.xyz', 'w') as f:
                                f.write(str(no_of_atoms) + '\n')
                                f.write(total_energy + '\n')
                            with open(f'{path}/{str(numi+1)}/movie.xyz', 'a') as f: 
                                f.write(str(no_of_atoms) + '\n')
                                f.write(total_energy + '\n')

                            for k in stack:
                                new = '\t\t'.join(k) + '\n'
                                with open(f'{path}/{str(numi+1)}/mod_{str(j)}.xyz', 'a') as f:
                                    f.write(new)
                                with open(f'{path}/{str(numi+1)}/movie.xyz', 'a') as f:  
                                    f.write(new)

                    #checker_dist = np.array(checker_dist)
                    #checker_dist_2 = np.array([np.average(x) for x in checker_dist])
                    checker_ang = np.array(checker_ang)
                    Dif = []
                    for numc, c in enumerate(checker_ang):
                        for numd, d in enumerate(checker_ang):
                            if numc != numd:
                                diff = c-d
                                Dif.append(np.around(diff, 4))
                    all_angles = np.array(Dif).reshape(1, -1)
                    filtered_angles = [x for x in all_angles[0] if abs(x) > 4]
                    if len(filtered_angles) < 5:
                        print(f"        Probably {numi+1}th eigenvector the {fg(0)}{bg(15)} rotational or transitional {attr(0)} vibrational motion")
                        print(f"         Or it could be a vibrational mode describes the {fg(0)}{bg(15)} 'breathing' {attr(0)} motion\
         {fg(0)}{bg(15)} V {attr(0)}")
                        #delete_perm = input("Would you like to delete the rot/trans/breath data ? (n/(y)): ").lower() or 'y'
                        delete_perm = 'n'
                        if delete_perm.lower() == 'y':
                            shutil.rmtree(f'{path}/{str(numi+1)}')
                        else:
                            pass
                        print()
                        print()
            ########################################################
            # Add the LM data at the end of the choosen eigvec dir #
            ########################################################
            new_coord = coord 
            new_coord = np.around((new_coord), decimals = 9)
            stack = np.c_[ID, new_coord]
            stack = stack.tolist()
            print()
            print()
            with open(f'{path}/{str(self.TO)}/mod_0.xyz', 'w') as f:
                f.write(str(no_of_atoms) + '\n')
                f.write(total_energy + '\n')
            with open(f'{path}/{str(self.TO)}/movie.xyz', 'a') as f:
                f.write(str(no_of_atoms) + '\n')
                f.write(total_energy + '\n')
            
            for k in stack:
                new = '\t\t'.join(k) + '\n'
                with open(f'{path}/{str(self.TO)}/mod_0.xyz', 'a') as f:
                    f.write(new)
                with open(f'{path}/{str(self.TO)}/movie.xyz', 'a') as f:
                    f.write(new)

        print('[movie.xyz] for the selected vibrational mode is ready')
        print('     Modifying xyz (coord + eigvec) are DONE')
        return None
         


    def Breathing_xyz(self, path, gulp_new_xyz, no_of_atoms, total_energy):
        with open(gulp_new_xyz, 'r') as f:
            lines = f.readlines()[2:]
            coord = [x.split() for x in lines]
            array = np.asarray(coord)
            coord = array[:, 1:].astype(float)
            ID = array[:, 0].astype(str)

            com = coord.sum(axis=0)
            com = com/np.shape(coord)[0]
            coord_x = np.subtract(coord[:, 0], com[0], out=coord[:, 0])
            coord_y = np.subtract(coord[:, 1], com[1], out=coord[:, 1])
            coord_z = np.subtract(coord[:, 2], com[2], out=coord[:, 2])
    
            coord = list(zip(coord_x, coord_y, coord_z))                 
            coord = np.array(coord)

            breathing_dir = f'{path}/Breathing'
            os.mkdir(f'{path}/100')
            for numii, ii in enumerate(range(800, 1200, 10)):
                if ii != 0:
                    deg = round(int(ii)/1000, 2)
                    breathing_coord = coord * deg 
                    breathing_coord = np.around((breathing_coord), 6)
                    stack = np.c_[ID, breathing_coord]
                    stack = stack.tolist()

                    with open(f'{path}/100/B_{str(ii)}.xyz', 'w') as f:
                        f.write(str(no_of_atoms) + '\n')
                        f.write(total_energy + '\n')
                    with open(f'{path}/100/movie.xyz', 'a') as f:
                        f.write(str(no_of_atoms) + '\n')
                        f.write(total_energy + '\n')
                    
                    for jj in stack:
                        new_line = '\t\t'.join(jj) + '\n' 
                        with open(f'{path}/100/B_{str(ii)}.xyz', 'a') as f:
                            f.write(new_line)
                        with open(f'{path}/100/movie.xyz', 'a') as f:
                            f.write(new_line)
        print()
        print()
        print()
        print(f'            [movie.xyz] for {numii+1} Breathing structures is ready')
        print('                                     and')
        print('SP Calculation for all of the MODIFIED (breathing, coord+eigvec) xyz is proceeding...')
        print()
        return None



    def Ext_xyz_gulp(self, FINAL_PATH_FULL, no_of_atoms, eigvec_array, FORCES, ENERGY):
        index = int(FINAL_PATH_FULL.split('/')[-2])
        drv_name = FINAL_PATH_FULL.split('/')[-1] + '_F_out.drv'
        xyz_name = FINAL_PATH_FULL.split('/')[-1] + '_eig.xyz'
        with open(f'{FINAL_PATH_FULL}/{xyz_name}', 'r') as f:
            lines = f.readlines()[2:]
        array = [x.split() for x in lines]
        array = np.asarray(array)
        coord = array[:, 1:].astype(float)
        atom = array[:, 0].astype(str)
        coord_and_eigvec = coord 
        atom = atom.reshape(-1, 1)
        atom_coord_and_force = np.concatenate([atom, coord_and_eigvec, FORCES], axis=1)
        atom_coord_and_force = atom_coord_and_force.tolist()
        
        ext_xyz = os.path.join(FINAL_PATH_FULL, 'ext_gulp.xyz')
        with open(ext_xyz, 'a') as f:
            f.write(str(no_of_atoms))
            f.write('\n')
            f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
Properties=species:S:1:pos:R:3:forces:R:3 energy={ENERGY} pbc="F F F"')
            f.write('\n')
            for i in atom_coord_and_force:
                new = [str(x) for x in i]
                new = '    '.join(new) + '\n'
                f.write(new) 

        with open('ext_movie.xyz', 'a') as f:
            f.write(str(no_of_atoms))
            f.write('\n')
            f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
Properties=species:S:1:pos:R:3:forces:R:3 energy={ENERGY} pbc="F F F"')
            f.write('\n')
            for i in atom_coord_and_force:
                new = [str(x) for x in i]
                new = '    '.join(new) + '\n'
                f.write(new)
        return None
        


    def gaussian(self, x, b, sigma2):
        pi = np.arccos(-1.0)
        sigma = np.sqrt(sigma2)
        prefix = 1.0/(sigma * np.sqrt(2.0 * pi))
        power = ((x-b)/sigma)**2
        power = power * (-0.5)
        prefix = 1.0
        gx = prefix * np.exp(power)
        return gx
 
 
   
    def RDF(self, NUMi, no_of_atoms, coord, ID, binwidth, sig2):
        # Calculate the interatomic disntaces
        all_dist = []
        het_dist = []
        homo_dist = []

        a=0
        npairs_all = 0
        npairs_het = 0
        npairs_homo = 0
        for numi, i in enumerate(coord):
            for numj, j in enumerate(coord):
                if numi != numj:            # ALL
                    npairs_all += 1
                    distance = np.round(np.linalg.norm(i-j), 6)
                    all_dist.append(distance)
                
                if ID[numi] != ID[numj]:
                    npairs_het += 1
                    distance = np.round(np.linalg.norm(i-j), 6)
                    het_dist.append(distance)
                
                if ID[numi] == ID[numj] and numi != numj:
                    npairs_homo += 1
                    distance = np.round(np.linalg.norm(i-j), 6)
                    homo_dist.append(distance)
        
        # Prepare the bin
        tmp = np.ceil(max(all_dist)) + 1
        tmp = tmp/binwidth
        nbins = int(round(tmp, 0) + 1)
        
        num = 0.0
        opdata = [0.0]
        for i in range(nbins-1):
            num += binwidth
            opdata.append(round(num, 2))
        return nbins, npairs_all, npairs_het, npairs_homo, all_dist, het_dist, homo_dist, opdata





