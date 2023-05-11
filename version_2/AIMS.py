""" Version 2.0.0 GAP-aims interface """

import os
import sys
import numpy as np
import random
import shutil
import subprocess

from tqdm import tqdm
from colored import fg, bg, attr

from quippy.potential import Potential

import plotly.graph_objects as go

class AIMS:
    def __init__(self, STEP, EIGVEC, DEBUG):
        self.STEP = STEP
        self.EIGVEC = EIGVEC
        self.DEBUG = DEBUG

    # Not used
    def _trunc(self, value, decimal=0):
        """ Truncate decimal point"""
        return np.trunc(value*10**decimal)/(10**decimal)

    # Not used
    def GET_FILE_LIST(self, path, ext=".xyz", DEBUG='n'):
        """ Get the file list in a directory which has the extension """
        files = [os.path.join(path, x) for x in os.listdir(path) if ext in x]
        files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        if DEBUG == "debug":
            print("\nDebugging mode on: Get_file_list")
            for i in files:
                print(i)
        else: pass
        return files

    def GET_LABEL_TOP_STR(self, xyz, DEBUG="n"):
        """ Called in {CHANGE_LABEL_TOP_STR} """
        if DEBUG == "debug":
            print("\nDebugging mode on: Label_top_str")
            print(xyz.split("-")[1].split(".xyz")[0])
        else: pass
        return xyz.split("-")[1].split(".xyz")[0]

    def CHANGE_LABEL_TOP_STR(self, top_structures="top_structures", ext=".xyz", DEBUG="n"):
        """ Change the xyz file name in the {top_structures} dir"""
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
            elif Max_len == 3:
                if len(str(label)) == 1:
                    label = "00" + label
                elif len(str(label)) == 2:
                    label = "0" + label
            elif Max_len == 2:
                if len(str(label)) == 1:
                    label = "0" + label
            else: pass

            rename = f"{label}.xyz"
            old = os.path.join(path, xyz)
            new = os.path.join(path, rename)
            os.rename(old, new)
            if DEBUG == "debug":
                print("\nDebugging mode on: Re_top_str")
                print(f"{old} --> {new}")
        return None

    def PREP_CON_SUBMIT_FILES(self, final_path_full, mod_xyz_label, SP):
        """ Prepare {control.in}, {submit.sh}, and {submit_vib.sh} file for FHI-aims """
        storage = '/home/uccatka/auto/aims_auto/copy_this_for_new/'
        control_single = "control.in.single"
        control = "control.in"
        job_sub = "submit_1.sh"
        job_vib_sub = "submit_vib_1.sh"

        if SP == "y":
            shutil.copy(storage + control_single, final_path_full + "/control.in")
            with open(os.path.join(storage, job_sub), 'r') as f:
                edit = f.read().replace("target_1", final_path_full)
                edit = edit.replace("target_2", f"{mod_xyz_label}-S")
                with open(f"{final_path_full}/submit.sh", 'w') as f:
                    f.write(edit)

        elif SP == "n":
            shutil.copy(storage + control, final_path_full + "/control.in")
            with open(os.path.join(storage, job_sub), 'r') as f:
                edit = f.read().replace("target_1", final_path_full)
                edit = edit.replace("target_2", mod_xyz_label)
                with open(f"{final_path_full}/submit.sh", 'w') as f:
                    f.write(edit)

        with open(os.path.join(storage, job_vib_sub), 'r') as f:
            edit = f.read().replace("target_1", final_path_full)
            edit = edit.replace("target_2", mod_xyz_label)
            with open(f"{final_path_full}/submit_vib.sh", 'w') as f:
                f.write(edit)
        return None

    def CONVERT_XYZ_TO_GEOMETRY(self, xyz):
        """ Convert xyz file to the {geometry.in} file"""

        rank_dir = str(xyz.split('/')[-1].split('.')[0])
        final_path_full = os.path.join(os.getcwd(), rank_dir)
        last_path = final_path_full.split('/')[-1]
        os.mkdir(rank_dir)
        path_geometry = os.path.join(rank_dir, "geometry.in")
        with open(xyz, 'r') as f:
            lines = f.readlines()
        del lines[0:2]
        no_of_atoms = len(lines)
        geo = [x.split() for x in lines]
        atom = [x[0] for x in geo]
        coord = [x[1:] for x in geo]
        for i in range(len(coord)):
            coord[i].insert(0, 'atom')
            coord[i].append(atom[i])
            edit = ' '.join(coord[i])
            with open(path_geometry, 'a') as f:
                f.write(f"{edit}\n")
        return final_path_full, last_path, no_of_atoms

    def SUBMIT_AIMS_OPT_JOB(self, final_path_full, SP):
        """ Submit the FHI-aims DFT optimisation calculation job """
        if SP == "y":
            print("Submiting normal mode calculation job")
            subprocess.run(["qsub", f"{final_path_full}/submit_vib.sh"])
            print()

        elif SP == "n":
            print("Submiting optimisation calculation job")
            subprocess.run(["qsub", f"{final_path_full}/submit.sh"])
            print()
        return None

    def GREP_AIMS_OPT(self, final_path_full, no_of_atoms):
        """ Get the data to build training data for the GAP training """
        marker_force = {}
        marker_coord = {}
        force = np.empty([1, 3])
        coord = np.empty([1, 3])
        atom = []
        aims_energy = []
        aims_final_energy = []
        with open(f"{final_path_full}/aims.out", 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if '| Total energy uncorrected      :' in lines[i]:
                    aims_energy.append(lines[i].split()[-2])
                #elif '| Total energy of the DFT / Hartree-Fock s.c.f. calculation' in lines[i]:
                #    aims_energy.append(lines[i].split()[11])

                elif 'Total atomic forces (unitary forces cleaned)' in lines[i]:
                    marker_force[i+1] = i + no_of_atoms + 1

                #elif 'in the first line of geometry.in' in lines[i]:
                #    marker_coord[i+3] = i + no_of_atoms + 3
                elif 'Updated atomic structure:' in lines[i]:
                    marker_coord[i+2] = i + no_of_atoms + 2
                elif 'Final atomic structure:' in lines[i]:
                    marker_coord[i+2] = i + no_of_atoms + 2

        # going through the data in reverse order
        for k, v in reversed(marker_coord.items()):
            ATOM = []
            for j in range(len(lines)):
                if j in range(k, v):
                    atomic_coord = np.array(lines[j].split()[1:-1]).astype(float)
                    #print(atomic_coord)
                    atomic_coord = np.reshape(atomic_coord, (1, 3))
                    coord = np.append(coord, atomic_coord, axis=0)
                    ATOM.append(lines[j].split()[-1])
            atom.append(ATOM)
        atom = np.array(atom)
        atom = np.reshape(atom, (-1, no_of_atoms, 1))
        coord = coord[1:]
        coord = np.reshape(coord, (-1, no_of_atoms, 3))

        for k, v in reversed(marker_force.items()):
            for j in range(len(lines)):
                if j in range(k, v):
                    atom_force = np.array(lines[j].split()[2:]).astype(float)
                    atom_force = np.reshape(atom_force, (1, 3))
                    force = np.append(force, atom_force, axis=0)
        force = force[1:]
        force = np.reshape(force, (-1, no_of_atoms, 3))
        return aims_energy, aims_final_energy, force, coord, atom, no_of_atoms

    def GREP_AIMS_VIB(self, final_path_full, no_of_atoms):
        """ Get the data from the normal mode calculations
        to build training data for GAP training """
        with open(os.path.join(final_path_full, "geometry.in"), 'r') as f:
            lines = f.readlines()
        lines = [x for x in lines if '#' not in x]
        lines = [x.split() for x in lines]
        geo = np.array(lines)[:, 1:-1].astype(float)
        atom_label = np.array(lines)[:, -1:].tolist()
        atom_label = [' '.join(x) for x in atom_label]
        hessian_output = [os.path.join(final_path_full, x) for x in os.listdir(final_path_full) 
                        if 'hessian.' in x if '.dat' in x][0]

        with open(hessian_output, 'r') as f:
            lines = f.readlines()
        arr = np.zeros((1, len(lines)))
        for i in lines:
            i = i.split()
            i = np.array(i).astype(float)
            i = np.reshape(i, (1, len(lines)))
            arr = np.append(arr, i, axis=0)
        arr = arr[1:]

        mass_eigval, mass_eigvec = np.linalg.eig(arr)
        str_mass_eigvec = mass_eigvec.astype(str)
        with open(os.path.join(final_path_full, 'eigvec.txt'), 'w') as f:
            for i in str_mass_eigvec:
                f.write(' '.join(i) + '\n')
        for numi, i in enumerate(np.transpose(mass_eigvec)):
            eigvec = i
            eigvec = np.reshape(eigvec, (int(len(lines)/3), 3))
            os.mkdir(os.path.join(final_path_full, str(numi+1)))
            for j in range(-1000, 1000+self.STEP, self.STEP):
                j = int(j) / 1000
                mod_freq = eigvec * j
                mod_geo = geo + mod_freq
                mod_geo = np.around(mod_geo, 9)
                mod_geo = mod_geo.real
                mod_geo = mod_geo.tolist()

                dest = os.path.join(final_path_full, str(numi+1))
                with open(os.path.join(dest, 'movie.xyz'), 'a') as f:
                    f.write(f"{str(no_of_atoms)}\n")
                    f.write("\n")
                for numk, k in enumerate(mod_geo):
                    k = [str(x) for x in k]
                    k = '   '.join(k)
                    with open(os.path.join(dest, 'movie.xyz'), 'a') as f:
                        f.write(f"{atom_label[numk]}  {k}\n")
        return None


    def AIMS_PREP_EXTENDED_XYZ(self, final_path_full, aims_final_energy, aims_energy, forces, coord, atom, no_of_atoms): 
        """ Prepare extended xyz file """
        print(coord)
        print(len(aims_energy))
        for i in range(len(coord)):
            atom_coord = np.c_[atom[i], coord[i]]
            atom_coord_force = np.c_[atom_coord, force[i]]
            atom_coord_force = atom_coord_force.tolist()
            ext_xyz = os.path.join(final_path_full, 'aims_ext.xyz')
            aims_energy = aims_energy[i]
            with open(ext_xyz, 'a') as f:
                f.write(str(no_of_atoms))
                f.write('\n')
                f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties=species:S:1:pos:R:3:forces:R:3 energy={aims_energy} pbc="F F F"')
                f.write('\n')
                for i in atom_coord_force:
                    new = [str(x) for x in i]
                    new = '\t\t'.join(new) + '\n'
                    f.write(new)

            with open('the_ext_movie.xyz', 'a') as f:
                f.write(str(no_of_atoms))
                f.write('\n')
                f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"')
                f.write('Properties=species:S:1:pos:R:3:forces:R:3')
                f.write(f'energy={aims_energy} pbc="F F F"')
                f.write('\n')
                for i in atom_coord_force:
                    new = [str(x) for x in i]
                    new = '\t\t'.join(new) + '\n'
                    f.write(new)
        return None

    def AIMS_FINAL_PREP(self):
        """ Prepare training data which will be used for the GAP training """
        with open('the_ext_movie.xyz', 'r') as f:
            lines = f.readlines()
        no_of_atoms = int(lines[0])
        coord_force_lines = [x for x in lines if len(x) > 10 and "Properties" not in x]

        coord = []
        ID = []
        for i in coord_force_lines:
            line = [float(x) for x in i.split()[1:]][:3]
            coord += line
            ID.append(i.split(" ")[0])

        From = []
        To = []
        for i in range(len(lines)):
            if len(i) <= 5:
                From.append(lines[i])
                To.append(lines[i])
        To.append(len(lines))
        To = To[1:]
        block = {From[i]: To[i] for i in range(len(From))}
        del From, To

        keys_list = list(block.keys())
        random.shuffle(keys_list)
        nkeys_80 = int(1.0 * len(keys_list))
        keys_80 = keys_list[:nkeys_80]
        keys_20 = keys_list[nkeys_80:]
        del nkeys_80

        train_80 = {k: block[k] for k in keys_80}
        valid_20 = {k: block[k] for k in keys_20}
        del keys_80, valid_20

        FIT_dir_path = os.path.join(full_wd, "FIT")
        os.mkdir(FIT_dir_path)
        Training_xyz_path = os.path.join(FIT_dir_path, "Training_set.xyz")
        Valid_xyz_path = os.path.join(FIT_dir_path, "Valid_set.xyz")

        with open(Training_xyz_path, "a") as f:
            for i in range(len(lines)):
                for j in train_80.keys():
                    if j <= i < block[j]:
                        f.write(i)
        with open(Valid_xyz_path, "a") as f:
            for i in range(len(lines)):
                for j in valid_20.keys():
                    if j <= i < block[j]:
                        f.write(i)

        with open('/home/uccatka/auto/for_GAP/Al_atom/aims.out', 'r') as f:
            lines = f.readlines()
        Al_aims_energy = [x for x in lines if '| Total energy of the DFT / Hartree-Fock s.c.f. calculation' in x]

        with open('/home/uccatka/auto/for_GAP/F_atom/aims.out', 'r') as f:
            lines = f.readlines()
        F_aims_energy = [x for x in lines if '| Total energy of the DFT / Hartree-Fock s.c.f. calculation' in x]
        del lines

        with open(Training_xyz_path, 'a') as f:
            f.write('1')
            f.write('\n')
            f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"')
            f.write('Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000')
            f.wrie('free_energy={Al_aims_energy} pbc="F F F\n')
            f.write('Al 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n')
        with open(Training_xyz_path, 'a') as f:
            f.write('1')
            f.write('\n')
            f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"')
            f.write('Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000')
            f.write(f'free_energy={F_aims_energy} pbc="F F F\n')
            f.write('F 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n')
        return None

    def GAP_2b_fit(self, wd_name, cutoff, sparse):
        """ Training GAP using two-body descriptor: distance_2b"""
        os.system(
"/scratch/home/uccatka/virtualEnv/bin/gap_fit \
energy_parameter_name=energy \
force_parameter_name=forces \
do_copy_at_file=F \
sparse_separate_file=F \
gp_file=%s/FIT/GAP.xml \
at_file=%s/FIT/Training_set.xyz \
default_sigma={0.008 0.04 0 0} \
sparse_jitter=1.0e-8 \
gap={distance_2b \
cutoff=%s \
covariance_type=ard_se delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
n_sparse=%s}"
% (wd_name, wd_name, cutoff, sparse)
)
        columns = shutil.get_terminal_size().columns
        print("\nCalculate Training data using the trained GAP IP".center(columns))
        os.system(
            "/scratch/home/uccatka/virtualEnv/bin/quip E=T F=T \
        atoms_filename=%s/FIT/Training_set.xyz param_filename=%s/FIT/GAP.xml \
        | grep AT | sed 's/AT//' > %s/FIT/quip_train.xyz"
            % (wd_name, wd_name, wd_name)
        )

        print("\nCalculate Validation data using the trained GAP IP".center(columns))
        os.system(
            "/scratch/home/uccatka/virtualEnv/bin/quip E=T F=T \
        atoms_filename=%s/FIT/Valid_set.xyz param_filename=%s/FIT/GAP.xml \
        | grep AT | sed 's/AT//' > %s/FIT/quip_validate.xyz"
            % (wd_name, wd_name, wd_name)
        )

        return None


    def VIS_ESSNTIAL(self, wd_name):
        """ Identical to the {GULP.py}"""
        cwd = os.getcwd()
        wd_path = os.path.join(cwd, wd_name)
        FIT_path = os.path.join(wd_path, "FIT")
        Train_xyz_path = os.path.join(FIT_path, "Training_set.xyz")
        Valid_xyz_path = os.path.join(FIT_path, "Valid_set.xyz")
        return wd_path, FIT_path, Train_xyz_path

    def DIMER_GAP_CALC(self, FIT_path):
        """ Identical to the {GULP.py} """
        columns = shutil.get_terminal_size().columns
        print()
        print()
        dimers = [Atoms("AlF", positions=[[0, 0, 0], [x, 0, 0]]) for x in np.arange(0.0, 6.01, 0.01)]
        pot = Potential("IP GAP", param_filename="%s/GAP.xml" % (FIT_path))
        dimer_curve = []
        for dim in dimers:
            dim.set_calculator(pot)
            dimer_curve.append(dim.get_potential_energy())
        x_axis = np.array([dim.positions[1, 0] for xim in dimers])
        y_axis = np.array(dimer_curve)
        data = np.c_[x_axis, y_axis]
        output_list = data.tolist(I)
        df = pd.DataFrame(output_list, columns=["r", "Al-F(GAP)"])
        del dimers, dimer_curve, x_axis, y_axis, data, output_list

        # Al-F scaled to match with scaled F-F
        dimers_AlF_scaled = [Atoms("AlF", positions=[[0, 0, 0], [round(x / np.sqrt(3), 3), 0, 0]]) for x in np.arange(0.0, 6.01, 0.01)]
        dimer_curve_AlF_scaled = []
        for dim_AlF_scaled in dimers_AlF_scaled:
            dim_AlF_scaled.set_calculator(pot)
            dimer_curve_AlF_scaled.append(dim_AlF_scaled.get_potential_energy())
        x_axis_AlF_scaled = np.array([dim_AlF_scaled.positions[1, 0] for dim_AlF_scaled in dimers_AlF_scaled])
        y_axis_AlF_scaled = np.array(dimer_curve_AlF_scaled)
        data_AlF_scaled = np.c_[x_axis_AlF_scaled, y_axis_AlF_scaled]
        output_list_AlF_scaled = data_AlF_scaled.tolist()
        df_AlF_scaled = pd.DataFrame(output_list_AlF_scaled, columns=["r_scaled", "Al-F(GAP)_scaled"])
        del dim_AlF_scaled, dimers_AlF_scaled, dimer_curve_AlF_scaled, x_axis_AlF_scaled, y_axis_AlF_scaled, data_AlF_scaled, output_list_AlF_scaled

        print()
        print()
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
        print()
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
        """ Identical to the {GULP.py} """
        with open(Train_xyz_path, 'r') as f:
            full_lines = f.readlines()
        no_of_atoms = int(full_lines[0])
        coord_force_lines = [x for x in full_lines if len(x) > 10 and "Properties" not in x][:-2]

        coord = []
        ID = []
        for i in coord_force_lines:
            line = [float(x) for x in i.split()[1:]][:3]
            coord += line
            ID.append(i.split(" ")[0])

        From = []
        To = []
        for numi, i in enumerate(full_lines):
            if len(i) <= 10:
                From.append(numi)
                To.append(numi)
        To.append(len(full_lines))
        To = To[1:-2]
        From = From[:-2]

        block = {From[i]: To[i] for i in range(len(From))}
        del From, To
        keys_list = list(block.keys())
        random.shuffle(keys_list)
        nkeys = int(1.0 * len(keys_list))
        keys = keys_list[:nkeys]

        train = {k: block[k] for k in keys}
        del keys

        coord = np.asarray(coord)
        coord = np.reshape(coord, (len(train.keys()), no_of_atoms, 3))

        ID = np.asarray(ID)
        ID = np.reshape(ID, (len(train.keys()), no_of_atoms))

        all_het_dist = []
        all_homo_dist = []

        dist_df_all = pd.DataFrame()
        dist_df_het = pd.DataFrame()
        dist_df_homo = pd.DataFrame()
        for numi, i in enumerate(coord):
            (nbins, npairs_all, npairs_het, npairs_homo, all_dist, het_dist, homo_dist, opdata
            ) = self.RDF(no_of_atoms, i, ID[numi], binwidth)
            all_het_dist += het_dist
            all_homo_dist += homo_dist

            #dist_df_all[numi] = all_dist
            #dist_df_het[f'{numi}_het'] = het_dist
            #dist_df_homo[f'{numi}_homo'] = homo_dist

        #dist_het_df = pd.concat([dist_df_het, dist_df_homo], axis=1)
        #p = os.path.join(wd_path, "all_bin.csv")
        #dist_het_df.to_csv(p)
        return all_het_dist, all_homo_dist






'''
""" testing the class"""
if __name__ == '__main__':
    aims_energy, aims_final_energy, force, coord, atom, no_of_atoms = AIMS('0', '0', '0').GREP_AIMS_OPT('./')
    #print(aims_energy)
    AIMS('0', '0', '0').AIMS_PREP_EXTENDED_XYZ('./', aims_final_energy, aims_energy, force, coord, atom, no_of_atoms)
'''


