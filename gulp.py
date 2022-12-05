""" for GAP_1_gulp.py, GAP_2_fit.py, GAP_3_vis.py"""
import os
import sys
import subprocess
import numpy as np
from colored import fg, attr  # bg

from sklearn.linear_model import LinearRegression
from Structure_Analysis import *


class GULP:
    """The class is for the preparing the GAP
    training dataset using the GULP software"""

    def __init__(self, STEP, EIGVEC, SP, DEBUG="n"):
        self.EIGVEC = EIGVEC
        self.STEP = STEP
        self.SP = SP
        self.DEBUG = DEBUG

    def _trunc(self, values, decs=0):
        """ Truncate decimal point """
        return np.trunc(values * 10**decs) / (10**decs)

    def Get_file_list(self, path, ext=".xyz", DEBUG="n"):
        """ call the list of file located in the path
        which have {ext} as an extension """

        files = [x for x in os.listdir(path) if ext in x]
        files = sorted([(path + "/" + x) for x in files])
        if DEBUG == "debug":
            print("\nDebugging mode on: Get_file_list")
            for i in files:
                print(i)
        else:
            pass
        return files

    def Label_top_str(self, xyz, DEBUG="n"):
        """ To change the name of the default KLMC xyz output
        files to rank of the structure get the rank"""

        if DEBUG == "debug":
            print("\nDebugging mode on: Label_top_str")
            print(xyz.split("-")[1].split(".xyz")[0])
        else:
            pass
        return xyz.split("-")[1].split(".xyz")[0]

    def Re_top_str(self, TOP_structures="top_structures", Extension=".xyz",
             DEBUG="n"):
        """Renaming the xyz files in top_structures to {rank}.xyz"""

        cwd = os.getcwd()
        path = f"{cwd}/{TOP_structures}/"

        try:
            xyz_orig = [x for x in os.listdir(path) if Extension in x]
        except FileNotFoundError:
            print()
            print("Cannot find the directory which have 'xyz' files")
            print()
            sys.exit()
        xyz_orig_ordered = sorted(
            xyz_orig, key=lambda x: int("".join(filter(str.isdigit, x)))
        )

        Max = xyz_orig_ordered[-1]
        Max_len = len(Max)

        change = []
        for xyz in xyz_orig_ordered:
            label = str(self.Label_top_str(xyz))

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

            rename = str(label) + ".xyz"
            change.append(rename)

            os.rename(path + "/" + xyz, path + "/" + rename)
            if DEBUG == "debug":
                print("\nDebugging mode on: Re_top_str")
                print(
                    f"{os.path.join(path, xyz)} --> \
                {os.path.join(path, rename)}"
                )

    def Convert_xyz_Gulp(self, f, DEBUG="n"):
        """grep cartesian coord from xyz file"""
        cation = []
        anion_core = []
        anion_shel = []
        ALL = []
        with open(f, "r") as coord:
            for i, line in enumerate(coord):
                if i > 1:
                    if "Al" in line:
                        c = line.replace("Al", "Al  core")
                        cation.append(c)
                        ALL.append(c)
                    if "F" in line:
                        a_core = line.replace("F", "F   core")
                        a_shel = line.replace("F", "F   shel")
                        anion_core.append(a_core)
                        anion_shel.append(a_shel)
                        ALL.append(a_core)

        anion_core = "".join(anion_core)
        anion_core = anion_core.split("\n")
        anion_core = "\n".join(anion_core)

        anion_shel = "".join(anion_shel)
        anion_shel = anion_shel.split("\n")
        anion_shel = "\n".join(anion_shel)

        cation = "".join(cation)
        cation = cation.split("\n")
        cation = "\n".join(cation)

        dest = f.split("/")[-1]
        dest = dest.split(".")[0]

        #Filter_1
        #                                            #
        # Calculating Cartesian coord eigVec, eigVal #
        #                                            #
        STR_ANAL = structure_shape(f)
        no_of_atoms, ID, ALL = STR_ANAL.load_xyz()  # load xyz file

        com = STR_ANAL.CenterofMass(no_of_atoms, ALL)
        transformed = STR_ANAL.Transformation(
            no_of_atoms, ALL, com
        )
        CarteigVal, CarteigVec, itensor = STR_ANAL.InertiaTensor(
            no_of_atoms, ALL
        )

        #                                         #
        # Calculating dipole moment (mass weight) #
        #                                         #
        atomic_Dipole, cluster_Dipole, amp_Dipole, mu = STR_ANAL.Dipole(
            no_of_atoms, ID, transformed
        )


        if DEBUG == "debug":
            print("\nDebugging mode on: Convert_xyz_Gulp")
            print(cation)
            print(anion_core)
            print()
            print("If you are using anion shell...")
            print(anion_shel)
            print()
            print("### original atomic position ###")
            print(ALL)
            print()
            print("### Center of mass of the original atomic position ###")
            print(com)
            print()
            print(
                "### Shift atomic positions to the COM \
            (my coordinate system (0, 0, 0)) ###"
            )
            print(transformed)
            print()
            print("### Inertia tensor ###")
            print(itensor)
            print()
            print(" ### Principal Axes of inertia ###")
            print(CarteigVal)
            print()
            print("### eigenvector ###")
            print(CarteigVec)
        else:
            pass

        return (
            cation,
            anion_core,
            anion_shel,
            dest,
            no_of_atoms,
            CarteigVal,
            CarteigVec,
            cluster_Dipole,
            amp_Dipole,
            mu,
        )

    def Write_Gulp(self, path, outXYZ, cation, anion_core, anion_shel, SP, DEBUG="n"):
        """Prepare GULP input files: full optimisation, single point calculation"""
        if SP == "y":
            keywords = "single eigenvectors nodens"  # shel conp eigenvectors
            with open(path + "/gulp.gin", "w") as f:
                f.write(f"{keywords}\ncartesian\n")
                f.write(cation)
                f.write(anion_core)
                # f.write(anion_shel)
                f.write("library /home/uccatka/auto/for_GAP/AlF_noC_RM\n")
                f.write("xtol opt 6.000\n")
                f.write("ftol opt 5.000\n")
                f.write("gtol opt 8.000\n")
                f.write("switch_min rfo gnorm 0.01\n")
                f.write("maxcyc 2000\n")
                f.write(f"output xyz {outXYZ}_eig\n")
                f.write(f"output drv {outXYZ}_F_out")

        if SP == "n":
            keywords = "opti conp conj prop eigenvectors nodens"
            with open(path + "/gulp.gin", "w") as f:
                f.write(f"{keywords}\ncartesian\n")
                f.write(cation)
                f.write(anion_core)
                # f.write(anion_shel)
                f.write("library /home/uccatka/auto/for_GAP/AlF_BM_RM\n")
                f.write("xtol opt 6.000\n")
                f.write("ftol opt 5.000\n")
                f.write("gtol opt 8.000\n")
                f.write("switch_min rfo gnorm 0.01\n")
                f.write("maxcyc 2000\n")
                f.write(f"output xyz {outXYZ}_eig\n")

        if DEBUG == "debug":
            print("\nDebugging mode on: Write_Gulp")
            print(f"{path}/gulp.gin")
            with open(f"{path}/gulp.gin", "r") as f:
                lines = f.readlines()
                for i in lines:
                    print(i.strip())
        else:
            pass

    def Run_Gulp(self, path_of_gulp, dest, DEBUG="n"):
        """Run GULP"""
        subprocess.check_output(
            ["/home/uccatka/software/gulp-5.1/Src/gulp", f"{dest}/gulp"]
        )
        gulp_output_path = os.path.join(path_of_gulp, "gulp.gout")
        if DEBUG == "debug":
            print("\nDebugging mode on: Run_Gulp")
            print(f"/home/uccatka/software/gulp-5.1/Src/gulp {dest}/gulp")
        else:
            pass

        return gulp_output_path

    def Grep_Data(self, gulp_output_path, no_of_atoms, dest, SP, DEBUG="n"):
        """Take essential data"""
        #                      #
        # From the [gulp.gout] #
        #                      #
        with open(gulp_output_path, "r") as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]

            From = []
            To = []
            Freedom = []
            eigval = []
            freq = []
            freq = list(range(3 * no_of_atoms))
            DUM = []
            for numi, i in enumerate(lines):
                if "Interatomic potentials     =" in i:
                    total_energy = i.split()[3]
                    if DEBUG == "debug":
                        print("\nDebugging mode on: Grep_Data")
                        print(f"Interatomic potentials     = {total_energy}")
                    else:
                        pass

                elif "Total lattice energy       = " in i:
                    if "eV" in i:
                        total_energy = i.split()[4]
                        if DEBUG == "debug":
                            print(total_energy)
                        else:
                            pass
                    else:
                        pass

                elif "Frequency   " in i:
                    From.append(numi + 7)
                    To.append(numi - 3)
                    Freedom.append(numi)
                    eigval.extend(i.split()[1:])
                    if DEBUG == "debug":
                        print(i)
                    else:
                        pass

                elif "Vibrational properties (for cluster)" in i:
                    To.append(numi - 6)
                    if DEBUG == "debug":
                        print(i)
                    else:
                        pass
                else:
                    pass
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
            arr_1 = np.stack(arr_1, axis=1)
            arr_2 = []
            split_marker = int(arr_1.shape[1] / len(Freedom))
            for numi, i in enumerate(arr_1):
                row = np.array(np.split(i, split_marker)).reshape(len(Freedom), -1, 3)
                arr_2.append(row)
            arr_2 = np.stack(arr_2, axis=1)  # .shape
            eigvec_array = arr_2.reshape(-1, no_of_atoms, 3)
            if DEBUG == "debug":
                print("Eigenvectors of the mass weighted hessian:")
                print(eigvec_array)
            else:
                pass

        #               #
        # atomic forces #
        #               #
        force_out = [x for x in os.listdir(dest) if "drv" in x]
        marker = []
        forces = []
        if SP == "y":
            if os.path.isdir(dest):
                if force_out[0] in os.listdir(dest):
                    with open(dest + "/" + force_out[0], "r") as f:
                        lines = f.readlines()
                        for numi, i in enumerate(lines):
                            if "gradients cartesian eV/Ang" in i:
                                marker.append(numi + 1)
                                marker.append(numi + no_of_atoms + 1)

                        for numj, j in enumerate(lines):
                            if numj in range(marker[0], marker[1]):
                                force = [float(x) for x in j.split()[1:]]
                                forces.append(force)

            FORCES_GULP = np.asarray(forces) * -1
            if DEBUG == "debug":
                print("Atomic forces:")
                print(FORCES_GULP)

            return total_energy, eigvec_array, freq, FORCES_GULP
        return total_energy, eigvec_array, freq, Freedom, eigval

    def Modifying_xyz(
        self,
        path,
        gulp_new_xyz,
        eigvec_array,
        freq,
        no_of_atoms,
        total_energy,
        Breath,
        DEBUG,
            ):
        """Preparing xyz configs for training dataset: optimised xyz coord+(eigvec*lambda)"""
        with open(gulp_new_xyz, "r") as f:
            lines = f.readlines()[2:]
            coord = [x.split() for x in lines]
            array = np.asarray(coord)
            coord = array[:, 1:].astype(float)
            ID = array[:, 0].astype(str)
            for numi, i in enumerate(range(len(freq))):
                if str(numi + 1) in self.EIGVEC:
                    os.mkdir(f"{path}/{str(numi+1)}")

                    # Resolution of vibrational mode
                    for j in range(-1000, 1000+self.STEP, self.STEP):
                        if j != 0:
                            mod_eigvec_array = eigvec_array[i] * (int(j) / 1000)
                            # step=10 -> [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5,...]
                            # step=30 -> [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5,
                            # ...]
                            new_coord = coord + mod_eigvec_array
                            stack = np.c_[ID, new_coord]
                            stack = stack.tolist()
                            with open(
                                f"{path}/{str(numi+1)}/mod_{str(j)}.xyz", "w"
                            ) as f:
                                f.write(str(no_of_atoms) + "\n")
                                f.write(total_energy + "\n")
                            with open(f"{path}/{str(numi+1)}/movie.xyz", "a") as f:
                                f.write(str(no_of_atoms) + "\n")
                                f.write(total_energy + "\n")

                            for k in stack:
                                new = "\t\t".join(k) + "\n"
                                with open(
                                    f"{path}/{str(numi+1)}/mod_{str(j)}.xyz", "a"
                                ) as f:
                                    f.write(new)
                                with open(f"{path}/{str(numi+1)}/movie.xyz", "a") as f:
                                    f.write(new)

                if "all" in self.EIGVEC:
                    if numi >= 6:
                        os.mkdir(f"{path}/{str(numi+1)}")
                        print(
                            f"{fg(2)} Optimised cartesian coordinates + [{numi+1} eigvec] {attr(0)}"
                        )
                        print(eigvec_array[i])
                        print()

                        # Resolution of vibrational mode
                        for j in range(-1000, 1000+self.STEP, self.STEP):
                            if j != 0:
                                mod_eigvec_array = eigvec_array[i] * (int(j) / 1000)
                                # step=10 -> [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5,...]
                                # step=30 -> [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5,
                                # ...]
                                new_coord = coord + mod_eigvec_array
                                stack = np.c_[ID, new_coord]
                                stack = stack.tolist()
                                with open(
                                    f"{path}/{str(numi+1)}/mod_{str(j)}.xyz", "w"
                                ) as f:
                                    f.write(str(no_of_atoms) + "\n")
                                    f.write(total_energy + "\n")
                                with open(f"{path}/{str(numi+1)}/movie.xyz", "a") as f:
                                    f.write(str(no_of_atoms) + "\n")
                                    f.write(total_energy + "\n")

                                for k in stack:
                                    new = "\t\t".join(k) + "\n"
                                    with open(
                                        f"{path}/{str(numi+1)}/mod_{str(j)}.xyz", "a"
                                    ) as f:
                                        f.write(new)
                                    with open(f"{path}/{str(numi+1)}/movie.xyz", "a") as f:
                                        f.write(new)

                    if DEBUG == "debug":
                        print("\nDebugging mode on: Modifying_xyz")
                        print(f"step size (λ) for eigenvectors: {self.STEP/1000}")
                        print("original coordinates:")
                        print(coord)
                        print("modified eigenvectors to add:")
                        print(mod_eigvec_array)
                        print("modified coordinates:")
                        print(new_coord)
                else:
                    pass
            ########################################################
            # Add the LM data at the end of the choosen eigvec dir #
            ########################################################
            new_coord = coord
            # new_coord = np.around((new_coord), decimals = 9)
            stack = np.c_[ID, new_coord]
            stack = stack.tolist()

            dest = [
                x
                for x in os.listdir(path)
                if os.path.isdir(os.path.join(f"{path}/{x}"))
            ]
            DEST = []
            for i in dest:
                try:
                    DEST.append(int(i))
                except ValueError:
                    pass
            DEST = sorted(DEST, key=int)

            with open(f"{path}/{str(DEST[-1])}/mod_0.xyz", "w") as f:
                f.write(str(no_of_atoms) + "\n")
                f.write(total_energy + "\n")
            with open(f"{path}/{str(DEST[-1])}/movie.xyz", "a") as f:
                f.write(str(no_of_atoms) + "\n")
                f.write(total_energy + "\n")

            for k in stack:
                new = "\t\t".join(k) + "\n"
                with open(f"{path}/{str(DEST[-1])}/mod_0.xyz", "a") as f:
                    f.write(new)
                with open(f"{path}/{str(DEST[-1])}/movie.xyz", "a") as f:
                    f.write(new)

            if DEBUG == "debug":
                print(f"{path}/{str(DEST[-1])}/mod_0.xyz")
                print(f"{path}/{str(DEST[-1])}/movie.xyz")
            else:
                pass

        print("          [movie.xyz] for the selected vibrational mode is ready")
        print(f"               [{j}] of Modifying xyz (coord + eigvec) are DONE")
        return None

    def Breathing_xyz(self, path, gulp_new_xyz, no_of_atoms, total_energy):
        """Preapring the breathing configurations: From _from/1000 to _to/1000 ratio of
        equilibrium interatomic distance e.g. 0.8 ~ 1.2"""
        _from = 800
        _to = 1200
        every_n = 10  # If step size is less than ~10 eigval of Inertia
        # tensor of principal moment of inertia cannot catch the difference
        with open(gulp_new_xyz, "r") as f:
            lines = f.readlines()[2:]
            coord = [x.split() for x in lines]
            array = np.asarray(coord)
            coord = array[:, 1:].astype(float)
            ID = array[:, 0].astype(str)

            com = coord.sum(axis=0)
            com = com / np.shape(coord)[0]
            coord_x = np.subtract(coord[:, 0], com[0], out=coord[:, 0])
            coord_y = np.subtract(coord[:, 1], com[1], out=coord[:, 1])
            coord_z = np.subtract(coord[:, 2], com[2], out=coord[:, 2])

            coord = list(zip(coord_x, coord_y, coord_z))
            coord = np.array(coord)

            os.mkdir(f"{path}/100")
            for numii, ii in enumerate(range(_from, _to, every_n)):
                # for numii, ii in enumerate(np.linspace(800, 1200, 100)):
                if ii != 0:
                    deg = round(int(ii) / 1000, 2)
                    breathing_coord = coord * deg
                    # breathing_coord = np.around((breathing_coord), 6)
                    stack = np.c_[ID, breathing_coord]
                    stack = stack.tolist()
                    with open(f"{path}/100/B_{str(ii)}.xyz", "w") as f:
                        f.write(str(no_of_atoms) + "\n")
                        f.write(total_energy + "\n")
                    with open(f"{path}/100/movie.xyz", "a") as f:
                        f.write(str(no_of_atoms) + "\n")
                        f.write(total_energy + "\n")

                    for jj in stack:
                        new_line = "\t\t".join(jj) + "\n"
                        with open(f"{path}/100/B_{str(ii)}.xyz", "a") as f:
                            f.write(new_line)
                        with open(f"{path}/100/movie.xyz", "a") as f:
                            f.write(new_line)
        print()
        print(f"              [movie.xyz] for {numii+1} Breathing structures is ready")
        print("                                       and")
        print(
            "SP Calculation for all of the MODIFIED \
        (breathing, coord+eigvec) xyz is proceeding..."
        )
        print()
        return None


    def Dist_calculator(self, coord, no_of_atoms):
        dist = []
        for i in range(no_of_atoms):
            for j in range(i+1, no_of_atoms):
                distance = np.round(np.linalg.norm(coord[i, :] - coord[j, :]), 9)
                dist.append(distance)
        return dist

    def Ext_xyz_gulp(self, FINAL_PATH_FULL, no_of_atoms, eigvec_array, FORCES, ENERGY):
        """Generate xyz (extended) file which contains cartesian
        coordinates & atomic forces"""
        xyz_name = FINAL_PATH_FULL.split("/")[-1] + "_eig.xyz"
        with open(f"{FINAL_PATH_FULL}/{xyz_name}", "r") as f:
            lines = f.readlines()[2:]
        array = [x.split() for x in lines]
        array = np.asarray(array)
        coord = array[:, 1:].astype(float)

        dist = self.Dist_calculator(coord, no_of_atoms)
        if any(item < 0.8 for item in dist):
            print()
            print()
            print("#### {xyz_name}")
            print(dist)
            print()
            return None
        else:
            print()
            print()
            print(f"{xyz_name}")
            print(dist)
            print()
            atom = array[:, 0].astype(str)
            coord_and_eigvec = coord
            atom = atom.reshape(-1, 1)
            atom_coord_and_force = np.concatenate([atom, coord_and_eigvec, FORCES], axis=1)
            atom_coord_and_force = atom_coord_and_force.tolist()

            ext_xyz = os.path.join(FINAL_PATH_FULL, "ext_gulp.xyz")

            with open(ext_xyz, "a") as f:
                f.write(str(no_of_atoms) + "\n")
                f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" ')
                f.write('Properties=species:S:1:pos:R:3:forces:R:3 ')
                f.write(f'energy={ENERGY} pbc="F F F"\n')
                for i in atom_coord_and_force:
                    new = [str(x) for x in i]
                    new = "    ".join(new) + "\n"
                    f.write(new)

            parent_wd = FINAL_PATH_FULL.split('/')[0]
            all_ext_movie = os.path.join(parent_wd, "ext_movie.xyz")
            with open(all_ext_movie, "a") as f:
                f.write(str(no_of_atoms) + "\n")
                f.write('Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" ')
                f.write('Properties=species:S:1:pos:R:3:forces:R:3 ')
                f.write('energy={ENERGY} pbc="F F F"\n')
                for i in atom_coord_and_force:
                    new = [str(x) for x in i]
                    new = "    ".join(new) + "\n"
                    f.write(new)
            return None

    def gaussian(self, x, b, sigma2):
        pi = np.arccos(-1.0)
        sigma = np.sqrt(sigma2)
        prefix = 1.0 / (sigma * np.sqrt(2.0 * pi))
        power = ((x - b) / sigma) ** 2
        power = power * (-0.5)
        prefix = 1.0
        gx = prefix * np.exp(power)
        return gx

    def RDF(self, NUMi, no_of_atoms, coord, ID, binwidth, sig2):
        """Binning the interatomic distnace"""
        # Calculate the interatomic disntaces
        all_dist = []
        het_dist = []
        homo_dist = []
        npairs_all = 0
        npairs_het = 0
        npairs_homo = 0

        all_dup_filter = []
        het_dup_filter = []
        homo_dup_filter = []
        for numi, i in enumerate(coord):
            for numj, j in enumerate(coord):
                if numi != numj and (str(numj)+str(numi) not in all_dup_filter):  # ALL
                    npairs_all += 1
                    distance = np.round(np.linalg.norm(i - j), 9)
                    all_dist.append(distance)
                    all_dup_filter.append(str(numi)+str(numj))
                if ID[numi] != ID[numj] and (str(numj)+str(numi) not in het_dup_filter):
                    npairs_het += 1
                    distance = np.round(np.linalg.norm(i - j), 9)
                    het_dist.append(distance)
                    het_dup_filter.append(str(numi)+str(numj))
                if ID[numi] == ID[numj] and numi != numj and (str(numj)+str(numi) not in homo_dup_filter):
                    npairs_homo += 1
                    distance = np.round(np.linalg.norm(i - j), 9)
                    homo_dist.append(distance)
                    homo_dup_filter.append(str(numi)+str(numj))

        # Prepare the bin
        tmp = np.ceil(max(all_dist)) + 1
        tmp = tmp / binwidth
        nbins = int(round(tmp, 0) + 1)

        num = 0.0
        opdata = [0.0]
        for i in range(nbins - 1):
            num += binwidth
            opdata.append(round(num, 2))
        return (
            nbins,
            npairs_all,
            npairs_het,
            npairs_homo,
            all_dist,
            het_dist,
            homo_dist,
            opdata,
                )

    def Dipole_filter_algo(self, no_of_atoms, all_mu):
        "Use dipole moment to detect duplicated (chiral) structures"
        half = int(len(all_mu) / 2)
        first_half = all_mu[:half]
        first_half = np.array(first_half)
        #####################################
        # Checek breathing vibrational mode #
        #####################################
        X = np.array(list(range(1, 201)))
        X = np.reshape(X, (-1, 1))
        Y = np.reshape(all_mu, (-1, 1))
        model = LinearRegression()
        model = model.fit(X, Y)
        breathing_check = model.score(X, Y)

        ###################################################
        # Check the config whether all of them are unique #
        ###################################################
        all_unique_check = abs(all_mu[0] - all_mu[-1])

        return breathing_check, all_unique_check


