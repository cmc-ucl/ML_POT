import os
import numpy as np
from time import time

import quippy
from quippy.potential import Potential

import ase
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.optimize import BFGS
from ase.io import read, write


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec

from ase.visualize import view

# function to calculate the mean absolute error (MAE)
def get_mae(test_ats, calc):
    test_set = read(test_ats, ":")
    dft_e = []
    dft_f = []
    model_e = []
    model_f = []
    for at in test_set:
        # Don't test on the isolated atoms
        if len(at) == 1:
            continue
        dft_e.append(at.get_potential_energy())
        dft_f.append(at.get_forces())
        at.set_calculator(calc)
        model_e.append(at.get_potential_energy())
        model_f.append(at.get_forces())
    mae_e = 1e3 * np.mean(np.abs(dft_e - np.array(model_e)))
    mae_f = 1e3 * np.mean(np.abs(dft_f - np.array(model_f)))
    print("Energy MAE is {0:.2f} meV".format(mae_e))
    print("Force MAE is {0:.2f} meV / A".format(mae_f))
    return mae_e, mae_f



abspath = os.path.abspath('./')

one = os.path.join(abspath, 'collect_100samples/GAP_0-0_10_1-1_y_3.0_100_800-1000/FIT')
one_train = os.path.join(one, 'Training_set.xyz')
one_read = read(one_train, ":")

two = os.path.join(abspath, 'collect_100samples/GAP_0-0_10_1-1_y_3.0_100_800-1500/FIT')
two_train = os.path.join(two, 'Training_set.xyz')
two_read = read(two_train, ":")

three = os.path.join(abspath, 'collect_100samples/GAP_0-0_10_1-1_y_3.0_100_800-2000/FIT')
three_train = os.path.join(three, 'Training_set.xyz')
three_read = read(three_train, ":")

#test = os.path.join(abspath, 'GAP_0-0_10_1-1_y_3.0_100/FIT/Training_set.xyz')
universal_test = os.path.join(abspath, 'GAP_7-9_10_1-1_n_3.0_100/FIT/Training_set.xyz') 
universal_test_read = read(universal_test, ":")



"""
####################################################
# Show the energy distribution of 800_2000 (three) # 
####################################################
energies = []
for i, at in enumerate(three_read):
    energies.append(at.get_potential_energy())

# Also plot the distribution of energies
fig = plt.figure(figsize=(8, 4.0))
gs = gridspec.GridSpec(1, 1)

ax1 = plt.subplot(gs[0, 0])

ax1.hist(energies, bins=15)
ax1.set_xlabel("Potential energy / eV")
ax1.set_ylabel("Count in dataset")

plt.tight_layout()



#############################################################################################
# Evaluating each 2b descriptor with the universal_test (data of 7~12 eigvec, no breathing) #
#############################################################################################
print()
print()
print("#################### 2b ####################")
print()
Two_B_E = []
Two_B_F = []

calculator = Potential(param_filename=os.path.join(one, 'GAP.xml'))
print("Train error:")
e, f = get_mae(one_train, calculator)
print("Test error:")
e, f = get_mae(universal_test, calculator)
print()
Two_B_E.append(e)
Two_B_F.append(f)

calculator = Potential(param_filename=os.path.join(two, 'GAP.xml'))
print()
print("Train error:")
e, f = get_mae(two_train, calculator)
print("Test error:")
e, f = get_mae(universal_test, calculator)
print()
Two_B_E.append(e)
Two_B_F.append(f)


calculator = Potential(param_filename=os.path.join(three, 'GAP.xml'))
print()
print("Train error:")
e, f = get_mae(three_train, calculator)
print("Test error:")
e, f = get_mae(universal_test, calculator)
print()
Two_B_E.append(e)
Two_B_F.append(f)




print()
print()
print("#################### SOAP ####################")
SOAP_E = []
SOAP_F = []

'''
os.system('/scratch/home/uccatka/virtualEnv/bin/gap_fit \
atoms_filename=%s \
gap={distance_2b \
cutoff=3.0 \
n_sparse=100 \
covariance_type=ard_se \
delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
add_species=T : \
soap l_max=4 \
n_max=8 \
atom_sigma=0.5 \
cutoff=3.5 \
radial_scaling=-0.5 \
cutoff_transition_width=1.0 \
central_weight=1.0 \
n_sparse=150 \
delta=0.1 \
covariance_type=dot_product \
zeta=2 \
sparse_method=cur_points} \
config_type_kernel_regularisation=\
{isolated_atom:0.0001:0.0:0.0:0.0} default_sigma={0.002 0.02 0 0} \
energy_parameter_name=energy \
force_parameter_name=forces \
sparse_jitter=1.0e-8 \
do_copy_at_file=F \
sparse_separate_file=T \
gp_file=%s' % (one_train, os.path.join(one, 'GAP_soap.xml')))
'''


calculator = Potential(param_filename=os.path.join(one, 'GAP_soap.xml'))
print()
print("Train error:")
e, f = get_mae(one_train, calculator)
print("Test error:")
e, f = get_mae(universal_test, calculator)
print()
SOAP_E.append(e)
SOAP_F.append(f)

'''
os.system('/scratch/home/uccatka/virtualEnv/bin/gap_fit \
atoms_filename=%s \
gap={distance_2b \
cutoff=3.0 \
n_sparse=100 \
covariance_type=ard_se \
delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
add_species=T : \
soap l_max=4 \
n_max=8 \
atom_sigma=0.5 \
cutoff=3.5 \
radial_scaling=-0.5 \
cutoff_transition_width=1.0 \
central_weight=1.0 \
n_sparse=150 \
delta=0.1 \
covariance_type=dot_product \
zeta=2 \
sparse_method=cur_points} \
config_type_kernel_regularisation=\
{isolated_atom:0.0001:0.0:0.0:0.0} default_sigma={0.002 0.02 0 0} \
energy_parameter_name=energy \
force_parameter_name=forces \
sparse_jitter=1.0e-8 \
do_copy_at_file=F \
sparse_separate_file=T \
gp_file=%s' % (two_train, os.path.join(two, 'GAP_soap.xml')))
'''

calculator = Potential(param_filename=os.path.join(two, 'GAP_soap.xml'))
print()
print("Train error:")
e, f = get_mae(two_train, calculator)
print("Test error:")
e, f = get_mae(universal_test, calculator)
print()
SOAP_E.append(e)
SOAP_F.append(f)

'''
os.system('/scratch/home/uccatka/virtualEnv/bin/gap_fit \
atoms_filename=%s \
gap={distance_2b \
cutoff=3.0 \
n_sparse=100 \
covariance_type=ard_se \
delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
add_species=T : \
soap l_max=4 \
n_max=8 \
atom_sigma=0.5 \
cutoff=3.5 \
radial_scaling=-0.5 \
cutoff_transition_width=1.0 \
central_weight=1.0 \
n_sparse=150 \
delta=0.1 \
covariance_type=dot_product \
zeta=2 \
sparse_method=cur_points} \
config_type_kernel_regularisation=\
{isolated_atom:0.0001:0.0:0.0:0.0} default_sigma={0.002 0.02 0 0} \
energy_parameter_name=energy \
force_parameter_name=forces \
sparse_jitter=1.0e-8 \
do_copy_at_file=F \
sparse_separate_file=T \
gp_file=%s' % (three_train, os.path.join(three, 'GAP_soap.xml')))
'''


calculator = Potential(param_filename=os.path.join(three, 'GAP_soap.xml'))
print()
print("Train error:")
e, f = get_mae(three_train, calculator)
print("Test error:")
e, f = get_mae(universal_test, calculator)
print()
SOAP_E.append(e)
SOAP_F.append(f)


#######################################
# learning curve plot for 2B and SOAP #
#######################################
fig = plt.figure(figsize=(8, 4.0))
gs = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(gs[0, 0])
ax1.set_xscale("log")
ax1.plot([50, 100, 200], Two_B_E, "o-b", label="2-body GAP")
ax1.plot([50, 100, 200], SOAP_E, "o-r", label="SOAP GAP")
ax1.set_xlabel("# of training configs")
ax1.set_ylabel("Energy MAE / meV")
ax1.set_xticks([50, 100, 200])
ax1.xaxis.set_major_formatter(ScalarFormatter())


ax2 = plt.subplot(gs[0, 1])
ax2.set_xscale("log")
ax2.plot([50, 100, 200], Two_B_F, "o-b", label="2-body GAP")
ax2.plot([50, 100, 200], SOAP_F, "o-r", label="SOAP GAP")
ax2.set_xlabel("# of training configs")
ax2.set_ylabel("Force MAE / meV")
ax2.set_xticks([50, 100, 200])
ax2.xaxis.set_major_formatter(ScalarFormatter())

ax2.legend()

plt.tight_layout()
plt.savefig('EnergyMAEvsNoConfig.png')






# Do a fit without the E0. - just to see. Then the reference energy is the average energy per atom

# Now create three training sets and a test set 
print()
print()
print("#################### w/o E0 ####################")

'''
os.system('gap_fit \
atoms_filename=%s \
gap=\
{distance_2b \
cutoff=3.0 \
n_sparse=100 \
covariance_type=ard_se \
delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
add_species=T : \
soap \
l_max=4 \
n_max=8 \
atom_sigma=0.5 \
cutoff=3.5 \
radial_scaling=-0.5 \
cutoff_transition_width=1.0 \
central_weight=1.0 \
n_sparse=70 \
delta=0.1 \
covariance_type=dot_product \
zeta=2 sparse_method=cur_points} \
e0_method=average \
default_sigma={0.002 0.02 0 0} \
energy_parameter_name=energy \
force_parameter_name=forces \
sparse_jitter=1.0e-8 \
do_copy_at_file=F \
sparse_separate_file=T \
gp_file=%s' \
% (os.path.join(three, 'avgE0.xyz'), os.path.join(three, 'GAP_soap_avgE0.xml')))
'''


avg0 = os.path.join(three, 'avgE0.xyz')
calculator = Potential(param_filename=os.path.join(three, 'GAP_soap_avgE0.xml'))
print("Train error:")
e, f = get_mae(three_train, calculator)
print("Test error:")
e, f = get_mae(universal_test, calculator)





######################################################################
# PLOT dimer curves of GAP 2b, 2b-SOAP, and SOAP of 800_2000 (three) # 
######################################################################
print('PLOT 2B curves of both')
def get_dimer(elements, calculator):
    Es = []
    Ds = []
    for i in range(100):
        dist = 0.2+0.05*i
        at = ase.Atoms(elements, positions=[[dist, 0, 0], [0, 0, 0]], cell=[50, 50, 50])
        at.pbc = False
        at.set_calculator(calculator)
        Ds.append(dist)
        Es.append(at.get_potential_energy())
    return Ds, Es

gap_2B = Potential(param_filename=os.path.join(three, 'GAP.xml'))
_, FF_gap_2B = get_dimer("FF", gap_2B)
_, AlF_gap_2B = get_dimer("AlF", gap_2B)
_, AlAl_gap_2B = get_dimer("AlAl", gap_2B)
gap_soap = Potential(param_filename=os.path.join(three, 'GAP_soap.xml'))
_, FF_gap_soap = get_dimer("FF", gap_soap)
_, AlF_gap_soap = get_dimer("AlF", gap_soap)
_, AlAl_gap_soap = get_dimer("AlAl", gap_soap)

print('soap_2B_3')
gap_soap_2B = Potential(param_filename=os.path.join(three, 'GAP_soap.xml'), calc_args="only_descriptor=3")
_, FF_gap_soap_2B = get_dimer("FF", gap_soap_2B)
print('soap_2B_2')
gap_soap_2B = Potential(param_filename=os.path.join(three, 'GAP_soap.xml'), calc_args="only_descriptor=2")
_, AlF_gap_soap_2B = get_dimer("AlF", gap_soap_2B)
print('soap_2B_1')
gap_soap_2B = Potential(param_filename=os.path.join(three, 'GAP_soap.xml'), calc_args="only_descriptor=1")
_, AlAl_gap_soap_2B = get_dimer("AlAl", gap_soap_2B)
print('soap_2B_w/o_E0')
gap_soap_avgE0 = Potential(param_filename=os.path.join(three, 'GAP_soap_avgE0.xml'))
_, FF_gap_soap_avgE0 = get_dimer("FF", gap_soap_avgE0)
_, AlF_gap_soap_avgE0 = get_dimer("AlF", gap_soap_avgE0)
_, AlAl_gap_soap_avgE0 = get_dimer("AlAl", gap_soap_avgE0)

# Dimer curves
fig = plt.figure(figsize=(12, 4.2))
gs = gridspec.GridSpec(1, 3)

ax1 = plt.subplot(gs[0, 0])
ax1.plot(_, FF_gap_2B, "-r", label="2B GAP")
ax1.plot(_, FF_gap_soap, "-", color="blue", label="SOAP GAP")
ax1.plot(_, FF_gap_soap_2B, "-", color="magenta", label="2B SOAP GAP")
ax1.plot(_, FF_gap_soap_avgE0, "-", color="cyan", label="SOAP GAP avg E0")
ax1.set_xlabel("Separation")
ax1.set_ylabel("Energy / eV")
ax1.set_title("F-F dimer")
#ax1.set_ylim(-75, -8)

ax2 = plt.subplot(gs[0, 1])
ax2.plot(_, AlF_gap_2B, "-r", label="2B GAP")
ax2.plot(_, AlF_gap_soap, "-", color="blue", label="SOAP GAP")
ax2.plot(_, AlF_gap_soap_2B, "-", color="magenta", label="2B SOAP GAP")
ax2.plot(_, AlF_gap_soap_avgE0, "-", color="cyan", label="SOAP GAP avg E0")
ax2.set_xlabel("Separation")
ax2.set_ylabel("Energy / eV")
ax2.set_title("Al-F dimer")
#ax2.set_ylim(-72, -30)
ax2.legend()

ax3 = plt.subplot(gs[0, 2])
ax3.plot(_, AlAl_gap_2B, "-r", label="2B GAP")
ax3.plot(_, AlAl_gap_soap, "-", color="blue", label="SOAP GAP")
ax3.plot(_, AlAl_gap_soap_2B, "-", color="magenta", label="2B SOAP GAP")
ax3.plot(_, AlAl_gap_soap_avgE0, "-", color="cyan", label="SOAP GAP avg E0")
ax3.set_xlabel("Separation")
#ax3.set_ylabel("Energy / eV")
ax3.set_title("Al-Al dimer")
#ax3.set_ylim(-115, -30)

plt.tight_layout()
plt.savefig('dimer_curve.png')



#################################################
# Test the GAP potential on the unseen test set #
#################################################
"""



############################################################################################
# Show histrogram of the energy distribution of ALL 800_2000 (three) and 800_3000 (bigger) #
############################################################################################
# N.B. three GAP SOAP --> three, big, eigvec712 
print()
print()
print("Even larger 'breathing' configurations")

bigger = os.path.join(abspath, 'collect_100samples/GAP_0-0_10_1-1_y_3.0_100_800-3000/FIT')
big_read = read(os.path.join(bigger, 'Training_set.xyz'), ":")

#three_read = read(three_train, ":")

energies_three = []
for i, at in enumerate(three_read):
    energies_three.append(at.get_potential_energy())

energies_bigger = []
for i, at in enumerate(big_read):
    energies_bigger.append(at.get_potential_energy())

# plot no of dataset vs pot E (800-2000, 800-3000)
fig = plt.figure(figsize=(9, 4.2))
gs = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(gs[0, 0])
ax1.hist(energies_three, bins=30)
ax1.set_xlabel("Potential energy / eV")
ax1.set_ylabel("Count in dataset")
ax1.set_title("Universal set")

ax2 = plt.subplot(gs[0, 1], sharex=ax1)
ax2.hist(energies_bigger, bins=30)
ax2.set_xlabel("Potential energy / eV")
ax2.set_title("800-3000 (bigger)")

plt.tight_layout()



#################################################################
# SCATTER PLOT OF THE DATA TRAIN on the left, test on the right #
#################################################################
if 'train.xyz' in os.listdir(bigger):
   os.remove(os.path.join(bigger, 'train.xyz'))
else:
    pass
if 'test.xyz' in os.listdir(bigger):
    os.remove(os.path.join(bigger, 'test.xyz'))

# Now create training set and a test set
for i, at in enumerate(big_read):
    at.info.pop("energy")
    if 250 < i < 399:
        write(os.path.join(bigger, 'test.xyz'), at, append=True)


# test GAP trained on 800-1500 on the [universal_test set] and [800-3000 test set]
calculator = Potential(param_filename=os.path.join(three, 'GAP_soap.xml'))
big_test = os.path.join(bigger, 'test.xyz')

print()
print("Training (800-2000) set")
_, fthree = get_mae(three_train, calculator)
print()
print("Test (800-3000) set")
_, fbig_test = get_mae(big_test, calculator)
print()
print("Test (7~12 eigvec, no breathing) set")
_, funiversal_test = get_mae(universal_test, calculator)

three_read = read(three_train, ":")
big_test_read = read(big_test, ':')
universal_test_read = read(universal_test, ':') 

gulp_fs_test = []
gulp_fs_bigger = []

gap_fs_test = []
gap_fs_bigger = []

gulp_fs_universal_test = []
gap_fs_universal_test =[]

for a in three_read:
    gulp_fs_test.append(a.get_forces())
    a.set_calculator(calculator)
    gap_fs_test.append(a.get_forces())

for b in big_test_read:
    gulp_fs_bigger.append(b.get_forces())
    b.set_calculator(calculator)
    gap_fs_bigger.append(b.get_forces())

for c in universal_test_read:
    gulp_fs_universal_test.append(c.get_forces())
    c.set_calculator(calculator)
    gap_fs_universal_test.append(c.get_forces())


gulp_fs_test = gulp_fs_test[:-2]
gulp_fs_bigger = gulp_fs_bigger[:-2]

gap_fs_test = gap_fs_test[:-2]
gap_fs_bigger = gap_fs_bigger[:-2]

gulp_fs_universal_test = gulp_fs_universal_test[:-2]
gap_fs_universal_test = gap_fs_universal_test[:-2]


min_bigger = np.min(np.array(gulp_fs_bigger)) - 0.5
max_bigger = np.max(np.array(gulp_fs_bigger)) + 0.5

min_universal_tset = np.min(np.array(gulp_fs_universal_test)) - 0.5
max_universal_test = np.min(np.array(gap_fs_universal_test)) + 0.5


fig = plt.figure(figsize=(15, 4.5))
gs = gridspec.GridSpec(1, 3)

ax1 = plt.subplot(gs[0, 0])
ax1.scatter(gulp_fs_test, gap_fs_test, label=f"MAE = {round(fthree, 4)} meV / A, {len(gap_fs_test)} points")
ax1.plot([min_bigger, max_bigger], [min_bigger, max_bigger], "-r")
ax1.set_xlabel("gulp force / eV / A")
ax1.set_ylabel("SOAP GAP force / eV / A")
ax1.set_title("Training set (800-2000)")
ax1.legend()

ax2 = plt.subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax2.scatter(gulp_fs_bigger, gap_fs_bigger, label=f"MAE = {round(fbig_test, 4)} meV / A, {len(gap_fs_bigger)} points")
ax2.plot([min_bigger, max_bigger], [min_bigger, max_bigger], "-r")
ax2.set_xlabel("gulp force / eV / A")
ax2.set_ylabel("SOAP GAP force / eV / A")
ax2.set_title("Test set (800-3000)")
ax2.legend()


ax3 = plt.subplot(gs[0, 2], sharex=ax1, sharey=ax1)
ax3.scatter(gulp_fs_universal_test, gap_fs_universal_test, label=f"MAE = {round(funiversal_test, 4)} meV / A, {len(gap_fs_universal_test)} points")
ax3.plot([min_bigger, max_bigger], [min_bigger, max_bigger], "-r")
ax3.set_xlabel("gulp force / eV / A")
ax3.set_ylabel("SOAP GAP force / eV / A")
ax3.set_title("Universal test set (7~12 eigvec, no breathing)")
ax3.legend()

plt.tight_layout()


comb_train = three_read + universal_test_read
write("train_mixed_three_universal.xyz", comb_train)
comb_train = os.path.join(abspath ,"train_mixed_three_universal.xyz")

#with open(comb_train, 'a') as f:
#    f.write('1\n')
#    f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
#Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 free_energy=0.00000 pbc="F F F"\n')
#    f.write('Al 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n')
#    f.write('1\n')
#    f.write(f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" \
#Properties=species:S:1:pos:R:3:forces:R:3 energy=0.000000000000 free_energy=0.00000 pbc="F F F"\n')
#    f.write('F 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000\n')

os.system('/scratch/home/uccatka/virtualEnv/bin/gap_fit \
atoms_filename=%s \
gap={distance_2b \
cutoff=3.0 \
n_sparse=100 \
covariance_type=ard_se \
delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
add_species=T : \
soap l_max=4 \
n_max=8 \
atom_sigma=0.5 \
cutoff=3.5 \
radial_scaling=-0.5 \
cutoff_transition_width=1.0 \
central_weight=1.0 \
n_sparse=150 \
delta=0.1 \
covariance_type=dot_product \
zeta=2 \
sparse_method=cur_points} \
config_type_kernel_regularisation=\
{isolated_atom:0.0001:0.0:0.0:0.0} default_sigma={0.002 0.02 0 0} \
energy_parameter_name=energy \
force_parameter_name=forces \
sparse_jitter=1.0e-8 \
do_copy_at_file=F \
sparse_separate_file=T \
gp_file=%s' % ('train_mixed_three_universal.xyz', 'comb_train.xml'))


if 'train.xyz' in os.listdir(three):
   os.remove(os.path.join(three, 'train.xyz'))
else:
    pass
if 'test.xyz' in os.listdir(three):
    os.remove(os.path.join(three, 'test.xyz'))

# Now create training set and a test set
for i, at in enumerate(three_read):
    at.info.pop("energy")
    if 100 < i < 199:
        write(os.path.join(three, 'test.xyz'), at, append=True)

three_test = os.path.join(three, 'test.xyz')
eigvec1112 = os.path.join(abspath, 'GAP_11-12_10_1-1_n_3.0_100/FIT')
# How did the test set errors change? 
calculator = Potential(param_filename=os.path.join(abspath, comb_train))
print("800-1500 (two) training set")
_, train_ftwo = get_mae(two_train, calculator)
print("7~9 eigvec training set")
_, train_funiversal = get_mae(universal_test, calculator)
print("three test set")
_, test_fthree = get_mae(three_test, calculator)
print("11~12 eigvec test set")
_, test_feigvec1112 = get_mae(eigvec1112, calculator)


train_two_read = two_read
train_universal_read = read(universal_test, ":")
test_three_read = read(three_test, ":")
test_eigvec1112_read = read(eigvec1112, ":")

calculator = Potential(param_filename=os.path.join(abspath, comb_train))

gulp_fs_two_train = []
gulp_fs_universal_tr = []
gap_fstwo_train = []
gap_fsuniversal_tr = []
gulp_fs_three_te = []
gulp_fs_eigvec1112 = []
gap_fsthree_te = []
gap_fseigve1112_te = []

for at in train_two_read:
    gulp_fs_two_train.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fstwo_train.append(at.get_forces())

for at in train_universal_read:
    gulp_fs_universal_tr.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fsuniversal_tr.append(at.get_forces())

for at in test_three_read:
    gulp_fs_three_te.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fsthree_te.append(at.get_forces())

for at in test_eigvec1112_read:
    gulp_fs_eigvec1112_te.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fseigvec1112_te.append(at.get_forces())

min900 = np.min(np.array(gulp_fs_eigvec1112_te)) - 0.6
max900 = np.max(np.array(gulp_fs_eigvec1112_te)) + 0.6

fig = plt.figure(figsize=(10.5, 5.5))
gs = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(gs[0, 0])
ax1.scatter(gulp_fs_three_te, gap_fsthree_te, c="orange", s=7, label="Test MAE = {0:.1f} meV / A".format(test_f300))
ax1.scatter(gulp_fs_two_train, gap_fstwo_train, c="blue", s=5, label="Train MAE = {0:.1f} meV / A".format(train_f300))
ax1.plot([min900, max900], [min900, max900], "-k")
ax1.set_xlabel("gulp force / eV / A")
ax1.set_ylabel("SOAP GAP force / eV / A")
ax1.set_title("breathing")
ax1.legend(fontsize=13)

ax2 = plt.subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax2.scatter(gulp_fs_eigvec1112, gap_fseigvec1112, c="orange", s=7, label="Test MAE = {0:.1f} meV / A".format(test_f900))
ax2.scatter(gulp_fs_universal_tr, gap_fsuniversal_tr, c="blue", s=5, label="Train MAE = {0:.1f} meV / A".format(train_f900))
ax2.plot([min900, max900], [min900, max900], "-k")
ax2.set_xlabel("gulp force / eV / A")
#ax2.set_ylabel("SOAP GAP force / eV / A")
ax2.set_title("eigvec")
ax2.legend(fontsize=13)

plt.tight_layout()


plt.show()






