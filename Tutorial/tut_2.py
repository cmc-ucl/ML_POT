import os, shutil
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

three_test = os.path.join(three, 'test.xyz')
three_test_read = read(three_test)

eigvec7_9 = os.path.join(abspath, 'GAP_7-9_10_1-1_n_3.0_100/FIT/Training_set.xyz')
eigvec7_9_read = read(eigvec7_9, ":")

eigvec11_12 = os.path.join(abspath, 'GAP_11-12_10_1-1_n_3.0_100/FIT/Training_set.xyz')
eigvec11_12_read = read(eigvec11_12, ":")



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

# Now create three unseen data for test 
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
print("Test (7~9 eigvec, no breathing) set")
_, feigvec7_9 = get_mae(eigvec7_9, calculator)
print()
print("Test (11~12 eigvec, no breathing) set")
_, feigvec11_12 = get_mae(eigvec11_12, calculator)

big_test_read = read(big_test, ':')

gulp_fs_test = []
gulp_fs_bigger = []

gap_fs_test = []
gap_fs_bigger = []

gulp_fs_eigvec7_9 = []
gap_fs_eigvec7_9 = []

gulp_fs_eigvec11_12 = []
gap_fs_eigvec11_12 = []

for a in three_read:
    gulp_fs_test.append(a.get_forces())
    a.set_calculator(calculator)
    gap_fs_test.append(a.get_forces())

for b in big_test_read:
    gulp_fs_bigger.append(b.get_forces())
    b.set_calculator(calculator)
    gap_fs_bigger.append(b.get_forces())

for c in eigvec7_9_read:
    gulp_fs_eigvec7_9.append(c.get_forces())
    c.set_calculator(calculator)
    gap_fs_eigvec7_9.append(c.get_forces())

for d in eigvec11_12_read:
    gulp_fs_eigvec11_12.append(d.get_forces())
    d.set_calculator(calculator)
    gap_fs_eigvec11_12.append(d.get_forces())

gulp_fs_test = gulp_fs_test[:-2]
gap_fs_test = gap_fs_test[:-2]

gulp_fs_bigger = gulp_fs_bigger[:-2]
gap_fs_bigger = gap_fs_bigger[:-2]

gulp_fs_eigvec7_9 = gulp_fs_eigvec7_9[:-2]
gap_fs_eigvec7_9 = gap_fs_eigvec7_9[:-2]

gulp_fs_eigvec11_12 = gulp_fs_eigvec11_12[:-2]
gap_fs_eigvec11_12 = gap_fs_eigvec11_12[:-2]

min_bigger = np.min(np.array(gulp_fs_bigger)) - 0.5
max_bigger = np.max(np.array(gulp_fs_bigger)) + 0.5

min_eigvec7_9 = np.min(np.array(gulp_fs_eigvec7_9)) - 0.5
max_eigvec7_9 = np.min(np.array(gulp_fs_eigvec7_9)) + 0.5

min_eigvec11_12 = np.min(np.array(gulp_fs_eigvec11_12)) - 0.5
max_eigvec11_12 = np.min(np.array(gulp_fs_eigvec11_12)) + 0.5

fig = plt.figure(figsize=(15, 4.5))
gs = gridspec.GridSpec(1, 4)

ax1 = plt.subplot(gs[0, 0])
ax1.scatter(gulp_fs_test, gap_fs_test, label=f"MAE = {round(fthree, 1)} meV / A, {len(gap_fs_test)} points")
ax1.plot([min_bigger, max_bigger], [min_bigger, max_bigger], "-r")
ax1.set_xlabel("gulp force / eV / A")
ax1.set_ylabel("SOAP GAP force / eV / A")
ax1.set_title("Test on the 800-2000 (three) training data set")
ax1.legend()

ax2 = plt.subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax2.scatter(gulp_fs_bigger, gap_fs_bigger, label=f"MAE = {round(fbig_test, 1)} meV / A, {len(gap_fs_bigger)} points")
#ax2.plot([min_bigger, max_bigger], [min_bigger, max_bigger], "-r")
ax2.set_xlabel("gulp force / eV / A")
ax2.set_ylabel("SOAP GAP force / eV / A")
ax2.set_title("Test on 800-3000 (big) configuration")
ax2.legend()

ax3 = plt.subplot(gs[0, 2], sharex=ax1, sharey=ax1)
ax3.scatter(gulp_fs_eigvec7_9, gap_fs_eigvec7_9, label=f"MAE = {round(feigvec7_9, 1)} meV / A, {len(gap_fs_eigvec7_9)} points")
#ax3.plot([min_eigvec7_9, max_eigvec7_9], [min_bigger, max_bigger], "-r")
ax3.set_xlabel("gulp force / eV / A")
ax3.set_ylabel("SOAP GAP force / eV / A")
ax3.set_title("Test on 7~9 eigvec dataset")
ax3.legend()

ax4 = plt.subplot(gs[0, 3], sharex=ax1, sharey=ax1)
ax4.scatter(gulp_fs_eigvec11_12, gap_fs_eigvec11_12, label=f"MAE = {round(feigvec11_12, 1)} meV / A, {len(gap_fs_eigvec11_12)} points")
#ax4.plot([min_eigvec11_12, max_eigvec11_12], [min_bigger, max_bigger], "-r")
ax4.set_xlabel("gulp force / eV / A")
ax4.set_ylabel("SOAP GAP force / eV / A")
ax4.set_title("Test on 11~12 eigvec dataset")
ax4.legend()

plt.tight_layout()
'''
if 'tutorial_2' in os.listdir('./'):
    shutil.rmtree('tutorial_2')
    os.mkdir('tutorial_2') 
else:
    os.mkdir('tutorial_2')

with open(two_train, 'r') as f:
    lines_two = f.readlines()[:-6]

with open(eigvec11_12, 'r') as f:
    lines_eigvec11_12 = f.readlines()

with open('tutorial_2/train_mixed.xyz', 'a') as f:
    for i in lines_two:
        f.write(i)
    for j in lines_eigvec11_12:
        f.write(j)

    

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
gp_file=%s' % ('tutorial_2/train_mixed.xyz', 'tutorial_2/mixed.xml'))
'''


# How did the test set errors change?
calculator = Potential(param_filename=os.path.join(abspath, 'tutorial_2/mixed.xml'))
print()
print("Training (800-1500) set")
_, train_ftwo = get_mae(two_train, calculator)
print()
print("Training (7-9 eigvec, no breathing) set")
_, train_feigvec7_9 = get_mae(eigvec11_12, calculator)
print()
print("Test (800-2000) set")
_, test_fthree = get_mae(three_test, calculator)
print()
print("Training (11-12 eigvec. no breathing) set")
_, test_feigvec11_12 = get_mae(eigvec7_9, calculator)


calculator = Potential(param_filename=os.path.join(abspath, 'tutorial_2/mixed.xml'))


gulp_fs_two_train = []
gap_fstwo_train = []

gulp_fs_three_te = []
gap_fsthree_te = []

gulp_fs_eigvec7_9_tr = []
gap_fseigvec7_9_tr = []

gulp_fs_eigvec11_12_te = []
gap_fseigvec11_12_te = []

for at in two_read:
    gulp_fs_two_train.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fstwo_train.append(at.get_forces())

for at in eigvec7_9_read:
    gulp_fs_eigvec7_9_tr.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fseigvec7_9_tr.append(at.get_forces())

for at in three_read:
    gulp_fs_three_te.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fsthree_te.append(at.get_forces())

for at in eigvec11_12_read:
    gulp_fs_eigvec11_12_te.append(at.get_forces())
    at.set_calculator(calculator)
    gap_fseigvec11_12_te.append(at.get_forces())

gulp_fs_two_train = gulp_fs_two_train[:-2] 
gap_fstwo_train = gap_fstwo_train[:-2] 

gulp_fs_three_te = gulp_fs_three_te[:-2] 
gap_fsthree_te = gap_fsthree_te[:-2] 

gulp_fs_eigvec7_9_tr = gulp_fs_eigvec7_9_tr[:-2] 
gap_fseigvec7_9_tr = gap_fseigvec7_9_tr[:-2] 

gulp_fs_eigvec11_12_te = gulp_fs_eigvec11_12_te[:-2]
gap_fseigvec11_12_te = gap_fseigvec11_12_te[:-2]

min_ = np.min(np.array(gulp_fs_eigvec11_12_te)) - 0.6
max_ = np.max(np.array(gulp_fs_eigvec11_12_te)) + 0.6

fig = plt.figure(figsize=(10.5, 5.5))
gs = gridspec.GridSpec(1, 2)

ax1 = plt.subplot(gs[0, 0])
ax1.scatter(gulp_fs_three_te, gap_fsthree_te, c="orange", s=7, label="Test MAE = {0:.1f} meV / A".format(test_fthree))
ax1.scatter(gulp_fs_two_train, gap_fstwo_train, c="blue", s=5, label="Train MAE = {0:.1f} meV / A".format(train_ftwo))
ax1.plot([min_, max_], [min_, max_], "-k")
ax1.set_xlabel("gulp force / eV / A")
ax1.set_ylabel("SOAP GAP force / eV / A")
ax1.set_title("Breathing configurations")
ax1.legend(fontsize=13)

ax2 = plt.subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax2.scatter(gulp_fs_eigvec11_12_te, gap_fseigvec11_12_te, c="orange", s=7, label="Test MAE = {0:.1f} meV / A".format(test_feigvec11_12))
ax2.scatter(gulp_fs_eigvec7_9_tr, gap_fseigvec7_9_tr, c="blue", s=5, label="Train MAE = {0:.1f} meV / A".format(train_feigvec7_9))
ax2.plot([min_, max_], [min_, max_], "-k")
ax2.set_xlabel("gulp force / eV / A")
#ax2.set_ylabel("SOAP GAP force / eV / A")
ax2.set_title("Vibrational modes implemented configurations")
ax2.legend(fontsize=13)

plt.tight_layout()


plt.show()




