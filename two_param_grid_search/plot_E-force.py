import os
import numpy as np
#import matplotlib
#matplotlib.use('gtk')
import matplotlib.pyplot as plt
from copy import deepcopy as cp

# ase imports
import ase.io
from ase import Atoms, Atom
from ase import units
from ase.build import molecule

####################
def rms_dict(x_ref, x_pred):
    """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""

    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)

    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')

    error_2 = (x_ref - x_pred) ** 2

    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))

    return {'rmse': average, 'std': std_}
##################


def energy_plot(in_file, out_file, ax, title='Plot of energy'):
    """ Plots the distribution of energy per atom on the FIT vs the input"""
    # read files
    in_atoms = ase.io.read(in_file, ':')
    out_atoms = ase.io.read(out_file, ':')
    # list energies
    ener_in = [at.get_potential_energy() / len(at.get_chemical_symbols()) for at in in_atoms]
    ener_out = [at.get_potential_energy() / len(at.get_chemical_symbols()) for at in out_atoms]
    # scatter plot of the data
    ax.scatter(ener_in, ener_out)
    # get the appropriate limits for the plot
    for_limits = np.array(ener_in +ener_out)
    elim = (for_limits.min() - 0.05, for_limits.max() + 0.05)
    ax.set_xlim(elim)
    ax.set_ylim(elim)
    # add line of slope 1 for refrence
    ax.plot(elim, elim, c='k')
    # set labels
    ax.set_ylabel('energy by GAP / eV')
    ax.set_xlabel('energy by FHI-aims / eV')
    #set title
    ax.set_title(title)
    # add text about RMSE
    _rms = rms_dict(ener_in, ener_out)
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/atom'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', horizontalalignment='right',
            verticalalignment='bottom')
    plt.savefig(title)


def force_plot(in_file, out_file, ax, symbol='AlF', title='Plot of force'):
    """ Plots the distribution of firce components per atom on the FIT vs the input
        only plots for the given atom type(s)"""

    in_atoms = ase.io.read(in_file, ':') # Read Atoms object(s) from file.
    out_atoms = ase.io.read(out_file, ':')

    # extract data for only one species
    in_force, out_force = [], []
    for at_in, at_out in zip(in_atoms, out_atoms): # zip function pairs same positioned atom as iterator of tuples
        sym_all = at_in.get_chemical_symbols() # get the symbols
        # add force for each atom
        for j, sym in enumerate(sym_all):
            if sym in symbol:
                in_force.append(at_in.get_forces()[j])
                #out_force.append(at_out.get_forces()[j]) \
                out_force.append(at_out.arrays['force'][j]) # because QUIP and ASE use different names

    # convert to np arrays, much easier to work with
    #in_force = np.array(in_force)
    #out_force = np.array(out_force)


    # scatter plot of the data
    ax.scatter(in_force, out_force)
    # get the appropriate limits for the plot
    for_limits = np.array(in_force + out_force)
    flim = (for_limits.min() - 1, for_limits.max() + 1)
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    # add line of
    ax.plot(flim, flim, c='k')
    # set labels
    ax.set_ylabel('force by GAP / (eV/angstrom)')
    ax.set_xlabel('force by FHI-aims / (eV/angstrom)')
    #set title
    ax.set_title(title)
    # add text about RMSE
    _rms = rms_dict(in_force, out_force)
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/�~E'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', horizontalalignment='right',
            verticalalignment='bottom')
    plt.savefig(title)


FIT = "./FIT"
gap = "./FIT/gap_vs_aims"
if not os.path.exists(FIT):
    os.makedirs(FIT)

if not os.path.exists(gap):
    os.makedirs(gap)

fig, ax_list = plt.subplots(nrows=3, ncols=2, gridspec_kw={'hspace': 0.3})
fig.set_size_inches(15, 20)
ax_list = ax_list.flat[:]

train = './FIT/Training_set.xyz'
#validate = './FIT/validate.xyz'
quip_train = './FIT/quip_train.xyz'
#quip_validate = './FIT/quip_validate.xyz'

energy_plot(train, quip_train, ax_list[0], './FIT/gap_vs_aims/Energy_on_training_data')
#energy_plot(validate, quip_validate, ax_list[1], './FIT/gap_vs_aims/Energy_on_validation_data')
#force_plot(validate, quip_validate, ax_list[2], 'Al', './FIT/gap_vs_aims/Force_on_training_data_Al')
force_plot(train, quip_train, ax_list[3], 'F', './FIT/gap_vs_aims/Force_on_training_data_F')
#force_plot(validate, quip_validate, ax_list[4], 'Al', './FIT/gap_vs_aims/Force_on_validation_data_Al')
#force_plot(validate, quip_validate, ax_list[5], 'F',  './FIT/gap_vs_aims/Force_on_validation_data_F')

# if you wanted to have the same limits on the force plots
#for ax in ax_list[2:]:
#    flim = (-20, 20)
#    ax.set_xlim(flim)
#    ax.set_ylim(flim)

print("aims vs gap == DONE")
