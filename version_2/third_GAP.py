""" """

import os
import sys
import random
from colored import fg, bg, attr
import pandas as pd
from tqdm import tqdm
import GULP

import pandas as pd
import numpy as np



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

print()
print(f"{fg(15)} {bg(5)} Visualisation {attr(0)}")

binwidth = 0.05
sig2 = 0.005

GULP = GULP.GULP(step, vib, SP='set')

wd_path, FIT_path, Train_xyz_path = GULP.VIS_ESSENTIAL(wd_name)
df_dimer, x_axis = GULP.DIMER_GAP_CALC(FIT_path)
all_het_dist, all_homo_dist = GULP.DIST_BIN_CALC(wd_path, FIT_path, Train_xyz_path, binwidth, sig2)
GULP.PLOT_DIMER(df_dimer, wd_name, FIT_path, x_axis, all_het_dist, all_homo_dist)



