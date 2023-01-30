""" """

import sys
import shutil
from colored import fg, bg, attr
import GULP




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
short_filter = arg[8]
energy_filter = arg[9]
if len(arg) == 11:
    DEBUG = arg[10]
else:
    DEBUG = "n"

# Visualising distribution of training data with interatomic distance
binwidth = 0.05
sig2 = 0.005

vib = vib.split()
_vib = "-".join(vib)

wd_name = f"GAP_{_vib}_{step}_{rank_from}-{rank_to}_{cutoff}_{sparse}"

columns = shutil.get_terminal_size().columns
print()
print(f"{fg(15)} {bg(124)} Visualisation {attr(0)}".center(columns))

GULP = GULP.GULP(step, vib, SP='set')
wd_path, FIT_path, Train_xyz_path = GULP.VIS_ESSENTIAL(wd_name)
df_dimer, x_axis = GULP.DIMER_GAP_CALC(FIT_path)
all_het_dist, all_homo_dist = GULP.DIST_BIN_CALC(wd_path, FIT_path, Train_xyz_path, binwidth, sig2)
GULP.PLOT_DIMER(df_dimer, wd_name, FIT_path, x_axis, all_het_dist, all_homo_dist)

print(f"{fg(15)} {bg(124)} The plot (./{wd_name}/plot.html) is saved -- Good luck! {attr(0)}".center(columns))


