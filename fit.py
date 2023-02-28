
import os
import sys

sys.path.append('/home/uccatka/auto/for_GAP')
import GULP_dist


STEP = True
EIGVEC = True
SP = True
DEBUG = None
GULP = GULP_dist.GULP(STEP, EIGVEC, SP, DEBUG)

fit_dir = sys.argv[1]
binwidth = 0.05
sig2 = 0.005
GULP.GAP_2b_fit(fit_dir, 3.0, 100)
wd_path, FIT_path, Train_xyz_path = GULP.VIS_ESSENTIAL(fit_dir)
df_dimer, x_axis = GULP.DIMER_GAP_CALC(FIT_path)
cat_an_dist, cat_cat_dist, an_an_dist = GULP.DIST_BIN_CALC(fit_dir, FIT_path, Train_xyz_path, binwidth, sig2)
print(cat_an_dist)
print()
print(cat_cat_dist)
print()
print(an_an_dist)
GULP.PLOT_DIMER(df_dimer, fit_dir, FIT_path, x_axis, cat_an_dist, cat_cat_dist, an_an_dist)









