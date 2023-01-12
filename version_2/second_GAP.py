""" """

import os
import sys
import shutil
from colored import fg, bg, attr
import GULP

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

columns = shutil.get_terminal_size().columns
print()
print(f"{fg(15)} {bg(39)} Training GAP {attr(0)}".center(columns))
GULP = GULP.GULP(step, vib, SP='set')
GULP.GAP_2b_fit(wd_name, cutoff, sparse)
print()
print(f"{fg(15)} {bg(39)} Training finished {attr(0)}".center(columns))


