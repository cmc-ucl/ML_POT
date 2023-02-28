import os, sys
import shutil
import numpy as np
import tarfile
from datetime import datetime

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

date = datetime.today().strftime('%Y-%m-%d')

for i in np.arange(1.1, 5.1, 0.05):
    cutoff = np.around(i, 2)
    for j in np.arange(5, 105, 5):
        sparse = j
        print()
        print()
        print(cutoff, sparse)
        os.system(f"python /home/uccatka/auto/for_GAP/GULP_GAP.py '7 8 9 10 11 12' 10 1 1 n {cutoff} {sparse} y")



