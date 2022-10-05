import os, sys

FROM = sys.argv[1]
TO = int(sys.argv[2])
STEP = int(sys.argv[3])
FROM_rank = int(sys.argv[4])
TO_rank = int(sys.argv[5])
Breath = sys.argv[6]
cutoff = float(sys.argv[7])
sparse = int(sys.argv[8])

os.system(f'python /home/uccatka/auto/for_GAP/GAP_1_gulp.py {FROM} {TO} {STEP} {FROM_rank} {TO_rank} {Breath} {cutoff} {sparse}')
print('\n'*10)
os.system(f'python /home/uccatka/auto/for_GAP/GAP_2_fit.py {FROM} {TO} {STEP} {FROM_rank} {TO_rank} {Breath} {cutoff} {sparse}') 
print('\n'*10)
os.system(f'python /home/uccatka/auto/for_GAP/GAP_3_vis.py {FROM} {TO} {STEP} {FROM_rank} {TO_rank} {Breath} {cutoff} {sparse}')
print('\n'*10)
