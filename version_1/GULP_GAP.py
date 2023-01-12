import os, sys

Arg = sys.argv
EIGVEC = Arg[1]
STEP = int(Arg[2])
RANK_from = int(Arg[3])
RANK_to = int(Arg[4])
Breath = Arg[5]
cutoff= float(Arg[6])
sparse = int(Arg[7])
dup_filter = Arg[8]

EIGVEC_list = EIGVEC.split()
EIGVEC_list = [''.join(EIGVEC)]
EIGVEC_list = ['"' + x + '"' for x in EIGVEC_list]
EIGVEC_list = EIGVEC_list[0]


os.system(f"python /home/uccatka/auto/for_GAP/GAP_1_gulp.py {EIGVEC_list} {STEP} {RANK_from} {RANK_to} {Breath} {cutoff} {sparse} {dup_filter}")
print('\n'*10)
os.system(f'python /home/uccatka/auto/for_GAP/GAP_2_fit.py {EIGVEC_list} {STEP} {RANK_from} {RANK_to} {Breath} {cutoff} {sparse}') 
print('\n'*10)
os.system(f'python /home/uccatka/auto/for_GAP/GAP_3_vis_hist.py {EIGVEC_list} {STEP} {RANK_from} {RANK_to} {Breath} {cutoff} {sparse}')
print('\n'*10)
