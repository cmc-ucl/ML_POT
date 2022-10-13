import os, sys
from colored import fg, bg, attr

Arg = sys.argv
EIGVEC = Arg[1]
STEP = int(Arg[2])
RANK_from = int(Arg[3])
RANK_to = int(Arg[4])
Breath = Arg[5]
cutoff= float(Arg[6])
sparse = int(Arg[7])

EIGVEC = EIGVEC.split()
_EIGVEC = '-'.join(EIGVEC)

wd_name = f'GAP_{_EIGVEC}_{STEP}_{RANK_from}-{RANK_to}_{Breath}_{cutoff}_{sparse}'


print(f'{fg(15)} {bg(5)} Training GAP {attr(0)}')
print()
print()

os.system('/scratch/home/uccatka/virtualEnv/bin/gap_fit \
energy_parameter_name=energy \
force_parameter_name=forces \
do_copy_at_file=F \
sparse_separate_file=F \
gp_file=%s/FIT/GAP.xml \
at_file=%s/FIT/Training_set.xyz \
default_sigma={0.008 0.04 0 0} \
sparse_jitter=1.0e-8 \
gap={distance_2b \
cutoff=%s \
covariance_type=ard_se delta=0.5 \
theta_uniform=1.0 \
sparse_method=uniform \
n_sparse=%s}' % (wd_name, wd_name, cutoff, sparse) )

print()
print(f"{fg(5)} {bg(15)} The GAP fitting has finished {attr(0)}")
print()
os.system("/scratch/home/uccatka/virtualEnv/bin/quip E=T F=T atoms_filename=%s/FIT/Training_set.xyz param_filename=%s/FIT/GAP.xml | grep AT | sed 's/AT//' > %s/FIT/quip_train.xyz" % (wd_name, wd_name, wd_name))
os.system("/scratch/home/uccatka/virtualEnv/bin/quip E=T F=T atoms_filename=%s/FIT/Valid_set.xyz param_filename=%s/FIT/GAP.xml | grep AT | sed 's/AT//' > %s/FIT/quip_validate.xyz" % (wd_name, wd_name, wd_name))







