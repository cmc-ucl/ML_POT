import sys
import os
import shutil
import numpy as np
from colored import fg, bg, attr

standard = 'fit conp'
relax = 'relax fit conp'
#executable = '/home/uccatka/software/gulp-5.1/Src/gulp' 
executable = '/opt/openmpi/gfortran/1.8.4/bin/mpirun -n 1 /usr/local/gulp/gulp-4.2.mpi'

cwd = os.getcwd()
in_dir = cwd + '/master'

try: 
    os.mkdir('storage')
except FileExistsError:
    print()
    print(f'{fg(1)} {bg(15)} [storage] directory already exists! {attr(0)}')
    print(f'I presume you have finished the IP fitting calculations so I will collect parameters and\
SOS in {fg(1)} {bg(15)} Data.csv {attr(0)}')
    print()
    permission = input("Would you like to continue to collect data? : ").lower()
    if 'y' in permission:
        os.system('python /home/uccatka/auto/IP_collecting_Datacsv.py')
        sys.exit()
    else:
        sys.exit()
else:
    print(f'{fg(1)} {bg(15)} [storage] {attr(0)} directory is generated for the parameter fitting!')
out_dir = cwd + '/storage'


#
# standard fit
#

# Preparing paramters 
input_f = []
for A in np.linspace(0, 10000, num=500):
    for Rho in np.linspace(0., 0.35, num=500):
        A = A.round(4)
        Rho = Rho.round(4)
        input_f.append(f'{A}_{Rho}')
        output_f = f'{A}_{Rho}.fit'

# Preparing [1]job locations, [2]submission scripts, [3]input files ([3-1]standard, [3-2]relax)
#[1]
groups = [input_f[x: x+500] for x in range(0, len(input_f), 500)]
for num, g in enumerate(groups):
    try:
        os.mkdir(f'{out_dir}/{num}')
    except FileExistsError:
        sys.exit(f'{out_dir}/{num} already exists!\n')
    else:
        print(f'{out_dir}/{num} is genereated for the jobs!\n')

    #[2]
    with open(f'{in_dir}/go.sh', 'r') as f:
        job = f.read().replace('JOB_N', f'troop_{num}')
        with open(f'{out_dir}/{num}/go.sh', 'w') as f:
            f.write(job)

    #[3]
    for i in g: 
        A = float(i.split('_')[0])               #.split('.gin')[0])
        Rho = float(i.split('_')[1].split('.gin')[0])

        with open(f'{in_dir}/gulp.gin', 'r') as f:
            contents = f.read()

            #[3-1]
            new_contents_stand = contents.replace('KEYWORDS', standard)
            new_contents_stand = new_contents_stand.replace('JED1', str(A))
            new_contents_stand = new_contents_stand.replace('JED2', str(Rho))

            with open(f'{out_dir}/{num}/{i}.in1', 'w') as f:
                f.write(new_contents_stand)

            #[3-2]
            new_contents_relax = contents.replace('KEYWORDS', relax)
            new_contents_relax = new_contents_relax.replace('JED1', str(A))
            new_contents_relax = new_contents_relax.replace('JED2', str(Rho))

            with open(f'{out_dir}/{num}/{i}.in2', 'w') as f:
                f.write(new_contents_relax)

        #[2]
        with open(f'{out_dir}/{num}/go.sh', 'a') as f:
            #j = i.replace('in1', 'fit')
            f.write(f'{executable} < {out_dir}/{num}/{i}.in1 > {out_dir}/{num}/{i}.fit\n')
            f.write(f'{executable} < {out_dir}/{num}/{i}.in2 > {out_dir}/{num}/{i}.relax\n')
            
    
    os.chdir(f'{out_dir}/{num}') 
    os.system(f'qsub {out_dir}/{num}/go.sh')
    os.chdir('../../')






