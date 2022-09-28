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
    permission = input("Would you like to continue to collect data? : [(Y)es, (N)o] ").lower()
    if 'y' in permission:
        os.system('python /home/uccatka/auto/IP/IP_collecting_Datacsv.py')
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
job_query = input("Which type of fit would you like to do? [(S)tandard,(R)elax, or (B)oth] : ").lower()

A_start = float(input("FROM which A value do you want to search? : "))
A_end = float(input("TO which A value? : "))
A_points = int(input("How many data points within the area (A) would like to search? : ")) + 1
print()
rho_start = float(input("FROM which rho value do you want to search? : "))
rho_end = float(input("TO which rho?: "))
rho_points = int(input("How many data points within the area (rho) would you like to search? : ")) + 1
 
input_f = []
for A in np.linspace(A_start, A_end, num=A_points):
    for Rho in np.linspace(rho_start, rho_end, num=rho_points):
        A = A.round(4)
        Rho = Rho.round(4)
        input_f.append(f'{A}_{Rho}')
        output_f = f'{A}_{Rho}.fit'

# Preparing [1]job locations, [2]submission scripts, [3]input files ([3-1]standard, [3-2]relax)
#[1]
groups = [input_f[x: x+500] for x in range(0, len(input_f), 500)]   # each directory takes total 100 input files
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

            #if any(x in answer_options for x in ('s', 'b')): 
            if 's' in job_query or 'b' in job_query:
            #[3-1]
                new_contents_stand = contents.replace('KEYWORDS', standard)
                new_contents_stand = new_contents_stand.replace('JED1', str(A))
                new_contents_stand = new_contents_stand.replace('JED2', str(Rho))

                with open(f'{out_dir}/{num}/{i}.in1', 'w') as f:
                    f.write(new_contents_stand)


            #if any(x in answer_options for x in ('r', 'b')): 
            if 'r' in job_query or 'b' in job_query:
                #[3-2]
                new_contents_relax = contents.replace('KEYWORDS', relax)
                new_contents_relax = new_contents_relax.replace('JED1', str(A))
                new_contents_relax = new_contents_relax.replace('JED2', str(Rho))

                with open(f'{out_dir}/{num}/{i}.in2', 'w') as f:
                    f.write(new_contents_relax)

        #[2]
        with open(f'{out_dir}/{num}/go.sh', 'a') as f:
            #j = i.replace('in1', 'fit')
            if 's' in job_query or 'b' in job_query: 
                f.write(f'{executable} < {out_dir}/{num}/{i}.in1 > {out_dir}/{num}/{i}.fit\n')
            if 'r' in job_query or 'b' in job_query: 
                f.write(f'{executable} < {out_dir}/{num}/{i}.in2 > {out_dir}/{num}/{i}.relax\n')
            
    
    os.chdir(f'{out_dir}/{num}') 
    os.system(f'qsub {out_dir}/{num}/go.sh')
    os.chdir('../../')

